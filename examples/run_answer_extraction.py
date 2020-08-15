# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Modified version of run_squad.py to project-specific task of answer extraction."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertForVQR, BertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from utils_answer_extraction import (read_instagram_examples, convert_examples_to_features)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForVQR, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    last_eval_loss = float('inf')
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            example_ids, inputs = unpack_batch(args, batch)
            outputs = model(**inputs)

            def list_to_dict(output):
                retval = {}
                keys = ['loss']
                if args.q_relevance: keys += ['q_logits']
                if args.r_relevance: keys += ['r_logits']
                if args.answer_extraction: keys += ['span_logits']
                count = 0
                for key in sorted(keys):
                    retval[key] = output[count]
                    count += 1
                return retval
                
            output = list_to_dict(outputs)
            loss = output['loss']  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            logger.info("Train loss: %f", loss.item())
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        eval_loss = evaluate(args, model, tokenizer)[0]
                        logger.info("Eval loss: %f", eval_loss)
                        tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # Log metrics
                    if args.local_rank == -1 and eval_loss < last_eval_loss:
                        last_eval_loss = eval_loss
                        output_dir = os.path.join(args.output_dir, 'best')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, features, token_mapping = load_and_cache_examples(args, tokenizer, evaluate=True, output_features=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    df = pd.DataFrame()
    total_loss = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            example_indices, inputs = unpack_batch(args, batch)
            outputs = model(**inputs)
            def list_to_dict(output):
                retval = {}
                keys = ['loss']
                if args.q_relevance: keys += ['q_logits']
                if args.r_relevance: keys += ['r_logits']
                if args.answer_extraction: keys += ['span_logits']
                count = 0
                for key in sorted(keys):
                    retval[key] = output[count]
                    count += 1
                return retval
                
            output = list_to_dict(outputs)
            total_loss += output['loss'].mean().item() if type(output['loss']) is not int else output['loss']

            # unpack outputs
            if args.q_relevance:
                q_logits = output['q_logits'].detach().cpu()
                q_relevance_label_ids = inputs['q_relevance_ids'].cpu().numpy()
                #metrics['q_relevance_auroc'] += auroc(q_logits, q_relevance_label_ids)
                #metrics['q_relevance_accuracy'] += accuracy(q_logits.numpy(), q_relevance_label_ids)
            if args.r_relevance:
                r_logits = output['r_logits'].detach().cpu()
                r_relevance_label_ids = inputs['r_relevance_ids'].cpu().numpy()
                #metrics['r_relevance_auroc'] += auroc(r_logits, r_relevance_label_ids)
                #metrics['r_relevance_accuracy'] += accuracy(r_logits.numpy(), r_relevance_label_ids)
            if args.answer_extraction:
                start_logits, end_logits = output['span_logits']
                start_pos_ids = inputs['start_positions'].cpu().numpy()
                end_pos_ids = inputs['end_positions'].cpu().numpy()
                start_logits = start_logits.detach().cpu()
                end_logits = end_logits.detach().cpu()
                start_idx = np.argmax(start_logits, axis=1)
                end_idx = np.argmax(end_logits, axis=1)

            response_starts = torch.LongTensor((inputs['token_type_ids'].shape[0]))
            
            for idx in range(inputs['token_type_ids'].shape[0]):
                response_start = (inputs['token_type_ids'][idx] == 1).nonzero()[0].item()
                response_starts[idx] = response_start
                idx_batch = example_indices.cpu().numpy()
                cols = []
                data = np.array([]).reshape(idx_batch.shape[0], 0)
            if args.q_relevance:
                q_prediction = np.expand_dims(np.argmax(q_logits, axis=1).numpy(), 1)
                data = np.hstack([data, q_logits.numpy(), q_prediction])
                cols.extend(['q_score_0', 'q_score_1', 'q_prediction'])
            if args.r_relevance:
                r_prediction = np.expand_dims(np.argmax(r_logits, axis=1).numpy(), 1)
                data = np.hstack([data, r_logits.numpy(), r_prediction])
                cols.extend(['r_score_0', 'r_score_1', 'r_prediction'])
            if args.answer_extraction:
                start_idx -= response_starts
                end_idx -= response_starts
                data = np.hstack([data, np.expand_dims(start_idx,1), np.expand_dims(end_idx,1)])
                cols.extend(['raw_span_start', 'raw_span_end'])
            batch_df = pd.DataFrame(data, index=idx_batch, columns=cols)
            df = df.append(batch_df)
            
        def convert_span(row):
            tok_to_orig = token_mapping[row.name]
            try:
                if row.raw_span_start <= row.raw_span_end:
                    return (tok_to_orig[int(row.raw_span_start)], tok_to_orig[int(row.raw_span_end)])
                else:
                    return None
            except Exception as e:
                print("Error converting span:", e)
                return None

    if args.q_relevance:
        df['q_prediction'] = df.q_prediction.apply(lambda x: True if x == 0 else False)
        
    if args.r_relevance:
        df['r_prediction'] = df.r_prediction.apply(lambda x: True if x == 0 else False)
        
    if args.answer_extraction:
        df['pred_span'] = df.apply(lambda x: convert_span(x), axis=1)
            
    return total_loss / len(eval_dataloader), df

def unpack_batch(args, batch):
    example_ids = batch[0]
    inputs = {'input_ids'  :   batch[1],
              'attention_mask' :   batch[2],
              'token_type_ids':   batch[3]}

    start_pos_ids, end_pos_ids = None, None
    count = 4
    if args.q_relevance:
        inputs['q_relevance_ids'] = batch[count]
        count += 1
    if args.r_relevance:
        inputs['r_relevance_ids'] = batch[count]
        count += 1
    if args.answer_extraction:
        inputs['start_positions'], inputs['end_positions'] = batch[count:]

    if args.model_type != 'distilbert':
        inputs['token_type_ids'] = None if args.model_type == 'xlm' else inputs['token_type_ids']
    if args.model_type in ['xlnet', 'xlm']:
        inputs.update({'cls_index': batch[5],
                       'p_mask': batch[6]})

    return example_ids, inputs


def load_and_cache_examples(args, tokenizer, evaluate=False, output_features=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.eval_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_features:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_instagram_examples(input_file=input_file)
        features, token_mapping = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    all_example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensors = [all_example_ids, all_input_ids, all_input_mask, all_segment_ids]
    if args.q_relevance:
        # add label ids (question is confused or not)
        tensors.append(torch.tensor([f.q_relevance_label_id for f in features], dtype=torch.long))
    if args.r_relevance:
        # add label ids (response is confused or not)
        tensors.append(torch.tensor([f.r_relevance_label_id for f in features], dtype=torch.long))
    if args.answer_extraction:
        # add start/stop indices of answer
        tensors.append(torch.tensor([f.ans_start_idx for f in features], dtype=torch.long))
        tensors.append(torch.tensor([f.ans_end_idx for f in features], dtype=torch.long))

    if output_features:
        return TensorDataset(*tensors), features, token_mapping
    else:
        return TensorDataset(*tensors)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=False,
                        help=".csv for training")
    parser.add_argument("--eval_file", default=None, type=str, required=False,
                        help=".csv for eval")
    parser.add_argument("--model_type", default='bert', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--q_relevance", action='store_true',
                        help="Whether it will be necessary to handle questions deemed to be irrelevant.")
    parser.add_argument("--r_relevance", action='store_true',
                        help="Whether is will be necessary to handle answers deemed as irrelevant.")
    parser.add_argument("--answer_extraction", action='store_true',
                        help="Whether or not to actually extract the relevant answer.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. ")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, q_relevance=args.q_relevance, r_relevance=args.r_relevance, answer_extraction=args.answer_extraction)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_features=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            print("EVAL")
            print(args.output_dir + '/**/' + WEIGHTS_NAME)
            #checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            #checkpoints = [os.path.join(args.output_dir, WEIGHTS_NAME)]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("WEIGHTS %s", WEIGHTS_NAME)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            loss, df = evaluate(args, model, tokenizer, prefix=global_step)
            logger.info("Eval loss: %d", loss)
            df.to_csv(os.path.join(checkpoint, '{}-eval.csv'.format(global_step)))


if __name__ == "__main__":
    main()
