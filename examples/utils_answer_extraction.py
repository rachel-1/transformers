""" Load custom Instagram dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
import pandas as pd
import numpy as np

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for custom dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self, **kwargs):
        self.example_id = kwargs['index']
        self.question = kwargs['question']
        self.response = kwargs['response_filtered']
        self.q_relevant = kwargs['q_relevant']
        self.r_relevant = kwargs['r_relevant']
        self.span = kwargs['span']
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "id: %s" % (self.example_id)
        s += ", question: %s" % (
            self.question)
        s += ", response: [%s]" % (self.response)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def read_instagram_examples(input_file):
    """Read a .csv file"""
    df = pd.read_csv(input_file)
    df['span'] = df.answer_intersection_span.apply(lambda x: pd.eval(x) if not pd.isna(x) else None)
    examples = []
    for i, row in df.iterrows():
        examples.append(InputExample(**dict(row)))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    token_mapping = {}
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.question)

        tokens_b = None
        if example.response:
            orig_to_tok_start_index = []
            orig_to_tok_end_index = []
            tok_to_orig_index = []
            tokens_b = []
            for (i, token) in enumerate(example.response):
                sub_tokens = tokenizer.tokenize(token)
                if len(sub_tokens) == 0: sub_tokens = [tokenizer.wordpiece_tokenizer.unk_token]
                orig_to_tok_start_index.append(len(tokens_b))
                orig_to_tok_end_index.append(len(tokens_b)+len(sub_tokens)-1)
                for sub_token in sub_tokens:
                    tokens_b.append(sub_token)
                    tok_to_orig_index.append(i)

            if example.span is not None and example.span != '':
                ans_start_idx = orig_to_tok_start_index[example.span[0]]
                ans_end_idx = orig_to_tok_end_index[example.span[1]]
            else:
                # these are just placeholders; their value will be ignored
                ans_start_idx, ans_end_idx = -1, -1

            # Trim tokens_b if necessary
            if len(tokens_a) + len(tokens_b) + len(['[CLS]', '[SEP]', '[SEP]']) > max_seq_length:
                orig_gt = tokens_b[ans_start_idx: ans_end_idx+1] if (example.span is not None) else None
                window_size = max_seq_length - 3 - len(tokens_a)

                # Take a window from the response such that the answer is contained
                if ans_start_idx == -1:
                    # since the -1 was just a placeholder, leaving these values is fine
                    ans_start_idx = np.random.randint(0, len(tokens_b))
                    ans_end_idx = ans_start_idx

                # truncate answer if it is too long
                if ans_end_idx - ans_start_idx + 1 >= window_size:
                    ans_end_idx = ans_start_idx + window_size - 1
                    gt_end = tok_to_orig_index[ans_end_idx]
                else:
                    gt_end = example.span[1] if example.span is not None else None
                    
                # calculate index of start of window
                lower_bound = max(ans_end_idx-window_size+1, 0)
                upper_bound = min(len(tokens_b)-window_size,ans_start_idx)
                window_start = np.random.randint(lower_bound, upper_bound+1)
                tokens_b = tokens_b[window_start:window_start+window_size]
                ans_start_idx -= window_start
                ans_end_idx -= window_start

                tok_to_orig_index = tok_to_orig_index[window_start:window_start+window_size]
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            if example.span is not None:
                ans_start_idx += len(tokens)
                ans_end_idx += len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        q_relevant_label_id = 1 if example.q_relevant else 0
        r_relevant_label_id = 1 if example.r_relevant else 0
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.example_id))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.q_relevant is not None:
                logger.info("q_relevant: %s (id = %d)" % (example.q_relevant, q_relevant_label_id))
            if example.r_relevant is not None:
                logger.info("r_relevant: %s (id = %d)" % (example.r_relevant, r_relevant_label_id))

        if not example.r_relevant:
            ans_start_idx = ans_end_idx = -1

        features.append(
                InputFeatures(example_id=example.example_id,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              q_relevance_label_id=q_relevant_label_id,
                              r_relevance_label_id=r_relevant_label_id,
                              ans_start_idx=ans_start_idx,
                              ans_end_idx=ans_end_idx))
        token_mapping[example.example_id] = tok_to_orig_index
    return features, token_mapping
