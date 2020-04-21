# -*- coding: utf-8 -*-
from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from utils import get_tokenize, get_chat_tokenize, missingdict, Pack
import logging
import os
import re
from nltk.corpus import stopwords
import itertools
from collections import defaultdict
import copy
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
HT = "<hash>"
MEN = "<men>"
URL = "<url>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"
DIG = "<digit>"
HTML_STOPWORDS = ['amp', 'gt', 'lt', 'll', '--']

class CMVCorpus(object):
    logger = logging.getLogger(__name__)
    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_data_size = config.max_data_size
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus, self.test_corpus= self._read_file(os.path.join(self._path))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _process_dialog(self, data):
        new_dialog = []
        all_lens = []
        all_dialog_lens = []

        for raw_dialog in data:
            dialog = {"title": self.tokenize(raw_dialog['title'].lower()),
                      "op": self.tokenize(raw_dialog["content"].lower()), "pos_conv_lst": [], "neg_conv_lst": []}
            for i, turns in enumerate(raw_dialog['comments']):   # for each comment lst
                if turns["win"]:
                    conv_lst = dialog["pos_conv_lst"]
                else:
                    conv_lst = dialog["neg_conv_lst"]
                new_utt_lst = []
                for turn in turns["utt_lst"]:
                    argument = self.tokenize(turn.lower())
                    all_lens.append(len(argument))
                    new_utt_lst.append(argument)
                conv_lst.append(new_utt_lst)
                all_dialog_lens.append(len(new_utt_lst))
            new_dialog.append(dialog)
            # cut for the max data size
            if len(new_dialog) >= self.max_data_size:
                break

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            all_words.append(dialog["op"] + dialog["title"])
            for turns in dialog["pos_conv_lst"] + dialog["neg_conv_lst"]:
                for turn in turns:
                    all_words.append(turn)

        self.vocab_bow = Dictionary(all_words)
        raw_vocab_size = len(self.vocab_bow)
        raw_wc = np.sum(list(self.vocab_bow.dfs.values()))

        # build useless stopwords vocab (e.g, very few words, single ascii words, some punctuation ,."')
        self.vocab_bow.filter_extremes(no_below=10, keep_n=max_vocab_cnt)
        bad_ids = HTML_STOPWORDS + ['cmv']
        self.vocab_bow.filter_tokens(list(map(self.vocab_bow.token2id.get, bad_ids)))
        self.vocab_bow.compactify()
        self.vocab_seq = copy.deepcopy(self.vocab_bow)      # for sequence model
        self.vocab_seq.token2id[self.vocab_seq[0]] = len(self.vocab_seq)
        self.vocab_seq.token2id[PAD] = 0
        self.vocab_seq.token2id[UNK] = len(self.vocab_seq)
        self.vocab_seq.compactify()
        self.pad_wid = self.vocab_seq.token2id.get(PAD)

        len_1_words = list(filter(lambda w: len(w) == 1 and re.match(r"[\x00-\x7f]", w) and
                                            w not in ["[", "]", "$", "?", "!", "\"", "'", "i", "a"] and True or False,
                                  self.vocab_bow.values()))
        self.vocab_bow.filter_tokens(list(map(self.vocab_bow.token2id.get, len_1_words)))
        # some makeup words
        # makeup_lst = [PAD]
        # for w in makeup_lst:
        #     self.vocab_bow.token2id[w] = len(self.vocab_bow)
        # self.vocab_bow.compactify()
        # self.pad_wid = self.vocab_bow.token2id.get(PAD)
        # here we keep stopwords and some meaningful punctuations
        non_stopwords = filter(lambda w: re.match(r"^[\w\d_-]*$", w) and w not in STOPWORDS and True or False, self.vocab_bow.values())
        self.vocab_bow_stopwords = copy.deepcopy(self.vocab_bow)
        self.vocab_bow_stopwords.filter_tokens(map(self.vocab_bow_stopwords.token2id.get, non_stopwords))
        self.vocab_bow_stopwords.compactify()
        self.vocab_bow_non_stopwords = copy.deepcopy(self.vocab_bow)
        self.vocab_bow_non_stopwords.filter_tokens(map(self.vocab_bow_non_stopwords.token2id.get, self.vocab_bow_stopwords.values()))
        self.vocab_bow_non_stopwords.compactify()
        remain_wc = np.sum(list(self.vocab_bow.dfs.values()))
        min_count = np.min(list(self.vocab_bow.dfs.values()))
        # create vocabulary list sorted by count
        print("Load corpus with train size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.test_corpus),
                 raw_vocab_size, len(self.vocab_bow), min_count,
                 1 - float(remain_wc) / raw_wc))

    def _read_file(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return self._process_dialog(data["train"]), self._process_dialog(data["test"])

    def _sent2id_seq(self, sent, vocab):
        return list(filter(lambda x: x is not None, [vocab.token2id.get(t) for t in sent]))

    def _sent2id_bow(self, sent, vocab):
        if sent:
            return vocab.doc2bow(sent)
        else:
            return []

    def _to_id_corpus(self, data, vocab_seq, vocab_bow):
        results = []
        word_cnt = 0
        msg_cnt = 0

        for dialog in data:
            # convert utterance and feature into numeric numbers
            id_dialog = Pack(title=self._sent2id_seq(dialog["title"], vocab_seq),
                             op=self._sent2id_seq(dialog["op"], vocab_seq),
                             pos_conv_seq_lst=[], pos_conv_bow_lst=[], neg_conv_seq_lst=[], neg_conv_bow_lst=[])
            for turns in dialog["pos_conv_lst"]:
                new_turns_bow = []
                new_turns_seq = []
                for turn in turns:
                    id_turn_seq = self._sent2id_seq(turn, vocab_seq)
                    id_turn_bow = self._sent2id_bow(turn, vocab_bow)
                    if id_turn_seq and id_turn_bow:  # filter empty utt
                        new_turns_bow.append(id_turn_bow)
                        new_turns_seq.append(id_turn_seq)
                        word_cnt += len(id_turn_seq)
                        msg_cnt += 1
                if new_turns_seq and new_turns_bow:
                    id_dialog["pos_conv_bow_lst"].append(new_turns_bow)
                    id_dialog["pos_conv_seq_lst"].append(new_turns_seq)
            for turns in dialog["neg_conv_lst"]:
                new_turns_bow = []
                new_turns_seq = []
                for turn in turns:
                    id_turn_seq = self._sent2id_seq(turn, vocab_seq)
                    id_turn_bow = self._sent2id_bow(turn, vocab_bow)
                    if id_turn_seq and id_turn_bow:  # filter empty utt
                        new_turns_bow.append(id_turn_bow)
                        new_turns_seq.append(id_turn_seq)
                        word_cnt += len(id_turn_seq)
                        msg_cnt += 1
                if new_turns_seq and new_turns_bow:
                    id_dialog["neg_conv_bow_lst"].append(new_turns_bow)
                    id_dialog["neg_conv_seq_lst"].append(new_turns_seq)
            if id_dialog.pos_conv_bow_lst and id_dialog.neg_conv_bow_lst:
                results.append(id_dialog)
        print("Load seq with %d msgs, %d words" % (msg_cnt, word_cnt))
        return results, msg_cnt, word_cnt

    def _to_id_corpus_bow(self, data, vocab):
        results = []
        word_cnt = 0
        msg_cnt = 0

        for dialog in data:
            # convert utterance and feature into numeric numbers
            id_dialog = Pack(title=self._sent2id_bow(dialog["title"], vocab),
                             op=self._sent2id_bow(dialog["op"], vocab),
                             pos_conv_bow_lst=[], neg_conv_bow_lst=[])
            for turns in dialog["pos_conv_lst"]:
                new_turns = []
                for turn in turns:
                    id_turn = self._sent2id_bow(turn, vocab)
                    if id_turn:     # filter empty utt
                        new_turns.append(id_turn)
                        word_cnt += np.sum([j for i, j in id_turn])
                        msg_cnt += 1
                if new_turns:
                    id_dialog["pos_conv_bow_lst"].append(new_turns)
            for turns in dialog["neg_conv_lst"]:
                new_turns = []
                for turn in turns:
                    id_turn = self._sent2id_bow(turn, vocab)
                    if id_turn:     # filter empty utt
                        new_turns.append(id_turn)
                        word_cnt += np.sum([j for i, j in id_turn])
                        msg_cnt += 1
                if new_turns:
                    id_dialog["neg_conv_bow_lst"].append(new_turns)
            if id_dialog.pos_conv_bow_lst and id_dialog.neg_conv_bow_lst:
                results.append(id_dialog)
        print("Load bow with %d msgs, %d words" % (msg_cnt, word_cnt))
        return results, msg_cnt, word_cnt

    def get_corpus_bow(self, keep_stopwords=True):
        if keep_stopwords:
            vocab = self.vocab_bow
        else:
            vocab = self.vocab_bow_non_stopwords
        id_train = self._to_id_corpus_bow(self.train_corpus, vocab)
        id_test = self._to_id_corpus_bow(self.test_corpus, vocab)
        return Pack(train=id_train, test=id_test, vocab_size=len(vocab))

    def get_corpus_seq(self):
        vocab = self.vocab_seq

        id_train = self._to_id_corpus_seq(self.train_corpus, vocab)
        id_test = self._to_id_corpus_seq(self.test_corpus, vocab)
        return Pack(train=id_train, test=id_test, vocab_size=len(vocab))

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus, self.vocab_seq, self.vocab_bow)
        id_test = self._to_id_corpus(self.test_corpus, self.vocab_seq, self.vocab_bow)
        # id_valid = self._to_id_corpus(self.valid_corpus, self.vocab_seq, self.vocab_bow)
        return Pack(train=id_train, test=id_test, vocab_size=len(self.vocab_bow))


class CourtCorpus(object):
    logger = logging.getLogger(__name__)
    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_data_size = config.max_data_size
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus, self.test_corpus = self._read_file(os.path.join(self._path))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _process_dialog(self, data):
        new_dialog = []
        all_lens = []
        all_dialog_lens = []

        for raw_dialog in data:
            dialog = {"pos_conv_lst": [], "neg_conv_lst": []}
            for i, turns in enumerate(raw_dialog['case_convs']):   # for each comment lst
                if turns["win"]:
                    conv_lst = dialog["pos_conv_lst"]
                else:
                    conv_lst = dialog["neg_conv_lst"]
                new_utt_lst = []
                for turn in turns["utt_lst"]:
                    argument = self.tokenize(turn.lower())
                    all_lens.append(len(argument))
                    new_utt_lst.append(argument)
                conv_lst.append(new_utt_lst)
                all_dialog_lens.append(len(new_utt_lst))
            new_dialog.append(dialog)
            # cut for the max data size
            if len(new_dialog) >= self.max_data_size:
                break

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turns in dialog["pos_conv_lst"] + dialog["neg_conv_lst"]:
                for turn in turns:
                    all_words.append(turn)

        self.vocab_bow = Dictionary(all_words)
        raw_vocab_size = len(self.vocab_bow)
        raw_wc = np.sum(list(self.vocab_bow.dfs.values()))

        # build useless stopwords vocab (e.g, very few words, single ascii words, some punctuation ,."')
        self.vocab_bow.filter_extremes(no_below=5, keep_n=max_vocab_cnt)
        bad_ids = HTML_STOPWORDS
        self.vocab_bow.filter_tokens(list(map(self.vocab_bow.token2id.get, bad_ids)))
        self.vocab_bow.compactify()
        self.vocab_seq = copy.deepcopy(self.vocab_bow)      # for sequence model
        self.vocab_seq.token2id[self.vocab_seq[0]] = len(self.vocab_seq)
        self.vocab_seq.token2id[PAD] = 0
        self.vocab_seq.token2id[UNK] = len(self.vocab_seq)
        self.vocab_seq.compactify()
        self.pad_wid = self.vocab_seq.token2id.get(PAD)

        len_1_words = list(filter(lambda w: len(w) == 1 and re.match(r"[\x00-\x7f]", w) and
                                            w not in ["[", "]", "$", "?", "!", "\"", "'", "i", "a"] and True or False,
                                  self.vocab_bow.values()))
        self.vocab_bow.filter_tokens(list(map(self.vocab_bow.token2id.get, len_1_words)))
        # some makeup words
        # makeup_lst = [PAD]
        # for w in makeup_lst:
        #     self.vocab_bow.token2id[w] = len(self.vocab_bow)
        # self.vocab_bow.compactify()
        # self.pad_wid = self.vocab_bow.token2id.get(PAD)
        # here we keep stopwords and some meaningful punctuations
        non_stopwords = filter(lambda w: re.match(r"^[\w\d_-]*$", w) and w not in STOPWORDS and True or False, self.vocab_bow.values())
        self.vocab_bow_stopwords = copy.deepcopy(self.vocab_bow)
        self.vocab_bow_stopwords.filter_tokens(map(self.vocab_bow_stopwords.token2id.get, non_stopwords))
        self.vocab_bow_stopwords.compactify()
        self.vocab_bow_non_stopwords = copy.deepcopy(self.vocab_bow)
        self.vocab_bow_non_stopwords.filter_tokens(map(self.vocab_bow_non_stopwords.token2id.get, self.vocab_bow_stopwords.values()))
        self.vocab_bow_non_stopwords.compactify()
        remain_wc = np.sum(list(self.vocab_bow.dfs.values()))
        min_count = np.min(list(self.vocab_bow.dfs.values()))
        # create vocabulary list sorted by count
        print("Load corpus with train size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.test_corpus),
                 raw_vocab_size, len(self.vocab_bow), min_count,
                 1 - float(remain_wc) / raw_wc))

    def _read_file(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return self._process_dialog(data["train"]), self._process_dialog(data["test"])

    def _sent2id_seq(self, sent, vocab):
        return list(filter(lambda x: x is not None, [vocab.token2id.get(t) for t in sent]))

    def _sent2id_bow(self, sent, vocab):
        if sent:
            return vocab.doc2bow(sent)
        else:
            return []

    def _to_id_corpus(self, data, vocab_seq, vocab_bow):
        results = []
        word_cnt = 0
        msg_cnt = 0

        for dialog in data:
            # convert utterance and feature into numeric numbers
            id_dialog = Pack(pos_conv_seq_lst=[], pos_conv_bow_lst=[], neg_conv_seq_lst=[], neg_conv_bow_lst=[])
            for turns in dialog["pos_conv_lst"]:
                new_turns_bow = []
                new_turns_seq = []
                for turn in turns:
                    id_turn_seq = self._sent2id_seq(turn, vocab_seq)
                    id_turn_bow = self._sent2id_bow(turn, vocab_bow)
                    if id_turn_seq and id_turn_bow:  # filter empty utt
                        new_turns_bow.append(id_turn_bow)
                        new_turns_seq.append(id_turn_seq)
                        word_cnt += len(id_turn_seq)
                        msg_cnt += 1
                if new_turns_seq and new_turns_bow:
                    id_dialog["pos_conv_bow_lst"].append(new_turns_bow)
                    id_dialog["pos_conv_seq_lst"].append(new_turns_seq)
            for turns in dialog["neg_conv_lst"]:
                new_turns_bow = []
                new_turns_seq = []
                for turn in turns:
                    id_turn_seq = self._sent2id_seq(turn, vocab_seq)
                    id_turn_bow = self._sent2id_bow(turn, vocab_bow)
                    if id_turn_seq and id_turn_bow:  # filter empty utt
                        new_turns_bow.append(id_turn_bow)
                        new_turns_seq.append(id_turn_seq)
                        word_cnt += len(id_turn_seq)
                        msg_cnt += 1
                if new_turns_seq and new_turns_bow:
                    id_dialog["neg_conv_bow_lst"].append(new_turns_bow)
                    id_dialog["neg_conv_seq_lst"].append(new_turns_seq)
            if id_dialog.pos_conv_bow_lst and id_dialog.neg_conv_bow_lst:
                results.append(id_dialog)
        print("Load seq with %d msgs, %d words" % (msg_cnt, word_cnt))
        return results, msg_cnt, word_cnt

    def _to_id_corpus_bow(self, data, vocab):
        results = []
        word_cnt = 0
        msg_cnt = 0

        for dialog in data:
            # convert utterance and feature into numeric numbers
            id_dialog = Pack(pos_conv_bow_lst=[], neg_conv_bow_lst=[])
            for turns in dialog["pos_conv_lst"]:
                new_turns = []
                for turn in turns:
                    id_turn = self._sent2id_bow(turn, vocab)
                    if id_turn:     # filter empty utt
                        new_turns.append(id_turn)
                        word_cnt += np.sum([j for i, j in id_turn])
                        msg_cnt += 1
                if new_turns:
                    id_dialog["pos_conv_bow_lst"].append(new_turns)
            for turns in dialog["neg_conv_lst"]:
                new_turns = []
                for turn in turns:
                    id_turn = self._sent2id_bow(turn, vocab)
                    if id_turn:     # filter empty utt
                        new_turns.append(id_turn)
                        word_cnt += np.sum([j for i, j in id_turn])
                        msg_cnt += 1
                if new_turns:
                    id_dialog["neg_conv_bow_lst"].append(new_turns)
            if id_dialog.pos_conv_bow_lst and id_dialog.neg_conv_bow_lst:
                results.append(id_dialog)
        print("Load bow with %d msgs, %d words" % (msg_cnt, word_cnt))
        return results, msg_cnt, word_cnt

    def get_corpus_bow(self, keep_stopwords=True):
        if keep_stopwords:
            vocab = self.vocab_bow
        else:
            vocab = self.vocab_bow_non_stopwords
        id_train = self._to_id_corpus_bow(self.train_corpus, vocab)
        id_test = self._to_id_corpus_bow(self.test_corpus, vocab)
        return Pack(train=id_train, test=id_test, vocab_size=len(vocab))

    def get_corpus_seq(self):
        vocab = self.vocab_seq

        id_train = self._to_id_corpus_seq(self.train_corpus, vocab)
        id_test = self._to_id_corpus_seq(self.test_corpus, vocab)
        return Pack(train=id_train, test=id_test, vocab_size=len(vocab))

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus, self.vocab_seq, self.vocab_bow)
        id_test = self._to_id_corpus(self.test_corpus, self.vocab_seq, self.vocab_bow)
        return Pack(train=id_train, test=id_test, vocab_size=len(self.vocab_bow))