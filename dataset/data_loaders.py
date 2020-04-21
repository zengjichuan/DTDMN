from __future__ import print_function
import numpy as np
import copy
from utils import Pack
from dataset.dataloader_bases import DataLoader
import itertools
import random
import logging
# CMV Conversation
"""
{
    "title": ...
    "op": ...
    "pos_conv_lst": [
        [],
        [],
    ],
    "neg_conv_lst": [
        [],
        [],
    ]
}
"""
logger = logging.getLogger()

class CMVDataLoader(DataLoader):
    def __init__(self, name, data, vocab_size, config):
        super(CMVDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.vocab_size = vocab_size    # here is bow's size
        seq_data, m_cnt, w_cnt = data
        self.data = self.permute_dialog(seq_data)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_conv_lens = [len(d.conv_lst) for d in self.data]
            self.indexes = list(np.argsort(all_conv_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def permute_dialog(self, data):
        rst = []
        for dialog in data:
            if self.name == "train":
                pairs = list(itertools.product(zip(dialog.pos_conv_bow_lst, dialog.pos_conv_seq_lst),
                                               zip(dialog.neg_conv_bow_lst, dialog.neg_conv_seq_lst)))
                for (pos_utts_bow, pos_utts_seq), (neg_utts_bow, neg_utts_seq) in pairs:
                    # length filter
                    pos_turns_len = len(pos_utts_bow)
                    neg_turns_len = len(neg_utts_bow)

                    if pos_turns_len <= neg_turns_len:
                        neg_utts_bow = neg_utts_bow[:pos_turns_len]
                        neg_utts_seq = neg_utts_seq[:pos_turns_len]
                        rst.append(Pack(op=dialog.op, title=dialog.title, pos_utts_bow=pos_utts_bow, pos_utts_seq=pos_utts_seq,
                                        neg_utts_bow=neg_utts_bow, neg_utts_seq=neg_utts_seq))
            elif self.name == "test":
                # if in test dataset, randomly select one pos and one neg in a conv
                pairs = list(itertools.product(zip(dialog.pos_conv_bow_lst, dialog.pos_conv_seq_lst),
                                               zip(dialog.neg_conv_bow_lst, dialog.neg_conv_seq_lst)))
                tmp_lst = []
                for (pos_utts_bow, pos_utts_seq), (neg_utts_bow, neg_utts_seq) in pairs:
                    # length filter
                    pos_turns_len = len(pos_utts_bow)
                    neg_turns_len = len(neg_utts_bow)
                    if pos_turns_len <= neg_turns_len:
                        tmp_lst.append(((pos_utts_bow, pos_utts_seq), (neg_utts_bow[:pos_turns_len], neg_utts_seq[:pos_turns_len])))
                if not tmp_lst:
                    continue
                (pos_utts_bow, pos_utts_seq), (neg_utts_bow, neg_utts_seq) = random.choice(tmp_lst)
                rst.append(Pack(op=dialog.op, title=dialog.title, pos_utts_bow=pos_utts_bow, pos_utts_seq=pos_utts_seq,
                                neg_utts_bow=neg_utts_bow, neg_utts_seq=neg_utts_seq))


        logger.info("%d conversation pairs after product" % (len(rst)))
        return rst

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        pos_utts_lens, neg_utts_lens, pos_utts_bow_lst, pos_utts_seq_lst, neg_utts_bow_lst, neg_utts_seq_lst = [], [], [], [], [], []
        pos_words_lens, neg_words_lens = [], []
        for row in rows:
            pos_utts_bow_lst.append(row.pos_utts_bow)
            pos_utts_seq_lst.append(row.pos_utts_seq)
            neg_utts_bow_lst.append(row.neg_utts_bow)
            neg_utts_seq_lst.append(row.neg_utts_seq)
            pos_utts_lens.append(len(row.pos_utts_seq))
            neg_utts_lens.append(len(row.neg_utts_seq))
            pos_words_lens.append(list(map(len, row.pos_utts_seq)))
            neg_words_lens.append(list(map(len, row.neg_utts_seq)))

        vec_pos_lens = np.array(pos_utts_lens)
        vec_neg_lens = np.array(neg_utts_lens)

        vec_pos_utts_seq = np.zeros((len(vec_pos_lens), np.max(vec_pos_lens), self.max_utt_size), dtype=np.int32)
        vec_neg_utts_seq = np.zeros((len(vec_neg_lens), np.max(vec_neg_lens), self.max_utt_size), dtype=np.int32)
        vec_pos_utts_bow = np.zeros((len(vec_pos_lens), np.max(vec_pos_lens), self.vocab_size), dtype=np.int32)
        vec_neg_utts_bow = np.zeros((len(vec_neg_lens), np.max(vec_neg_lens), self.vocab_size), dtype=np.int32)

        vec_pos_masks = np.zeros((len(vec_pos_lens), np.max(vec_pos_lens)), dtype=np.int32)
        vec_neg_masks = np.zeros((len(vec_neg_lens), np.max(vec_neg_lens)), dtype=np.int32)

        vec_pos_words_lens = np.zeros((len(vec_pos_lens), np.max(vec_pos_lens)), dtype=np.int32)
        vec_neg_words_lens = np.zeros((len(vec_neg_lens), np.max(vec_neg_lens)), dtype=np.int32)

        assert len(pos_utts_lens) == len(neg_utts_lens)
        for b_id in range(len(pos_utts_lens)):
            vec_pos_masks[b_id, :vec_pos_lens[b_id]] = np.ones(vec_pos_lens[b_id])
            vec_neg_masks[b_id, :vec_neg_lens[b_id]] = np.ones(vec_neg_lens[b_id])
            pos_new_array_seq = np.zeros((vec_pos_lens[b_id], self.max_utt_size), dtype=np.int32)
            pos_new_array_bow = np.zeros((vec_pos_lens[b_id], self.vocab_size), dtype=np.int32)
            neg_new_array_seq = np.zeros((vec_neg_lens[b_id], self.max_utt_size), dtype=np.int32)
            neg_new_array_bow = np.zeros((vec_neg_lens[b_id], self.vocab_size), dtype=np.int32)

            vec_pos_words_lens[b_id, :vec_pos_lens[b_id]] = np.array(pos_words_lens[b_id])
            vec_neg_words_lens[b_id, :vec_neg_lens[b_id]] = np.array(neg_words_lens[b_id])

            # for pos
            for i, (pos_seq, pos_bow) in enumerate(zip(pos_utts_seq_lst[b_id], pos_utts_bow_lst[b_id])):
                for j, ele in enumerate(pos_seq[:self.max_utt_size]):
                    pos_new_array_seq[i, j] = ele
                pos_new_array_bow[i, :] = self._bow2vec(pos_bow, self.vocab_size)
            vec_pos_utts_seq[b_id, 0:vec_pos_lens[b_id], :] = pos_new_array_seq
            vec_pos_utts_bow[b_id, 0:vec_pos_lens[b_id], :] = pos_new_array_bow
            # for neg
            for i, (neg_seq, neg_bow) in enumerate(zip(neg_utts_seq_lst[b_id], neg_utts_bow_lst[b_id])):
                for j, ele in enumerate(neg_seq[:self.max_utt_size]):
                    neg_new_array_seq[i, j] = ele
                neg_new_array_bow[i, :] = self._bow2vec(neg_bow, self.vocab_size)
            vec_neg_utts_seq[b_id, 0:vec_neg_lens[b_id], :] = neg_new_array_seq
            vec_neg_utts_bow[b_id, 0:vec_neg_lens[b_id], :] = neg_new_array_bow


        return Pack(pos_utts_seq=vec_pos_utts_seq, neg_utts_seq=vec_neg_utts_seq,
                    pos_utts_bow=vec_pos_utts_bow, neg_utts_bow=vec_neg_utts_bow,
                    pos_masks=vec_pos_masks, neg_masks=vec_neg_masks,
                    pos_lens=vec_pos_lens, neg_lens=vec_neg_lens,
                    pos_words_lens=vec_pos_words_lens, neg_words_lens=vec_neg_words_lens)

    def _bow2vec(self, bow, vec_size):
        vec = np.zeros(vec_size, dtype=np.int32)
        for id, val in bow:
            vec[id] = val
        return vec


class CourtDataLoader(DataLoader):
    def __init__(self, name, data, vocab_size, config):
        super(CourtDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.max_utt_turn = 100
        self.vocab_size = vocab_size    # here is bow's size
        seq_data, m_cnt, w_cnt = data
        self.data = self.permute_dialog(seq_data)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_conv_lens = [len(d.conv_lst) for d in self.data]
            self.indexes = list(np.argsort(all_conv_lens))[::-1]
        else:
            self.indexes = list(range(len(self.data)))

    def permute_dialog(self, data):
        rst = []
        for dialog in data:
            pairs = list(itertools.product(zip(dialog.pos_conv_bow_lst, dialog.pos_conv_seq_lst),
                                           zip(dialog.neg_conv_bow_lst, dialog.neg_conv_seq_lst)))
            for (pos_utts_bow, pos_utts_seq), (neg_utts_bow, neg_utts_seq) in pairs:
                # length filter, here we turncat the longer one
                pos_turns_len = len(pos_utts_bow)
                neg_turns_len = len(neg_utts_bow)
                if pos_turns_len <= neg_turns_len:
                    neg_utts_bow = neg_utts_bow[:pos_turns_len]
                    neg_utts_seq = neg_utts_seq[:pos_turns_len]
                else:
                    pos_utts_bow = pos_utts_bow[:neg_turns_len]
                    pos_utts_seq = pos_utts_seq[:neg_turns_len]
                rst.append(Pack(pos_utts_bow=pos_utts_bow, pos_utts_seq=pos_utts_seq,
                                    neg_utts_bow=neg_utts_bow, neg_utts_seq=neg_utts_seq))
        logger.info("%d conversation pairs after product" % (len(rst)))
        return rst

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        pos_utts_lens, neg_utts_lens, pos_utts_bow_lst, pos_utts_seq_lst, neg_utts_bow_lst, neg_utts_seq_lst = [], [], [], [], [], []
        pos_words_lens, neg_words_lens = [], []
        for row in rows:
            pos_utts_bow_lst.append(row.pos_utts_bow)
            pos_utts_seq_lst.append(row.pos_utts_seq)
            neg_utts_bow_lst.append(row.neg_utts_bow)
            neg_utts_seq_lst.append(row.neg_utts_seq)
            pos_utts_lens.append(len(row.pos_utts_seq))
            neg_utts_lens.append(len(row.neg_utts_seq))
            pos_words_lens.append(list(map(len, row.pos_utts_seq)))
            neg_words_lens.append(list(map(len, row.neg_utts_seq)))

        vec_pos_lens = np.array(pos_utts_lens)
        vec_neg_lens = np.array(neg_utts_lens)

        vec_pos_utts_seq = np.zeros((len(vec_pos_lens), min(self.max_utt_turn, np.max(vec_pos_lens)), self.max_utt_size), dtype=np.int32)
        vec_neg_utts_seq = np.zeros((len(vec_neg_lens), min(self.max_utt_turn, np.max(vec_neg_lens)), self.max_utt_size), dtype=np.int32)
        vec_pos_utts_bow = np.zeros((len(vec_pos_lens), min(self.max_utt_turn, np.max(vec_pos_lens)), self.vocab_size), dtype=np.int32)
        vec_neg_utts_bow = np.zeros((len(vec_neg_lens), min(self.max_utt_turn, np.max(vec_neg_lens)), self.vocab_size), dtype=np.int32)

        vec_pos_words_lens = np.zeros((len(vec_pos_lens), min(self.max_utt_turn, np.max(vec_pos_lens))), dtype=np.int32)
        vec_neg_words_lens = np.zeros((len(vec_neg_lens), min(self.max_utt_turn, np.max(vec_neg_lens))), dtype=np.int32)

        vec_pos_masks = np.zeros((len(vec_pos_lens), min(self.max_utt_turn, np.max(vec_pos_lens))), dtype=np.int32)
        vec_neg_masks = np.zeros((len(vec_neg_lens), min(self.max_utt_turn, np.max(vec_neg_lens))), dtype=np.int32)

        assert len(pos_utts_lens) == len(neg_utts_lens)
        for b_id in range(len(pos_utts_lens)):
            pos_len = min(vec_pos_lens[b_id], self.max_utt_turn)
            neg_len = min(vec_neg_lens[b_id], self.max_utt_turn)
            vec_pos_masks[b_id, :pos_len] = np.ones(pos_len)
            vec_neg_masks[b_id, :neg_len] = np.ones(neg_len)
            pos_new_array_seq = np.zeros((pos_len, self.max_utt_size), dtype=np.int32)
            pos_new_array_bow = np.zeros((pos_len, self.vocab_size), dtype=np.int32)
            neg_new_array_seq = np.zeros((neg_len, self.max_utt_size), dtype=np.int32)
            neg_new_array_bow = np.zeros((neg_len, self.vocab_size), dtype=np.int32)

            vec_pos_words_lens[b_id, :pos_len] = np.array(pos_words_lens[b_id])
            vec_neg_words_lens[b_id, :neg_len] = np.array(neg_words_lens[b_id])

            # for pos
            for i, (pos_seq, pos_bow) in enumerate(zip(pos_utts_seq_lst[b_id], pos_utts_bow_lst[b_id])):
                if i >= pos_len:
                    break
                for j, ele in enumerate(pos_seq[:self.max_utt_size]):
                    pos_new_array_seq[i, j] = ele
                pos_new_array_bow[i, :] = self._bow2vec(pos_bow, self.vocab_size)
            vec_pos_utts_seq[b_id, 0:pos_len, :] = pos_new_array_seq
            vec_pos_utts_bow[b_id, 0:pos_len, :] = pos_new_array_bow
            # for neg
            for i, (neg_seq, neg_bow) in enumerate(zip(neg_utts_seq_lst[b_id], neg_utts_bow_lst[b_id])):
                if i >= neg_len:
                    break
                for j, ele in enumerate(neg_seq[:self.max_utt_size]):
                    neg_new_array_seq[i, j] = ele
                neg_new_array_bow[i, :] = self._bow2vec(neg_bow, self.vocab_size)
            vec_neg_utts_seq[b_id, 0:neg_len, :] = neg_new_array_seq
            vec_neg_utts_bow[b_id, 0:neg_len, :] = neg_new_array_bow

        return Pack(pos_utts_seq=vec_pos_utts_seq, neg_utts_seq=vec_neg_utts_seq,
                    pos_utts_bow=vec_pos_utts_bow, neg_utts_bow=vec_neg_utts_bow,
                    pos_masks=vec_pos_masks, neg_masks=vec_neg_masks,
                    pos_lens=vec_pos_lens, neg_lens=vec_neg_lens,
                    pos_words_lens=vec_pos_words_lens, neg_words_lens=vec_neg_words_lens)

    def _bow2vec(self, bow, vec_size):
        vec = np.zeros(vec_size, dtype=np.int32)
        for id, val in bow:
            vec[id] = val
        return vec