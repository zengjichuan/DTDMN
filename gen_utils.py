from __future__ import print_function

import engine
import utils
from models.dtdmn import INFER, VIS
import logging
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from engine import LossManager
import torch
import itertools
import json

logger = logging.getLogger()


def generate(model, data_feed, config, num_batch=1, dest_f=None):
    """
    Generate latent representation and visualization data
    :param model:
    :param data_feed:
    :param config:
    :param num_batch:
    :param dest_f:
    :return:
    """
    model.eval()
    old_batch_size = config.batch_size

    # if num_batch != None:
    #     config.batch_size = 5

    data_feed.epoch_init(config, ignore_residual=False, shuffle=False, verbose=False)
    config.batch_size = old_batch_size


    # data_seq, msg_cnt, word_cnt = data_seq


    # data_seq = list(itertools.chain.from_iterable(data_seq))  # flatten seq data

    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    gen_items = []

    weight_matrix_t = model.ntm.x_decoder.weight.data.cpu().numpy()       # vocab_size * (topic_num)
    beta_t = weight_matrix_t.T
    weight_matrix_d = model.discm.x_decoder.weight.data.cpu().numpy()
    beta_d = weight_matrix_d.T
    topic_words = []
    disc_words = []
    weight_matrix = model.decoder.weight.data.cpu().numpy()

    for beta_k in beta_t:
        topic_words.append([model.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-15:-1]])
    for beta_k in beta_d:
        disc_words.append([model.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-15:-1]])
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        batch_data = model.get_batch(batch)
        rst = model(batch_data, mode=INFER)

        pos_w_corr = rst.pos_w_corr_seq.cpu().data.numpy()
        neg_w_corr = rst.neg_w_corr_seq.cpu().data.numpy()

        pos_ctx_attn = rst.pos_ctx_attn.cpu().data.numpy()
        neg_ctx_attn = rst.neg_ctx_attn.cpu().data.numpy()

        pos_utt_attn = rst.pos_utt_attn.cpu().data.numpy()
        neg_utt_attn = rst.neg_utt_attn.cpu().data.numpy()

        pred = rst.pred.squeeze().cpu().data.numpy()

        pos_gen = rst.pos_gen_seq.cpu().data.numpy()
        neg_gen = rst.neg_gen_seq.cpu().data.numpy()

        pos_seq = batch_data.batch_pos_utts_seq.cpu().data.numpy()
        neg_seq = batch_data.batch_neg_utts_seq.cpu().data.numpy()
        pos_bow = batch_data.batch_pos_utts_bow.cpu().data.numpy()
        neg_bow = batch_data.batch_neg_utts_bow.cpu().data.numpy()
        pos_lens = batch_data.batch_pos_lens.cpu().data.numpy()
        neg_lens = batch_data.batch_neg_lens.cpu().data.numpy()

        # print(tar.shape)

        index_base = (data_feed.ptr - 1) * config.batch_size    # cause +1 in get_batch
        str_dict = {}
        for b_id in range(pos_seq.shape[0]):
            pos_str_lst = []
            pos_source_lst = []
            pos_weight_lst = []
            for t in range(pos_lens[b_id]):   # for every time slice
                pos_vocab_weight = []
                pos_vocab_source = []
                for v_id in range(weight_matrix.shape[0]):
                    pos_bow_att = pos_gen[b_id][t] * weight_matrix[v_id]
                    pos_max_ind = np.argmax(pos_bow_att)
                    pos_max_val = pos_bow_att[pos_max_ind]
                    pos_vocab_weight.append(pos_max_val)
                    pos_vocab_source.append(pos_max_ind)
                # filter with bow
                pos_bow_ind = pos_bow[b_id][t] > 0
                pos_vocab_weight = np.array(pos_vocab_weight)[pos_bow_ind]
                pos_vocab_source = np.array(pos_vocab_source)[pos_bow_ind]
                # map bow to seq
                pos_seq_ind = list(pos_seq[b_id][t][pos_seq[b_id][t] > 0])  # remove 0 padding
                pos_bwid_seq = engine.bow2seqids(model, np.argwhere(pos_bow_ind).ravel())
                pos_str_seq = engine.get_seq_sent(model, pos_seq_ind)
                pos_alias_source, pos_alias_weight = index_source_weight(pos_bwid_seq, pos_seq_ind,
                                                                         pos_vocab_source.astype(np.int),
                                                                         pos_vocab_weight.astype(np.float))

                # pos_wid_seq = engine.seq2bowids(model, pos_seq[b_id][t])
                # pos_str_seq = engine.get_sent(model, pos_wid_seq)
                # pos_alias_index = list(map(list(np.argwhere(pos_bow_ind).ravel()).index, pos_wid_seq))
                # pos_vocab_weight = pos_vocab_weight[pos_alias_index]
                # pos_vocab_source = pos_vocab_source[pos_alias_index]

                pos_str_lst.append(pos_str_seq)
                pos_source_lst.append(pos_alias_source)
                pos_weight_lst.append(pos_alias_weight)

                # logger.info("PosStr: {}".format(pos_str_seq))
                # logger.info("PosSource: {}".format(pos_vocab_source))
                # logger.info("PosWeight: {}".format(pos_vocab_weight))

            neg_str_lst = []
            neg_source_lst = []
            neg_weight_lst = []
            for t in range(neg_lens[b_id]):   # for every time slice
                neg_vocab_weight = []
                neg_vocab_source = []

                for v_id in range(weight_matrix.shape[0]):
                    neg_bow_att = neg_gen[b_id][t] * weight_matrix[v_id]
                    neg_max_ind = np.argmax(neg_bow_att)
                    neg_max_val = neg_bow_att[neg_max_ind]
                    neg_vocab_weight.append(neg_max_val)
                    neg_vocab_source.append(neg_max_ind)
                neg_bow_ind = neg_bow[b_id][t] > 0
                neg_vocab_weight = np.array(neg_vocab_weight)[neg_bow_ind]
                neg_vocab_source = np.array(neg_vocab_source)[neg_bow_ind]
                # map bow to seq
                neg_seq_ind = list(neg_seq[b_id][t][neg_seq[b_id][t] > 0])  # remove 0 padding
                neg_bwid_seq = engine.bow2seqids(model, np.argwhere(neg_bow_ind).ravel())
                neg_str_seq = engine.get_seq_sent(model, neg_seq_ind)
                neg_alias_source, neg_alias_weight = index_source_weight(neg_bwid_seq, neg_seq_ind,
                                                                         neg_vocab_source.astype(np.int),
                                                                         neg_vocab_weight.astype(np.float))


                neg_str_lst.append(neg_str_seq)
                neg_source_lst.append(neg_alias_source)
                neg_weight_lst.append(neg_alias_weight)

                # logger.info("NegStr: {}".format(neg_str_seq))
                # logger.info("NegSource: {}".format(neg_vocab_source))
                # logger.info("NegWeight: {}".format(neg_vocab_weight))

            # some filtering
            if pred[b_id] == 1:
                if get_hash(pos_str_lst[-1]) not in str_dict:
                    gen_items.append({"pred": "pos", "str": pos_str_lst, "source": pos_source_lst, "weight": pos_weight_lst,
                                              "w_corr": pos_w_corr[b_id], "ctx_attn": pos_ctx_attn[b_id],
                                              "utt_attn": pos_utt_attn[b_id], "topic_words": topic_words, "disc_words": disc_words})
                    str_dict[get_hash(pos_str_lst[-1])] = 1  # mark as processed
                if get_hash(neg_str_lst[-1]) not in str_dict:
                    gen_items.append({"pred": "neg", "str": neg_str_lst, "source": neg_source_lst, "weight": neg_weight_lst,
                                                "w_corr": neg_w_corr[b_id], "ctx_attn": neg_ctx_attn[b_id],
                                                "utt_attn": neg_utt_attn[b_id], "topic_words": topic_words, "disc_words": disc_words})
                    str_dict[get_hash(neg_str_lst[-1])] = 1  # mark as processed

    if gen_items and dest_f is not None:
        pickle.dump(gen_items, dest_f)

    logger.info("Generation Done")


def get_hash(word_lst):
    words = " ".join(word_lst)
    return hash(words)


def index_source_weight(tar, lst, source, weight):
    """
    if bow is not in seq, set weight=0, source=-1

    """
    rst_source = []
    rst_weight = []
    for e in lst:
        if e not in tar:
            rst_source.append(-1)
            rst_weight.append(0.0)
        else:
            idx = tar.index(e)
            rst_source.append(source[idx])
            rst_weight.append(weight[idx])
    return rst_source, rst_weight


def generate_with_mask(model, data_feed, config, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size

    # if num_batch != None:
    #     config.batch_size = 5

    data_feed.epoch_init(config, ignore_residual=False, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

    # data_seq, msg_cnt, word_cnt = data_seq

    # data_seq = list(itertools.chain.from_iterable(data_seq))  # flatten seq data

    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    str_dict = {}
    gen_items = []

    weight_matrix_t = model.ntm.x_decoder.weight.data.cpu().numpy()  # vocab_size * (topic_num)
    beta_t = weight_matrix_t.T
    weight_matrix_d = model.discm.x_decoder.weight.data.cpu().numpy()
    beta_d = weight_matrix_d.T
    topic_words = []
    disc_words = []
    weight_matrix = model.decoder.weight.data.cpu().numpy()

    for beta_k in beta_t:
        topic_words.append([model.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-15:-1]])
    for beta_k in beta_d:
        disc_words.append([model.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-15:-1]])

    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        batch_data = model.get_batch(batch)

        pos_seq = batch_data.batch_pos_utts_seq.cpu().data.numpy()
        neg_seq = batch_data.batch_neg_utts_seq.cpu().data.numpy()
        pos_lens = batch_data.batch_pos_lens.cpu().data.numpy()
        neg_lens = batch_data.batch_neg_lens.cpu().data.numpy()
        rst = model(batch_data, mode=VIS)

        for b_id in range(pos_seq.shape[0]):
            pos_str_lst = []
            neg_str_lst = []
            pos_pred_logit = rst.pos_pred_logit.squeeze().cpu().data.numpy()
            neg_pred_logit = rst.neg_pred_logit.squeeze().cpu().data.numpy()

            for t in range(pos_lens[b_id]):   # for every time slice
                pos_seq_ind = list(pos_seq[b_id][t][pos_seq[b_id][t] > 0])  # remove 0 padding
                pos_str_seq = engine.get_seq_sent(model, pos_seq_ind)
                pos_str_lst.append(pos_str_seq)

            for t in range(neg_lens[b_id]):   # for every time slice
                neg_seq_ind = list(neg_seq[b_id][t][neg_seq[b_id][t] > 0])  # remove 0 padding
                neg_str_seq = engine.get_seq_sent(model, neg_seq_ind)
                neg_str_lst.append(neg_str_seq)

            if get_hash(pos_str_lst[-1]) not in str_dict:
                gen_items.append({"pred": "pos", "str": pos_str_lst, "pred_logit": pos_pred_logit[b_id],
                                  "topic_words": topic_words, "disc_words": disc_words})
                str_dict[get_hash(pos_str_lst[-1])] = 1  # mark as processed
            if get_hash(neg_str_lst[-1]) not in str_dict:
                gen_items.append({"pred": "neg", "str": neg_str_lst, "pred_logit": neg_pred_logit[b_id],
                                  "topic_words": topic_words, "disc_words": disc_words})
                str_dict[get_hash(neg_str_lst[-1])] = 1  # mark as processed

    if gen_items and dest_f is not None:
        pickle.dump(gen_items, dest_f)

    logger.info("Generation Done")


def generate_with_act(model, data_feed, config, num_batch=1, dest_f=None):
    """
    Generate latent representation and visualization data
    :param model:
    :param data_feed:
    :param config:
    :param num_batch:
    :param dest_f:
    :return:
    """
    model.eval()
    old_batch_size = config.batch_size

    # if num_batch != None:
    #     config.batch_size = 5

    data_feed.epoch_init(config, ignore_residual=False, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    gen_items = []

    weight_matrix_t = model.ntm.x_decoder.weight.data.cpu().numpy()       # vocab_size * (topic_num)
    beta_t = weight_matrix_t.T
    weight_matrix_d = model.discm.x_decoder.weight.data.cpu().numpy()
    beta_d = weight_matrix_d.T
    topic_words = []
    disc_words = []

    for beta_k in beta_t:
        topic_words.append([model.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-15:-1]])
    for beta_k in beta_d:
        disc_words.append([model.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-15:-1]])
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        batch_data = model.get_batch(batch)
        rst = model(batch_data, mode=INFER)

        pos_w_corr = rst.pos_w_corr_seq.cpu().data.numpy()
        neg_w_corr = rst.neg_w_corr_seq.cpu().data.numpy()

        pos_ctx_attn = rst.pos_ctx_attn.cpu().data.numpy()
        neg_ctx_attn = rst.neg_ctx_attn.cpu().data.numpy()

        pos_utt_attn = rst.pos_utt_attn.cpu().data.numpy()
        neg_utt_attn = rst.neg_utt_attn.cpu().data.numpy()

        pred = rst.pred.squeeze().cpu().data.numpy()

        pos_gen = rst.pos_gen_seq.cpu().data.numpy()
        neg_gen = rst.neg_gen_seq.cpu().data.numpy()

        pos_seq = batch_data.batch_pos_utts_seq.cpu().data.numpy()
        neg_seq = batch_data.batch_neg_utts_seq.cpu().data.numpy()
        pos_bow = batch_data.batch_pos_utts_bow.cpu().data.numpy()
        neg_bow = batch_data.batch_neg_utts_bow.cpu().data.numpy()
        pos_lens = batch_data.batch_pos_lens.cpu().data.numpy()
        neg_lens = batch_data.batch_neg_lens.cpu().data.numpy()

        # print(tar.shape)

        index_base = (data_feed.ptr - 1) * config.batch_size    # cause +1 in get_batch
        str_dict = {}
        for b_id in range(pos_seq.shape[0]):
            pos_str_lst = []
            pos_source_lst = []
            pos_weight_lst = []
            for t in range(pos_lens[b_id]):   # for every time slice
                # map bow to seq
                pos_seq_ind = list(pos_seq[b_id][t][pos_seq[b_id][t] > 0])  # remove 0 padding
                pos_str_seq = engine.get_seq_sent(model, pos_seq_ind)
                pos_str_lst.append(pos_str_seq)

            neg_str_lst = []
            for t in range(neg_lens[b_id]):   # for every time slice
                # map bow to seq
                neg_seq_ind = list(neg_seq[b_id][t][neg_seq[b_id][t] > 0])  # remove 0 padding
                neg_str_seq = engine.get_seq_sent(model, neg_seq_ind)
                neg_str_lst.append(neg_str_seq)

            # some filtering
            if pred[b_id] == 1:
                if get_hash(pos_str_lst[-1]) not in str_dict:
                    gen_items.append({"pred": "pos", "str": pos_str_lst, "w_corr": pos_w_corr[b_id],
                                              "utt_attn": pos_utt_attn[b_id], "topic_words": topic_words, "disc_words": disc_words})
                    str_dict[get_hash(pos_str_lst[-1])] = 1  # mark as processed
                if get_hash(neg_str_lst[-1]) not in str_dict:
                    gen_items.append({"pred": "neg", "str": neg_str_lst, "w_corr": neg_w_corr[b_id],
                                                "utt_attn": neg_utt_attn[b_id], "topic_words": topic_words, "disc_words": disc_words})
                    str_dict[get_hash(neg_str_lst[-1])] = 1  # mark as processed

    if gen_items and dest_f is not None:
        pickle.dump(gen_items, dest_f)

    logger.info("Generation Done")
