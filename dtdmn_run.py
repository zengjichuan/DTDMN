# -*- coding: utf-8 -*-
"""
for CMV: y=10
for Court: y=8, batch_size=8
"""
from __future__ import print_function

import argparse
import json
import logging
import os

import torch

import engine
from dataset import corpora
from dataset import data_loaders
from models import dtdmn
from utils import str2bool, prepare_dirs_loggers, get_time, process_config

import gen_utils

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=list, default=['data/cmv/cmv.json'])
data_arg.add_argument('--log_dir', type=str, default='logs')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--y', type=int, default=10)
net_arg.add_argument('--y_size', type=int, default=1)
net_arg.add_argument('--k', type=int, default=50)
net_arg.add_argument('--k_size', type=int, default=40)
net_arg.add_argument('--wn_pd', type=int, default=200)
net_arg.add_argument('--use_attribute', type=str2bool, default=False)
net_arg.add_argument('--variable_length', type=str2bool, default=True)

net_arg.add_argument('--rnn_cell', type=str, default='gru')
net_arg.add_argument('--embed_size', type=int, default=200)
net_arg.add_argument('--hidden_size', type=int, default=512)
net_arg.add_argument('--mem_state_size', type=int, default=200)
net_arg.add_argument('--transition_dim', type=int, default=200)
net_arg.add_argument('--utt_type', type=str, default='attn_rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=256)
net_arg.add_argument('--ctx_cell_size', type=int, default=512)
net_arg.add_argument('--dec_cell_size', type=int, default=512)
net_arg.add_argument('--bi_ctx_cell', type=str2bool, default=False)
net_arg.add_argument('--max_utt_len', type=int, default=200)
net_arg.add_argument('--max_dec_len', type=int, default=40)
net_arg.add_argument('--max_vocab_cnt', type=int, default=50000)
net_arg.add_argument('--max_data_size', type=int, default=500000)
net_arg.add_argument('--rnn_num_layers', type=int, default=1)
net_arg.add_argument('--rnn_dim', type=int, default=600)
net_arg.add_argument('--use_attn', type=str2bool, default=False)
net_arg.add_argument('--attn_type', type=str, default='cat')
net_arg.add_argument('--sparsity', type=float, default=0.3)
net_arg.add_argument('--greedy_q', type=str2bool, default=True)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--window_size', type=int, default=20)
train_arg.add_argument('--step_size', type=int, default=1)
train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.1)
train_arg.add_argument('--init_lr', type=float, default=0.001)
train_arg.add_argument('--beta1', type=float, default=0.96)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--clip-norm', type=float, default=50.0)
train_arg.add_argument('--weight-decay', type=float, default=0.0)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.99996)
train_arg.add_argument('--use_l1_reg', type=str2bool, default=True)
train_arg.add_argument('--rnn_dropout_rate', type=float, default=0.3)
train_arg.add_argument('--annealing_steps', type=float, default=0)
train_arg.add_argument('--improve_threshold', type=float, default=0.996)
train_arg.add_argument('--patient_increase', type=float, default=4.0)
train_arg.add_argument('--early_stop', type=str2bool, default=False)
train_arg.add_argument('--max_epoch', type=int, default=200)

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--output_mask', type=str2bool, default=False)
misc_arg.add_argument('--output_vis', type=str2bool, default=False)
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--jit', type=str2bool, default=False)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--print_step', type=int, default=50)
misc_arg.add_argument('--ckpt_step', type=int, default=250)
misc_arg.add_argument('--freeze_step', type=int, default=10000)
misc_arg.add_argument('--batch_size', type=int, default=16)
misc_arg.add_argument('--preview_batch_num', type=int, default=1)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=10)
misc_arg.add_argument('--forward_only', type=str2bool, default=False)
data_arg.add_argument('--load_sess', type=str, default="2018-10-20T15-50-54-topic_disc.py")
data_arg.add_argument('--token', type=str, default="")
logger = logging.getLogger()


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))
    corpus_client = corpora.CMVCorpus(config)
    # corpus_client = corpora.CourtCorpus(config)

    conv_corpus = corpus_client.get_corpus()
    # train_conv, test_conv, valid_conv, vocab_size = conv_corpus['train'],\
    #                                     conv_corpus['test'],\
    #                                     conv_corpus['valid'],\
    #                                     conv_corpus['vocab_size']

    train_conv, test_conv, vocab_size = conv_corpus['train'], \
                                                    conv_corpus['test'], \
                                                    conv_corpus['vocab_size']

    # create data loader that feed the deep models
    train_feed = data_loaders.CMVDataLoader("train", train_conv, vocab_size, config)
    test_feed = data_loaders.CMVDataLoader("test", test_conv, vocab_size, config)
    # valid_feed = data_loaders.CMVDataLoader("Valid", valid_conv, vocab_size, config)

    model = dtdmn.DTDMN(corpus_client, config)

    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file_train = os.path.join(config.session_dir, "{}-train.pkl".format(get_time()))
        dump_file_test = os.path.join(config.session_dir, "{}-test.pkl".format(get_time()))
        dump_file_valid = os.path.join(config.session_dir, "{}-valid.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file_train = os.path.join(config.session_dir, "{}-train.pkl".format(get_time()))
        dump_file_test = os.path.join(config.session_dir, "{}-test.pkl".format(get_time()))
        dump_file_valid = os.path.join(config.session_dir, "{}-valid.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")


    if config.use_gpu:
        model.cuda()

    if config.forward_only is False:
        try:
            engine.train(model, train_feed, test_feed, config)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")

    # config.batch_size = 10
    model.load_state_dict(torch.load(model_file))
    engine.inference(model, test_feed, config, num_batch=None)

    if config.output_vis:
        with open(dump_file_train, "wb") as gen_f:
            gen_utils.generate_with_act(model, train_feed, config, num_batch=None, dest_f=gen_f)
        with open(dump_file_test, "wb") as gen_f:
            gen_utils.generate_with_act(model, test_feed, config, num_batch=None, dest_f=gen_f)

    # if config.output_mask:
    #     with open(dump_file_valid, "wb") as gen_f:
    #         gen_utils.generate_with_mask(model, valid_feed, config, num_batch=None, dest_f=gen_f)


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
