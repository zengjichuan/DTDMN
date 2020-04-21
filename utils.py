from __future__ import print_function

import os
import json
import logging
from datetime import datetime
import torch
import nltk
import sys
from collections import defaultdict
from argparse import Namespace

INT = 0
LONG = 1
FLOAT = 2


class Pack(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return super(Pack, self).__getattr__(name)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

    # @staticmethod
    # def msg_from_dict(dictionary, tokenize, speaker2id, bos_id, eos_id, include_domain=False):
    #     pack = Pack()
    #     for k, v in dictionary.items():
    #         pack[k] = v
    #     pack['speaker'] = speaker2id[pack.speaker]
    #     pack['conf'] = dictionary.get('conf', 1.0)
    #     utt = pack['utt']
    #     if 'QUERY' in utt or "RET" in utt:
    #         utt = str(utt)
    #         utt = utt.translate(None, ''.join([':', '"', "{", "}", "]", "["]))
    #         utt = unicode(utt)
    #     if include_domain:
    #         pack['utt'] = [bos_id, pack['speaker'], pack['domain']] + tokenize(utt) + [eos_id]
    #     else:
    #         pack['utt'] = [bos_id, pack['speaker']] + tokenize(utt) + [eos_id]
    #     return pack


def process_config(config):
    if config.forward_only:
        load_sess = config.load_sess
        beam_size = config.beam_size
        gen_type = config.gen_type
        output_vis = config.output_vis
        output_mask = config.output_mask

        load_path = os.path.join(config.log_dir, load_sess, "params.json")
        config = load_config(load_path)
        config.forward_only = True
        config.load_sess = load_sess
        config.beam_size = beam_size
        config.gen_type = gen_type
        # config.batch_size = 5
        config.output_vis = output_vis
        config.output_mask = output_mask
    return config


def prepare_dirs_loggers(config, script=""):
    logFormatter = logging.Formatter("%(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    if config.forward_only:
        return

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    dir_name = "{}-{}".format(get_time(), script) if script else get_time()
    config.session_dir = os.path.join(config.log_dir, dir_name + "_" + config.token)  # append token
    os.mkdir(config.session_dir)

    fileHandler = logging.FileHandler(os.path.join(config.session_dir,
                                                   'session.log'))
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # save config
    param_path = os.path.join(config.session_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Save params in "+param_path)

def load_config(load_path):
    data = json.load(open(load_path, "rb"))
    config = Namespace()
    config.__dict__ = data
    return config


def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def str2bool(v):
    return v.lower() in ('true', '1')


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


# this function takes a torch mini-batch and reverses each sequence
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = mini_batch.new_zeros(mini_batch.size())
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


def mask_mean(batch, mask):
    data = torch.sum(batch * mask.unsqueeze(2), dim=1)
    return data / torch.sum(mask, dim=1, keepdim=True)


def get_tokenize():
    return nltk.RegexpTokenizer(r'\w+|#\w+|<\w+>|%\w+|[^\w\s]+').tokenize


def get_chat_tokenize():

    return nltk.RegexpTokenizer(u'\w+|:d|:p|\'s|\'ve|\'re|\'ll|<digit>|<quote>|<url>|"|\'|'
                                u'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff]|'
                                u'[^\w\s]+(?=["\'])|(?<=["\'])[^\w\s]+|[^\w\s]+').tokenize


class missingdict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()

