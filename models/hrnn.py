"""
Simple joint topic model
"""

import argparse
import time
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import criterions
from models.encoders import EncoderRNN, RnnUttEncoder, MultiFC, RnnContextEncoder
from models.model_bases import BaseModel
from utils import Pack, FLOAT, LONG, cast_type, reverse_sequences, mask_mean

INFER = 0
TRAIN = 1


class HRNN(BaseModel):
    """
    Simple neural topic model
    """
    def __init__(self, corpus, config):
        super(HRNN, self).__init__(config)
        self.config = config
        self.vocab_seq = corpus.vocab_seq
        self.vocab_size = len(self.vocab_seq)
        self.word_num_per_doc = config.wn_pd

        # encoder, generator and decoder
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size,
                                               padding_idx=corpus.pad_wid)

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.rnn_dropout_rate,
                                         use_attn=config.utt_type == 'attn_rnn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding)
        self.ctx_encoder = RnnContextEncoder(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      config.rnn_dropout_rate,
                                      config.rnn_cell,
                                      bidirection=True,
                                      use_attn=True)
        # predictor
        self.predictor = MultiFC(config.ctx_cell_size*2, 256, 1, num_hidden_layers=1, short_cut=True,
                                 active_func="relu")

        self.pair_bce_loss = criterions.PairBCELoss()

    def qzx_forward(self, batch_utt):
        x_out = torch.tanh(self.x_encoder(batch_utt))
        z_mu = self.q_z_mu(x_out)
        z_logvar = self.q_z_logvar(x_out)

        sample_z = self.reparameterize(z_mu, z_logvar)
        return Pack(sample_z=sample_z, z_mu=z_mu, z_logvar=z_logvar)

    def pxz_forward(self, results):
        x_gen = self.x_generator(results.sample_z)
        x_logit = self.x_decoder(x_gen)

        results['x_gen'] = x_gen
        results['x_logit'] = x_logit
        return results

    def epoch_forward(self, batch_data, mask):
        # this is the number of time steps we need to process in the mini-batch
        T_max = batch_data.size(1)

        x_logit_lst = []
        z_sample_lst = []
        z_mu_lst = []
        z_logvar_lst = []

        for t in range(T_max):
            vae_x_resp = self.pxz_forward(self.qzx_forward(batch_data[:, t, :]))

            x_logit_lst.append(vae_x_resp.x_logit.unsqueeze(1))
            z_sample_lst.append(vae_x_resp.sample_z.unsqueeze(1))
            z_mu_lst.append(vae_x_resp.z_mu.unsqueeze(1))
            z_logvar_lst.append(vae_x_resp.z_logvar.unsqueeze(1))

        x_logit_seq = torch.cat(x_logit_lst, dim=1)
        z_sample_seq = torch.cat(z_sample_lst, dim=1)
        z_mu_seq = torch.cat(z_mu_lst, dim=1)
        z_logvar_seq = torch.cat(z_logvar_lst, dim=1)

        # for prediction
        pred_logit = self.predictor(mask_mean(z_sample_seq, mask))
        return Pack(x_logit_seq=x_logit_seq, z_mu_seq=z_mu_seq, z_logvar_seq=z_logvar_seq, z_sample_seq=z_sample_seq,
                    pred_logit=pred_logit)


    def get_batch(self, data_feed):
        """
        process data batch and tensorlize
        :param data_feed:
        :return:
        """
        batch_pos_utts_seq = self.np2var(data_feed.pos_utts_seq, LONG)
        batch_neg_utts_seq = self.np2var(data_feed.neg_utts_seq, LONG)
        batch_pos_utts_bow = self.np2var(data_feed.pos_utts_bow, FLOAT)
        batch_neg_utts_bow = self.np2var(data_feed.neg_utts_bow, FLOAT)
        batch_pos_masks = self.np2var(data_feed.pos_masks, FLOAT)
        batch_neg_masks = self.np2var(data_feed.neg_masks, FLOAT)
        batch_pos_lens = self.np2var(data_feed.pos_lens, LONG)
        batch_neg_lens = self.np2var(data_feed.neg_lens, LONG)
        batch_pos_words_lens = self.np2var(data_feed.pos_words_lens, LONG)
        batch_neg_words_lens = self.np2var(data_feed.neg_words_lens, LONG)
        batch_data = Pack(batch_pos_utts_seq=batch_pos_utts_seq, batch_neg_utts_seq=batch_neg_utts_seq,
                          batch_pos_utts_bow=batch_pos_utts_bow, batch_neg_utts_bow=batch_neg_utts_bow,
                          batch_pos_masks=batch_pos_masks, batch_neg_masks=batch_neg_masks,
                          batch_pos_lens=batch_pos_lens, batch_neg_lens=batch_neg_lens,
                          batch_pos_words_lens=batch_pos_words_lens, batch_neg_words_lens=batch_neg_words_lens)
        return batch_data


    def forward(self, batch_data, mode=TRAIN):
        # depack batch input
        batch_pos_utts_bow = batch_data.batch_pos_utts_bow
        batch_pos_utts_seq = batch_data.batch_pos_utts_seq
        batch_neg_utts_bow = batch_data.batch_neg_utts_bow
        batch_neg_utts_seq = batch_data.batch_neg_utts_seq
        batch_pos_masks = batch_data.batch_pos_masks
        batch_neg_masks = batch_data.batch_neg_masks
        batch_pos_lens = batch_data.batch_pos_lens
        batch_neg_lens = batch_data.batch_neg_lens
        batch_pos_words_lens = batch_data.batch_pos_words_lens
        batch_neg_words_lens = batch_data.batch_neg_words_lens

        pos_c_inputs, _, _, _ = self.utt_encoder(batch_pos_utts_seq, batch_pos_words_lens, return_all=True)
        pos_c_outs, _, _, pos_ctx_attn = self.ctx_encoder(pos_c_inputs, lens=batch_pos_lens, masks=batch_pos_masks, return_all=True)
        neg_c_inputs, _, _, _ = self.utt_encoder(batch_neg_utts_seq, batch_neg_words_lens, return_all=True)
        neg_c_outs, _, _, neg_ctx_attn = self.ctx_encoder(neg_c_inputs, lens=batch_neg_lens, masks=batch_neg_masks, return_all=True)
        pos_c_outs, neg_c_outs = pos_c_outs.squeeze(0), neg_c_outs.squeeze(0)


        pos_pred_logit = self.predictor(pos_c_outs)
        neg_pred_logit = self.predictor(neg_c_outs)

        if mode == INFER:
            pred = pos_pred_logit > neg_pred_logit
            results = Pack(pred=pred)
            return results

        # loss
        bce_loss = self.pair_bce_loss(pos_pred_logit, neg_pred_logit)

        results = Pack(bce_loss=bce_loss)
        return results


    def valid_loss(self, loss, batch_cnt=None, annealing_factor=None):
        pred_loss = loss.bce_loss
        total_loss = pred_loss
        return total_loss

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
