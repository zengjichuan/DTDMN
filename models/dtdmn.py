"""
Dynamic topic memory model
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
VIS = 2

class NtmModule(nn.Module):
    def __init__(self, vocab_size, topic_num, hidden_size):
        super(NtmModule, self).__init__()
        self.x_encoder = MultiFC(vocab_size, hidden_size, hidden_size,
                                   num_hidden_layers=1, short_cut=True, active_func="relu")
        self.q_z_mu, self.q_z_logvar = nn.Linear(hidden_size, topic_num), nn.Linear(hidden_size, topic_num)
        self.generator = MultiFC(topic_num, topic_num, topic_num, num_hidden_layers=0, short_cut=False)
        self.x_decoder = nn.Linear(topic_num, vocab_size)

    def forward(self, bow_data):
        x_out = torch.tanh(self.x_encoder(bow_data))
        z_mu, z_logvar = self.q_z_mu(x_out), self.q_z_logvar(x_out)
        sample_z = self.reparameterize(z_mu, z_logvar)
        z_gen = self.generator(sample_z)
        z_softmax = F.softmax(z_gen, dim=z_gen.dim()-1)
        x_logit = self.x_decoder(z_gen)
        return Pack(z_mu=z_mu, z_logvar=z_logvar, z_softmax=z_softmax, x_logit=x_logit, z_gen=z_gen)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

class DiscModule(nn.Module):
    def __init__(self, vocab_size, disc_num, disc_size, hidden_size, use_gpu=False):
        super(DiscModule, self).__init__()
        self.vocab_size = vocab_size
        self.disc_num = disc_num
        self.disc_size = disc_size
        self.use_gpu = use_gpu
        self.x_encoder = MultiFC(vocab_size, hidden_size, disc_num, num_hidden_layers=1, short_cut=True)
        self.generator = MultiFC(disc_num, disc_num, disc_num, num_hidden_layers=0, short_cut=False)
        self.x_decoder = nn.Linear(disc_num, vocab_size, bias=False)
        self.cat_connector = GumbelConnector()

    def forward(self, bow_data):
        qy_logit = self.x_encoder(bow_data).view(-1, self.disc_num)
        qy_logit_multi = qy_logit.repeat(self.disc_size, 1, 1)
        sample_y_multi, y_ids_multi = self.cat_connector(qy_logit_multi, 1.0,
                                                         self.use_gpu, return_max_id=True)
        sample_y = sample_y_multi.mean(0)
        y_ids = y_ids_multi.view(self.disc_size, -1).transpose(0, 1)
        y_gen = self.generator(sample_y)
        y_softmax = F.softmax(y_gen, dim=y_gen.dim()-1)
        x_logit = self.x_decoder(y_gen)
        return Pack(qy_logit=qy_logit, y_ids=y_ids, y_softmax=y_softmax, x_logit=x_logit, y_gen=y_gen)


class MemoryModule(nn.Module):
    def __init__(self, emb_size, mem_size, mem_state_size):    # here mem_size is topic_num
        super(MemoryModule, self).__init__()
        self.mem_size = mem_size
        self.emb_size = emb_size
        self.mem_state_size = mem_state_size

        self.a_conn = nn.Linear(emb_size, mem_state_size)
        self.e_conn = nn.Linear(emb_size, mem_state_size)

    def forward(self, w_corr, utt_emb, pre_mem, wr_corr=None):
        """
        Forward
        :param w_corr: correlation weight (batch_size, mem_size)
        :param utt_emb: utterance embedding (batch_size, emb_size)
        :param pre_mem: memory (mem_size, mem_state_dim)
        :return: mem: (mem_size, mem_state_dim)  read_content: (batch_size, mem_state_size)
        """
        # write process
        e_f = F.sigmoid(self.e_conn(utt_emb))   #(batch_size, mem_state_dim)
        e_w = 1 - torch.bmm(w_corr.view(-1, self.mem_size, 1),
                            e_f.view(-1, 1, self.mem_state_size))   #(batch_size, mem_size, mem_state_dim)
        a_f = torch.tanh(self.a_conn(utt_emb))  #(batch_size, mem_state_dim)
        a_w = torch.bmm(w_corr.view(-1, self.mem_size, 1), a_f.view(-1, 1, self.mem_state_size))
        mem = pre_mem * e_w + a_w

        # read process
        if wr_corr is not None:
            read_content = torch.bmm(wr_corr.view(-1, 1, self.mem_size), mem).view(-1, self.mem_state_size)
        else:
            read_content = torch.bmm(w_corr.view(-1, 1, self.mem_size), mem).view(-1, self.mem_state_size)
        return Pack(mem=mem, read_content=read_content)


class DTDMN(BaseModel):
    def __init__(self, corpus, config):#input_dim=88, z_dim=100, emission_dim=100,
                 # transition_dim=200, rnn_dim=600, num_layers=1, rnn_dropout_rate=0.0,
                 # num_iafs=0, iaf_dim=50, use_cuda=False):
        super(DTDMN, self).__init__(config)
        self.config = config
        self.vocab_seq = corpus.vocab_seq
        self.vocab_bow = corpus.vocab_bow
        self.bow_size = len(corpus.vocab_bow)
        self.vocab_bow_stopwords = corpus.vocab_bow_stopwords
        self.vocab_bow_non_stopwords = corpus.vocab_bow_non_stopwords
        # self.stopwords_ids = list(map(self.vocab_bow.token2id.get, self.vocab_bow_stopwords.values()
        self.word_num_per_doc = config.wn_pd

        # seq encoder
        self.embedding = nn.Embedding(len(self.vocab_seq), config.embed_size,
                                               padding_idx=corpus.pad_wid)

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.rnn_dropout_rate,
                                         use_attn=config.utt_type == 'attn_rnn',
                                         vocab_size=len(self.vocab_seq),
                                         embedding=self.embedding)

        # topic module
        self.ntm = NtmModule(self.bow_size, config.k, config.hidden_size)
        self.discm = DiscModule(self.bow_size, config.y, config.y_size, config.hidden_size, config.use_gpu)
        self.tmn = MemoryModule(self.utt_encoder.output_size, config.k+config.y, config.mem_state_size)

        self.decoder = nn.Linear(config.k+config.y, self.bow_size, bias=True)

        self.ctx_encoder = RnnContextEncoder(config.mem_state_size,
                                             config.ctx_cell_size,
                                             config.rnn_dropout_rate,
                                             config.rnn_cell,
                                             bidirection=True,
                                             use_attn=True)

        # predictor
        self.predictor = MultiFC(config.ctx_cell_size*2, 256, 1, num_hidden_layers=1, short_cut=True,
                                 active_func="relu")    # *2 for bidirection

        self.pair_bce_loss = criterions.PairBCELoss()
        self.nll_loss_sw = criterions.PPLLoss(self.config, vocab=self.vocab_bow, ignore_vocab=self.vocab_bow_non_stopwords)
        self.nll_loss_kw = criterions.PPLLoss(self.config, vocab=self.vocab_bow,
                                                    ignore_vocab=self.vocab_bow_stopwords)
        self.nll_loss = criterions.PPLLoss(self.config)
        self.kl_loss = criterions.GaussianKLLoss()
        self.cat_kl_loss = criterions.CatKLLoss()
        self.reg_l1_loss = criterions.L1RegLoss(config.sparsity)
        # init mem (topic_num, mem_state_size)
        self.init_mem = nn.Parameter(torch.zeros(config.k+config.y, config.mem_state_size))
        self.log_uniform_y = torch.log(torch.ones(1) / config.y)
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

    def epoch_forward(self, batch_bow_data, batch_seq_data, lens, words_lens, masks):
        # this is the number of time steps we need to process in the mini-batch
        T_max = batch_bow_data.size(1)

        qxy_logit_lst = []
        qxz_logit_lst = []
        qy_logit_lst = []
        read_content_lst = []
        z_mu_lst = []
        z_logvar_lst = []
        w_corr_lst = []
        gen_lst = []
        x_logit_lst = []

        # get embedding
        c_inputs, _, _, utt_attn = self.utt_encoder(batch_seq_data, words_lens, return_all=True) #(batch_size, seq_len, out_emb)
        pre_mem = self.init_mem
        for t in range(T_max):
            ntm_resp = self.ntm(batch_bow_data[:, t, :])
            discm_resp = self.discm(batch_bow_data[:, t, :])
            y_softmax, qxy_logit, y_gen = discm_resp.y_softmax, discm_resp.x_logit, discm_resp.y_gen
            qy_logit = discm_resp.qy_logit
            z_softmax, qxz_logit, z_gen = ntm_resp.z_softmax, ntm_resp.x_logit, ntm_resp.z_gen
            z_mu, z_logvar = ntm_resp.z_mu, ntm_resp.z_logvar
            w_corr = torch.cat([z_softmax, y_softmax], dim=1)

            gen = torch.cat([z_gen, y_gen], dim=1)
            x_logit = self.decoder(gen)

            utt_emb = c_inputs[:, t, :]
            tmn_resp = self.tmn(w_corr, utt_emb, pre_mem)
            mem, read_content = tmn_resp.mem, tmn_resp.read_content
            pre_mem = mem

            qxz_logit_lst.append(qxz_logit.unsqueeze(1))
            qxy_logit_lst.append(qxy_logit.unsqueeze(1))
            qy_logit_lst.append(qy_logit.unsqueeze(1))
            read_content_lst.append(read_content.unsqueeze(1))
            z_mu_lst.append(z_mu.unsqueeze(1))
            z_logvar_lst.append(z_logvar.unsqueeze(1))
            w_corr_lst.append(w_corr.unsqueeze(1))
            gen_lst.append(gen.unsqueeze(1))
            x_logit_lst.append(x_logit.unsqueeze(1))

        qxz_logit_seq = torch.cat(qxz_logit_lst, dim=1)
        qxy_logit_seq = torch.cat(qxy_logit_lst, dim=1)
        qy_logit_seq = torch.cat(qy_logit_lst, dim=1)
        read_content_seq = torch.cat(read_content_lst, dim=1)
        z_mu_seq = torch.cat(z_mu_lst, dim=1)
        z_logvar_seq = torch.cat(z_logvar_lst, dim=1)
        w_corr_seq = torch.cat(w_corr_lst, dim=1)
        gen_seq = torch.cat(gen_lst, dim=1)
        x_logit_seq = torch.cat(x_logit_lst, dim=1)
        # for prediction
        agg_c_outs, _, _, ctx_attn = self.ctx_encoder(read_content_seq, lens=lens, masks=masks, return_all=True)
        agg_c_outs = agg_c_outs.squeeze(0)

        pred_logit = self.predictor(agg_c_outs)
        return Pack(z_mu_seq=z_mu_seq, z_logvar_seq=z_logvar_seq, qxz_logit_seq=qxz_logit_seq, qxy_logit_seq=qxy_logit_seq,
                    qy_logit_seq=qy_logit_seq, pred_logit=pred_logit, x_logit_seq=x_logit_seq,
                    w_corr_seq=w_corr_seq, gen_seq=gen_seq, ctx_attn=ctx_attn, utt_attn=utt_attn)

    def vis_forward(self, batch_bow_data, batch_seq_data, lens, words_lens, masks):
        # this is the number of time steps we need to process in the mini-batch
        T_max = batch_bow_data.size(1)
        batch_size = batch_bow_data.size(0)
        w_size = self.config.k + self.config.y


        agg_c_last_lst = []

        # get embedding
        c_inputs, _, _, utt_attn = self.utt_encoder(batch_seq_data, words_lens, return_all=True) #(batch_size, seq_len, out_emb)
        pre_mem = self.init_mem

        w_corr_eye = torch.eye(w_size).view(1, w_size, w_size)
        w_corr_eye = w_corr_eye.repeat(batch_size, 1, 1)

        masked_read_content_t = torch.zeros(batch_size, T_max, self.config.mem_state_size)
        if self.use_gpu:
            w_corr_eye = w_corr_eye.cuda()
            masked_read_content_t = masked_read_content_t.cuda()

        for wi in range(self.config.k + self.config.y):
            wr_corr = w_corr_eye[:, wi, :]

            read_content_lst = []
            for t in range(T_max):
                utt_emb = c_inputs[:, t, :]

                ntm_resp = self.ntm(batch_bow_data[:, t, :])
                discm_resp = self.discm(batch_bow_data[:, t, :])
                y_softmax, qxy_logit, y_gen = discm_resp.y_softmax, discm_resp.x_logit, discm_resp.y_gen
                z_softmax, qxz_logit, z_gen = ntm_resp.z_softmax, ntm_resp.x_logit, ntm_resp.z_gen
                w_corr = torch.cat([z_softmax, y_softmax], dim=1)

                tmn_resp = self.tmn(w_corr, utt_emb, pre_mem, wr_corr=wr_corr)
                mem, read_content = tmn_resp.mem, tmn_resp.read_content
                pre_mem = mem

                masked_read_content = torch.zeros_like(masked_read_content_t)
                masked_read_content[:, t] = read_content    # only t dim is activated

                read_content_lst.append(masked_read_content)

            read_content_seq = torch.cat(read_content_lst, dim=0)   # (T_max*batch_size, mem_size)
            agg_c_last = self.ctx_encoder(read_content_seq, lens=lens, masks=None)  # (T_max*batch_size, rnn_cell*2)
            agg_c_last_lst.append(agg_c_last)

        agg_c_last_seq = torch.cat(agg_c_last_lst, dim=0)   # (w_size*T_max*batch_size, rnn_cell)
        pred_logit_seq = self.predictor(agg_c_last_seq)
        pred_logit = pred_logit_seq.view(w_size, T_max, batch_size).permute(2, 1, 0)  # (batch_size, T_max, w_size)

        return Pack(pred_logit=pred_logit, utt_attn=utt_attn)


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

        if mode == VIS:
            pos_fwd_resp = self.vis_forward(batch_pos_utts_bow, batch_pos_utts_seq, batch_pos_lens,
                                              batch_pos_words_lens, batch_pos_masks)
            neg_fwd_resp = self.vis_forward(batch_neg_utts_bow, batch_neg_utts_seq, batch_neg_lens,
                                              batch_neg_words_lens, batch_neg_masks)
            pos_pred_logit, neg_pred_logit = pos_fwd_resp.pred_logit, neg_fwd_resp.pred_logit
            return Pack(pos_pred_logit=pos_pred_logit, neg_pred_logit=neg_pred_logit)

        pos_fwd_resp = self.epoch_forward(batch_pos_utts_bow, batch_pos_utts_seq, batch_pos_lens, batch_pos_words_lens,
                                          batch_pos_masks)
        neg_fwd_resp = self.epoch_forward(batch_neg_utts_bow, batch_neg_utts_seq, batch_neg_lens, batch_neg_words_lens,
                                          batch_neg_masks)

        pos_pred_logit, neg_pred_logit = pos_fwd_resp.pred_logit, neg_fwd_resp.pred_logit
        pos_qxz_logit_seq, neg_qxz_logit_seq = pos_fwd_resp.qxz_logit_seq, neg_fwd_resp.qxz_logit_seq
        pos_qxy_logit_seq, neg_qxy_logit_seq = pos_fwd_resp.qxy_logit_seq, neg_fwd_resp.qxy_logit_seq
        pos_qy_logit_seq, neg_qy_logit_seq = pos_fwd_resp.qy_logit_seq, neg_fwd_resp.qy_logit_seq
        pos_z_mu_seq, pos_z_logvar_seq = pos_fwd_resp.z_mu_seq, pos_fwd_resp.z_logvar_seq
        neg_z_mu_seq, neg_z_logvar_seq = neg_fwd_resp.z_mu_seq, neg_fwd_resp.z_logvar_seq
        pos_w_corr_seq, neg_w_corr_seq = pos_fwd_resp.w_corr_seq, neg_fwd_resp.w_corr_seq
        pos_gen_seq, neg_gen_seq = pos_fwd_resp.gen_seq, neg_fwd_resp.gen_seq
        pos_ctx_attn, neg_ctx_attn = pos_fwd_resp.ctx_attn, neg_fwd_resp.ctx_attn
        pos_utt_attn, neg_utt_attn = pos_fwd_resp.utt_attn, neg_fwd_resp.utt_attn
        pos_x_logit_seq, neg_x_logit_seq = pos_fwd_resp.x_logit_seq, neg_fwd_resp.x_logit_seq
        if mode == INFER:
            pred = pos_pred_logit > neg_pred_logit
            results = Pack(pred=pred, pos_w_corr_seq=pos_w_corr_seq, neg_w_corr_seq=neg_w_corr_seq,
                           pos_gen_seq=pos_gen_seq, neg_gen_seq=neg_gen_seq,
                           pos_ctx_attn=pos_ctx_attn, neg_ctx_attn=neg_ctx_attn,
                           pos_utt_attn=pos_utt_attn, neg_utt_attn=neg_utt_attn)
            return results

        # loss
        gau_kl_loss = self.kl_loss(pos_z_mu_seq, pos_z_logvar_seq, mask=batch_pos_masks) + \
                  self.kl_loss(neg_z_mu_seq, neg_z_logvar_seq, mask=batch_neg_masks)
        pos_avg_log_qy, neg_avg_log_qy = torch.log(mask_mean(torch.exp(F.log_softmax(pos_qy_logit_seq, dim=2)), batch_pos_masks) + 1e-15), \
                                         torch.log(mask_mean(torch.exp(F.log_softmax(neg_qy_logit_seq, dim=2)), batch_neg_masks) + 1e-15)
        cat_kl_loss = self.cat_kl_loss(pos_avg_log_qy, self.log_uniform_y, unit_average=True) + \
                      self.cat_kl_loss(neg_avg_log_qy, self.log_uniform_y, unit_average=True)
        pxz_nll_loss = self.nll_loss_kw(F.log_softmax(pos_qxz_logit_seq, dim=2), batch_pos_utts_bow, mask=batch_pos_masks) + \
                   self.nll_loss_kw(F.log_softmax(neg_qxz_logit_seq, dim=2), batch_neg_utts_bow, mask=batch_neg_masks)
        pxy_nll_loss = self.nll_loss_sw(F.log_softmax(pos_qxy_logit_seq, dim=2), batch_pos_utts_bow, mask=batch_pos_masks)\
                       + self.nll_loss_sw(F.log_softmax(neg_qxy_logit_seq, dim=2), batch_neg_utts_bow, mask=batch_neg_masks)
        bce_loss = self.pair_bce_loss(pos_pred_logit, neg_pred_logit)
        nll_loss = self.nll_loss(F.log_softmax(pos_x_logit_seq, dim=2), batch_pos_utts_bow, mask=batch_pos_masks) + \
                   self.nll_loss(F.log_softmax(neg_x_logit_seq, dim=2), batch_neg_utts_bow, mask=batch_neg_masks)
        if self.config.use_l1_reg:
            l1_reg = self.reg_l1_loss(self.ntm.x_decoder.weight, torch.zeros_like(self.ntm.x_decoder.weight))
        else:
            l1_reg = None

        results = Pack(bce_loss=bce_loss, gau_kl_loss=gau_kl_loss, cat_kl_loss=cat_kl_loss,
                       pxz_nll_loss=pxz_nll_loss, pxy_nll_loss=pxy_nll_loss, l1_reg=l1_reg, nll_loss=nll_loss)
        return results


    def valid_loss(self, loss, batch_cnt=None, annealing_factor=None):
        pred_loss = loss.bce_loss
        kl_loss = loss.gau_kl_loss + loss.cat_kl_loss
        if annealing_factor is not None:
            kl_loss *= annealing_factor
        vae_loss = loss.pxz_nll_loss + loss.pxy_nll_loss + kl_loss  # + pkl_loss
        if self.config.use_l1_reg:
            vae_loss += loss.l1_reg

        if batch_cnt is not None and batch_cnt >= self.config.freeze_step:
            total_loss = pred_loss

            self.flush_valid = True
            for param in self.ntm.x_encoder.parameters():
                param.requires_grad = False
            for param in self.ntm.q_z_mu.parameters():
                param.requires_grad = False
            for param in self.ntm.q_z_logvar.parameters():
                param.requires_grad = False
            for param in self.ntm.generator.parameters():
                param.requires_grad = False
            for param in self.ntm.x_decoder.parameters():
                param.requires_grad = False
            for param in self.discm.x_encoder.parameters():
                param.requires_grad = False
            for param in self.discm.generator.parameters():
                param.requires_grad = False
            for param in self.discm.x_decoder.parameters():
                param.requires_grad = False
        else:
            total_loss = vae_loss

        return total_loss

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


class GumbelConnector(nn.Module):
    def __init__(self):
        super(GumbelConnector, self).__init__()

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = -torch.log(-torch.log(u + eps) + eps)
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature, use_gpu, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, use_gpu)
        _, y_hard = torch.max(y, dim=-1, keepdim=True)
        if hard:
            y_onehot = cast_type(torch.zeros(y.size()), FLOAT, use_gpu)
            y_onehot.scatter_(-1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y
