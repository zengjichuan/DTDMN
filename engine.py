# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from models.model_bases import summary
import torch
from dataset.corpora import PAD, EOS, EOT
import os
import pickle
from models.dmm import INFER, TRAIN
from collections import defaultdict
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from utils import Pack

logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None:
                if type(val) is torch.Tensor:
                    self.losses[key].append(val.item())
                else:
                    self.losses[key].append(val)

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key:
                str_losses.append("PPL({}) {:.3f}".format(key, avg_loss))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses)


def print_topic_words(decoder, vocab_dic, n_top_words=10):
    beta_exp = decoder.weight.data.cpu().numpy().T
    for k, beta_k in enumerate(beta_exp):
        topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words-1:-1]]
        yield 'Topic {}: {}'.format(k, ' '.join(x.encode('utf-8') for x in topic_words))


def seq2bowids(model, seq_data):
    seq_words = [model.vocab_seq[w_id] for w_id in seq_data]
    return list(filter(lambda x: x is not None, [model.vocab_bow.token2id.get(w) for w in seq_words]))

def bow2seqids(model, bow_data):
    bow_words = [model.vocab_bow[w_id] for w_id in bow_data]
    return list(filter(lambda x: x is not None, [model.vocab_seq.token2id.get(w) for w in bow_words]))


def get_bow_sent(model, data):
    sent = [model.vocab_bow[w_id] for w_id in data]
    return sent

def get_seq_sent(model, data):
    sent = [model.vocab_seq[w_id] for w_id in data]
    return sent


def train(model, train_feed, test_feed, config):

    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    train_loss = LossManager()
    model.train()

    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    inference(model, test_feed, config, num_batch=None)
    while True:
        train_feed.epoch_init(config, verbose=done_epoch==0, shuffle=True)
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break

            if config.annealing_steps > 0 and batch_cnt < config.annealing_steps:
                # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                annealing_factor = 0.1 + 0.9 * (float(batch_cnt + 1) / float(config.annealing_steps))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0

            if batch_cnt == config.freeze_step:
                # update optimizer with l2 penalty
                config.weight_deday = 0.2
                # change to adagrad
                config.op = "adagrad"
                config.init_lr = 0.01
                optimizer = model.get_optimizer(config)
                # shrink ckpt_step and print_step
                config.print_step = 20
                config.ckpt_step = 100

            optimizer.zero_grad()   # clean all grad params
            # get training batches
            batch_data = model.get_batch(batch)

            loss = model(batch_data)
            model.backward(batch_cnt, loss, annealing_factor)
            optimizer.step()

            batch_cnt += 1
            train_loss.add_loss(loss)

            if batch_cnt % config.print_step == 0:
                logger.info(train_loss.pprint("Train", window=config.print_step,
                                              prefix="{}/{}-({:.3f})".format(batch_cnt % config.ckpt_step,
                                                                         config.ckpt_step, annealing_factor)))

                # update l1 strength
                if config.use_l1_reg and batch_cnt < config.freeze_step:
                    model.reg_l1_loss.update_l1_strength(model.ntm.x_decoder.weight)



            if batch_cnt % config.ckpt_step == 0:
                logger.info("\n=== Evaluating Model ===")

                done_epoch += 1

                # validation
                logging.info("Discourse Words:")
                logging.info('\n'.join(print_topic_words(model.discm.x_decoder, model.vocab_bow)))
                logging.info("Topic Words:")
                logging.info("\n".join(print_topic_words(model.ntm.x_decoder, model.vocab_bow)))
                logger.info(train_loss.pprint("Train"))
                valid_loss = validate(model, test_feed, config, batch_cnt)
                inference(model, test_feed, config, num_batch=None)
                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience,
                                       done_epoch * config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info("Update patience to {}".format(patience))

                    if config.save_model:
                        logger.info("Model Saved.")
                        torch.save(model.state_dict(),
                                   os.path.join(config.session_dir, "model"))

                    best_valid_loss = valid_loss

                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch:
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")

                    logger.info("Best validation loss %f" % best_valid_loss)

                    return

                # exit eval model
                model.train()
                train_loss.clear()
                logger.info("\n**** Epcoch {}/{} ****".format(done_epoch,
                                                       config.max_epoch))


def validate(model, valid_feed, config, batch_cnt=None):
    model.eval()
    valid_feed.epoch_init(config, shuffle=False, verbose=True)

    losses = LossManager()
    while True:
        batch = valid_feed.next_batch()
        if batch is None:
            break
        batch_data = model.get_batch(batch)
        loss = model(batch_data)

        losses.add_loss(loss)
        losses.add_backward_loss(model.valid_loss(loss, batch_cnt))
    valid_loss = losses.avg_loss()

    logger.info(losses.pprint(valid_feed.name))
    model.train()
    return valid_loss


def inference(model, data_feed, config, num_batch=1, dest_f=None):
    model.eval()
    pre_batch_size = config.batch_size
    # config.batch_size = 5
    data_feed.epoch_init(config, ignore_residual=False, shuffle=num_batch is not None, verbose=False)

    logger.info("Inference: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    pred_lst = []
    corr, total = 0, 0
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        data_batch = model.get_batch(batch)
        resp = model(data_batch, mode=INFER)
        # record pred and true items
        pred_ = resp.pred.squeeze()
        pred = pred_.cpu().data.numpy()
        pred_lst.append(pred)
    pred_vec = np.concatenate(pred_lst)
    true_vec = np.ones_like(pred_vec)
    true_vec[:true_vec.size // 2] = 0  # make half to true vec to be 0
    pred_vec = pred_vec ^ true_vec ^ 1  # "not xor" 1,1->1, 1,0->0, 0,1->0, 0,0->1

    logger.info("Test - accuracy: %.4f, f1: %.4f" % (accuracy_score(true_vec, pred_vec),
                                                                     f1_score(true_vec, pred_vec)))
    config.batch_size = pre_batch_size
    model.train()
