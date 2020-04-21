import torch
import torch.nn as nn
from models.base_modules import BaseRNN
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNN(nn.Module):
    def __init__(self, emb_dim, emb_num, config):
        super(CNN, self).__init__()
        if config.emb_mat is None:
            V = emb_num
            D = emb_dim
        else:
            V,D = config.emb_mat.shape
        C = config.cls_num
        Ci = 1
        Co = config.kernel_num
        Ks = config.kernel_sizes
        self.static = config.static_emb
        self.embed = nn.Embedding(V, D, padding_idx=config.pad_wid)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class MultiFC(nn.Module):
    """
    Applies fully connected layers to an input vector
    """
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=0, short_cut=False, active_func="tanh"):
        super(MultiFC, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.short_cut = short_cut
        if active_func == "tanh":
            self.active_func = torch.tanh
        if active_func == "relu":
            self.active_func = F.relu
        elif active_func == "softplus":
            self.active_func = F.softplus
        if num_hidden_layers == 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc_layers = nn.ModuleList()
            self.fc_input = nn.Linear(input_size, hidden_size)
            self.fc_output = nn.Linear(hidden_size, output_size)
            if short_cut:
                self.es = nn.Linear(input_size, output_size)
            for i in range(self.num_hidden_layers):
                self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, input_var):
        if self.num_hidden_layers == 0:
            out = self.fc(input_var)
        else:
            x = self.active_func(self.fc_input(input_var))
            for i in range(self.num_hidden_layers):
                x = self.active_func(self.fc_layers[i](x))
            out = self.fc_output(x)
        if self.short_cut:
            out = out + self.es(input_var)
        return out


class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, input_size, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, rnn_cell='gru',
                 variable_lengths=False, bidirection=False):

        super(EncoderRNN, self).__init__(-1, input_size, hidden_size,
                                         input_dropout_p, dropout_p, n_layers,
                                         rnn_cell, bidirection)

        self.variable_lengths = variable_lengths
        self.output_size = hidden_size*2 if bidirection else hidden_size

    def forward(self, input_var, input_lengths=None, init_state=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len, embedding size): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        if self.input_dropout_p != 0:
            embedded = self.input_dropout(input_var)
        else:
            embedded = input_var
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths,
                                                         batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                         batch_first=True)
        # if self.use_attn:
        #     fc1 = torch.tanh(self.key_w(output))
        #     attn = self.query(fc1).squeeze(2)
        #     attn = self.mask_softmax(attn, attn.dim() - 1, input_mask).unsqueeze(2)
        #     utt_embedded = attn * output
        #     utt_embedded = torch.sum(utt_embedded, dim=1)
        #     return utt_embedded
        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(self, utt_cell_size, dropout,
                 rnn_cell='gru', bidirection=True, use_attn=False,
                 embedding=None, vocab_size=None, embed_dim=None,
                 feat_size=0):
        super(RnnUttEncoder, self).__init__()
        self.bidirection = bidirection
        self.utt_cell_size = utt_cell_size

        if embedding is None:
            self.embed_size = embed_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = embedding
            self.embed_size = embedding.embedding_dim

        self.rnn = EncoderRNN(self.embed_size+feat_size,
                              utt_cell_size, 0.0, dropout,
                              rnn_cell=rnn_cell, variable_lengths=False,
                              bidirection=bidirection)

        self.multipler = 2 if bidirection else 1
        self.output_size = self.utt_cell_size * self.multipler
        self.use_attn = use_attn
        self.feat_size = feat_size
        if use_attn:
            self.key_w = nn.Linear(self.utt_cell_size*self.multipler,
                                   self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, lens, feats=None, init_state=None, return_all=False):
        batch_size = int(utterances.size()[0])
        max_ctx_lens = int(utterances.size()[1])
        utt_dim = int(utterances.size()[2])
        max_utt_len = min(lens.max(), utt_dim)

        # repeat the init state
        if init_state is not None:
            init_state = init_state.repeat(1, max_ctx_lens, 1)

        # get word embeddings
        flat_words = utterances.view(-1, utt_dim)
        words_embeded = self.embedding(flat_words)
        flat_lens = lens.view(-1)
        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, utt_dim, 1)
            words_embeded = torch.cat([words_embeded, flat_feats], dim=2)

        # sort and pack input
        lens_sorted, perm_idx = flat_lens.sort(dim=0, descending=True)
        inputs = words_embeded[perm_idx]
        # cap max_utt_len, add one to len 0, avoid error in pack
        lens_sorted[lens_sorted == 0] += 1
        lens_sorted[lens_sorted > max_utt_len] = max_utt_len
        packed_input = pack_padded_sequence(inputs, lens_sorted.cpu().numpy(), batch_first=True)
        enc_outs, enc_last = self.rnn(packed_input, init_state=init_state)
        # unpack and unsort
        enc_outs, _ = pad_packed_sequence(enc_outs, batch_first=True)
        _, unperm_idx = perm_idx.sort(dim=0)
        enc_outs = enc_outs[unperm_idx]

        if self.use_attn:
            fc1 = torch.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = self.mask_softmax(attn, attn.dim()-1, flat_lens, max_utt_len).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = torch.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_lens, self.output_size)

        if return_all:
            attn = attn.view(batch_size, max_ctx_lens, -1)
            return utt_embedded, enc_outs, enc_last, attn
        else:
            return utt_embedded

    def mask_softmax(self, logit, dim, lens, max_len):
        idxes = torch.arange(int(max_len), out=logit.data.new()).expand(logit.shape)
        masks = (idxes.float() < lens.float().unsqueeze(1)).float()
        exp = torch.exp(logit) * masks
        sum_exp = exp.sum(dim=dim, keepdim=True) + 0.0001
        return exp / sum_exp

class RnnContextEncoder(nn.Module):
    def __init__(self, context_cell_size, hidden_size, dropout,
                 rnn_cell='gru', bidirection=True, use_attn=False):
        super(RnnContextEncoder, self).__init__()
        self.bidirection = bidirection
        self.context_cell_size = context_cell_size

        self.rnn = EncoderRNN(self.context_cell_size,
                              hidden_size, 0.0, dropout,
                              rnn_cell=rnn_cell, variable_lengths=False,
                              bidirection=bidirection)

        self.multipler = 2 if bidirection else 1
        self.output_size = hidden_size * self.multipler
        self.use_attn = use_attn
        if use_attn:
            self.key_w = nn.Linear(hidden_size*self.multipler,
                                   hidden_size)
            self.query = nn.Linear(hidden_size, 1)

    def forward(self, inputs, lens, masks=None, init_state=None, alias=False, return_all=False):
        batch_size = int(inputs.size()[0])
        if alias:
            # sort and pack input
            lens_sorted, perm_idx = lens.sort(dim=0, descending=True)
            inputs = inputs[perm_idx]
            packed_input = pack_padded_sequence(inputs, lens_sorted.cpu().numpy(), batch_first=True)
            enc_outs, enc_last = self.rnn(packed_input, init_state=init_state)
            # unpack and unsort
            enc_outs, _ = pad_packed_sequence(enc_outs, batch_first=True)
            _, unperm_idx = perm_idx.sort(dim=0)
            enc_outs = enc_outs[unperm_idx]
        else:
            enc_outs, enc_last = self.rnn(inputs, init_state=init_state)

        if self.use_attn:
            fc1 = torch.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            # attn = self.mask_softmax(attn, attn.dim()-1, masks).unsqueeze(2)
            attn = F.softmax(attn, attn.dim()-1).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = torch.sum(utt_embedded, dim=1)
            if masks is not None:
                attn_r = attn.squeeze() * masks
        else:
            attn_r = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(batch_size, self.output_size)


        if return_all:
            return utt_embedded, enc_outs, enc_last, attn_r
        else:
            return utt_embedded

    def mask_softmax(self, logit, dim, masks):
        exp = torch.exp(logit) * masks
        sum_exp = exp.sum(dim=dim, keepdim=True) + 0.0001
        return exp / sum_exp
