# Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SequenceEncoder(nn.Module):

    def __init__(self, feature_dim, nhead, nhid, nlayers, dropout=0.5):
        """
        Params:
            feature_dim (int):  feature dimension
            nhead (int):        the number of heads in the multiheadattention models
            nhid (int):         the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers (int):      the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        """
        super(SequenceEncoder, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_dim, dropout)
        encoder_layers = TransformerEncoderLayer(feature_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.feature_dim = feature_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, src):
        """
        Params:
            src (tensor): input of size (seq_len, batch_size, feature_dim). Can accept variable length sequence.
        """

        # src: (T, N, E)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        #output = self.transformer_encoder(src, self.src_mask)    # (T, N, E)
        output = self.transformer_encoder(src)    # (T, N, E)
        return output


class TransformerClassifier(nn.Module):

    def __init__(self, seq_len, feature_dim, nhead, nhid, nlayers, num_classes, dropout=0.5):
        """
        Encode sequence using Transformer, flatten the encoded feature, then classify.

        Params:
            seq_len (int):      sequence length
            feature_dim (int):  feature dimension
            nhead (int):        the number of heads in the multiheadattention models
            nhid (int):         the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers (int):      the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            num_classes (int):  number of categories
        """
        super(TransformerClassifier, self).__init__()
        self.encoder = SequenceEncoder(feature_dim, nhead, nhid, nlayers, dropout)
        self.classifier = nn.Linear(seq_len * feature_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Params:
            src (tensor): input of size (batch_size, seq_len, ninp). Can accept variable length sequence.
        """

        # src: (N, T, E)
        src = src.transpose(1, 0)                               # (T, N, E)
        output = self.encoder(src)                              # (T, N, E)

        output = output.transpose(1,0)                          # (N, T, E)
        output = output.reshape(output.size(0), -1)             # (N, T*E)
        output = self.classifier(output)                        # (N, C): C as in num_classes
        return output


class TransformerClassifierAvg(nn.Module):

    def __init__(self, feature_dim, nhead, nhid, nlayers, num_classes, dropout=0.5):
        """
        Encode sequence using Transformer, then classify.
        Average pool the output sequence.

        Params:
            feature_dim (int):  feature dimension
            nhead (int):        the number of heads in the multiheadattention models
            nhid (int):         the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers (int):      the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            num_classes (int):  number of categories
        """
        super(TransformerClassifierAvg, self).__init__()
        self.encoder = SequenceEncoder(feature_dim, nhead, nhid, nlayers, dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Params:
            src (tensor): input of size (batch_size, seq_len, ninp). Can accept variable length sequence.
        """

        # src: (N, T, E)
        src = src.transpose(1, 0)                               # (T, N, E)
        output = self.encoder(src)                              # (T, N, E)

        output = self.classifier(output)                        # (T, N, C): C as in num_classes
        output = torch.mean(output, 0)                          # (N, C)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



if __name__ == '__main__':
    emsize = 256 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    num_classes = 172
    dropout = 0.5 # the dropout value

    seq_len = 8
    model = TransformerClassifier(seq_len, emsize, nhead, nhid, nlayers, num_classes, dropout)


    batch_size = 32

    src = torch.rand((batch_size, seq_len, emsize))
    output = model(src)
    print(output.shape)


