# Reference: https://github.com/eriklindernoren/Action-Recognition

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSequenceEncoder(nn.Module):

    def __init__(self, feature_dim, hidden_dim, num_layers, bidirectional=True, dropout=0.5):
        """
        Params:
            feature_dim (int):  feature dimension
            hidden_dim (int):  The number of features in the hidden state h
            num_layers (int):   Number of recurrent layers.
        """
        super(LSTMSequenceEncoder, self).__init__()
        self.model_type = 'LSTM'
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers = num_layers, batch_first=True, dropout = dropout, bidirectional=bidirectional)
        self.hidden_state = None

        self.feature_dim = feature_dim

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, src):
        """
        Params:
            src (tensor): input of size (batch_size, seq_len, feature_dim). Can accept variable length sequence.
        """

        # src: (N, T, E)
        src, self.hidden_state = self.lstm(src, self.hidden_state)
        return src 


class LSTMClassifier(nn.Module):

    def __init__(self, num_classes, feature_dim, lstm_layers=1, lstm_hidden_dim=1024, lstm_dropout=0.5, lstm_bidirectional=True, attention=True):
        """
        Encode sequence using LSTM, flatten the encoded feature, then classify.

        Params:
            feature_dim (int):  feature dimension
            num_classes (int):  number of categories
        """
        super(LSTMClassifier, self).__init__()
        self.encoder = LSTMSequenceEncoder(feature_dim, lstm_hidden_dim, lstm_layers, lstm_bidirectional, lstm_dropout)

        self.output_layers = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim if lstm_bidirectional else lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_hidden_dim, momentum=0.01),
            nn.Linear(lstm_hidden_dim, num_classes),
        )


        self.attention = attention
        if attention:
            self.attention_layer = nn.Linear(2 * lstm_hidden_dim if lstm_bidirectional else lstm_hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for layer in self.output_layers.modules():
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)
        if self.attention:
            self.attention_layer.bias.data.zero_()
            self.attention_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """
        Params:
            x (tensor): input of size (batch_size, seq_len, feature_dim). Can accept variable length sequence.
        """

        # src: (N, T, E)

        self.encoder.reset_hidden_state()

        x = self.encoder(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)


if __name__ == '__main__':
    hidden_dim = 256
    feature_dim = 200  # embedding dimension
    nlayers = 2
    num_classes = 172
    dropout = 0.5 # the dropout value

    model = LSTMClassifier(num_classes, feature_dim, nlayers, hidden_dim, dropout, True, True)

    batch_size = 32
    seq_len = 8
    src = torch.rand((batch_size, seq_len, feature_dim))
    output = model(src)
    print(output.shape)


