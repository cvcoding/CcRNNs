import torch
from torch import nn
import sys
sys.path.append("../../")
from tcn_word import TemporalConvNet
import convlstm
import torch.nn.functional as F


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, nhid_lstm,
                 kernel_size=3, dropout=0.3, emb_dropout=0.25, tied_weights=False):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)

        # self.layer_lstm = convlstm.ConvLSTM(input_size, num_channels, input_size, nhid_lstm, (3, 1), 2)
        self.layer_lstm = convlstm.ConvLSTM(input_size, num_channels, input_size, nhid_lstm, kernel_size, 2, dropout)
        # self.layercustom = nn.LSTM(input_size, nhid_lstm, 2)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.drop2 = nn.Dropout(emb_dropout*2)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # emb = self.encoder(input)
        emb = self.drop(self.encoder(input))
        y, _ = self.layer_lstm(emb)
        # y = y[0].squeeze()
        y = self.drop2(y)
        # y = y.permute(0, 2, 1)#.cuda()
        # y = self.layer_lstm.tcn(y).transpose(1, 2)  # input should have dimension (N, C, L)
        y = self.decoder(y)
        return y.contiguous()
        # return F.log_softmax(y, dim=-1)

        # emb = self.drop(self.encoder(input))
        # y = emb.permute(0, 2, 1).cuda()
        # y = self.layer_lstm.tcn(y).transpose(1, 2)  # input should have dimension (N, C, L)
        # y = self.decoder(y)
        # return y.contiguous()

        # emb = self.drop(self.encoder(input))
        # y, _ = self.layercustom(emb)
        # # y = y[0].squeeze()
        # y = self.drop(y)
        # y = self.decoder(y)
        # return F.log_softmax(y, dim=-1)
        # return y.contiguous()

