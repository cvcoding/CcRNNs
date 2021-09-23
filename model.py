import torch.nn.functional as F
from torch import nn
from cifar_tcn_lstm import convlstm
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, nhid, kernel_size, dropout):
        super(TCN, self).__init__()
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.layer1 = convlstm.ConvLSTM(input_size, num_channels, 3, nhid, kernel_size, 1, dropout)#总层数，2分片，1级联层
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = inputs.squeeze().unsqueeze(2)#.cuda()
        y, _ = self.layer1(x)
        y = y.permute(0, 2, 1)#.cuda()
        o = self.linear(y[:, :, -1])
        return F.log_softmax(o, dim=1)
