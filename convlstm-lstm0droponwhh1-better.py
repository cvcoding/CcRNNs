import torch.nn as nn
import torch
from tcn_word import TemporalConvNet
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.kernel_size = kernel_size
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2


        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        self.fc = nn.Linear(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, bias = False)
        # self.zw = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.zu = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        # self.rw = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.ru = nn.Linear(self.input_dim, self.hidden_dim, bias=False)


        # self.fc2 = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim, bias = False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w_hh = Parameter(torch.Tensor(1, self.input_dim + self.hidden_dim)).to(self.device)
        self.bias = Parameter(torch.Tensor(1, 1)).to(self.device)
        self.dropout = 0.45
        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        # for weight in self.parameters():
        #     nn.init.xavier_normal_(weight)
        self.bias = Parameter(torch.tensor(1e-3))
        torch.nn.init.xavier_normal_(self.w_hh)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur  = cur_state

        combined = torch.cat([input_tensor.to(self.device), h_cur.to(self.device)], dim=1)  # concatenate along channel axis

        combined_conv = self.fc(combined.squeeze())

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)




        # self.zw.weight = torch.nn.Parameter(F.dropout(self.zw.weight, p=self.dropout, training=self.training))
        # self.rw.weight = torch.nn.Parameter(F.dropout(self.rw.weight, p=self.dropout, training=self.training))
        #
        # cc_z = self.zw(h_cur.to(self.device)) + self.zu(input_tensor.to(self.device))
        # cc_r = self.rw(h_cur.to(self.device)) + self.ru(input_tensor.to(self.device))
        #
        #
        # z = torch.sigmoid(cc_z).to(self.device)
        # r = torch.sigmoid(cc_r).to(self.device)
        # rh = r * h_cur.to(self.device)
        # conbined2 = torch.cat([input_tensor.to(self.device), rh.to(self.device)], dim=1)
        # con2 = torch.tanh(self.fc2(conbined2.squeeze()).to(self.device))
        #
        # h_next = (1-z) * h_cur.to(self.device) + z*con2



        # bias = Parameter(torch.Tensor(1, 1)).to(self.device)
        # delta_u = torch.sigmoid(F.linear(combined.squeeze().to(self.device), self.w_hh))   #sigmoid
        # delta_u = torch.sigmoid(F.linear(combined.squeeze().to(self.device), self.w_hh, self.bias))
        delta_u = torch.sigmoid(F.relu(F.linear(combined.squeeze(), self.w_hh))) / 1e3
        return h_next,c_next, delta_u

    def init_hidden(self, batch_size):
        # height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, device=self.fc.weight.device),
                torch.zeros(batch_size, self.hidden_dim, device=self.fc.weight.device))


def inverse4sigmoid(x):
    y = torch.log(x / (1 - x))
    return y


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_size, num_channels, input_dim, hidden_dim, kernel_size, num_layers, dropout,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        for i in range(num_layers):
            hidden_dim[i] = hidden_dim[num_layers - 1]  # int(hidden_dim[num_layers-1]/2)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.tcn = TemporalConvNet(input_size, num_channels, hidden_dim[0], kernel_size[0], dropout)
        self.dropout = nn.Dropout(0.3)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.W_a = Parameter(torch.Tensor(self.hidden_dim[0], self.hidden_dim[0])).to(self.device)
        self.U_a = Parameter(torch.Tensor(self.hidden_dim[0], self.hidden_dim[0])).to(self.device)
        self.V_a = Parameter(torch.Tensor(self.hidden_dim[0], self.hidden_dim[0])).to(self.device) # For more than 2 lstm layers
        # self.V_a = Parameter(torch.Tensor(self.hidden_dim[0], self.input_dim)).to(self.device)  # for 1 lstm layers
        self.v_a = Parameter(torch.Tensor(1, self.hidden_dim[0])).to(self.device)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.LN = nn.LayerNorm(hidden_dim[0], eps=0, elementwise_affine=True)
        self.LN1 = nn.BatchNorm1d(hidden_dim[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        # for weight in self.parameters():
        #     nn.init.xavier_normal_(weight)
        nn.init.xavier_normal_(self.W_a)
        nn.init.xavier_normal_(self.U_a)
        nn.init.xavier_normal_(self.V_a)
        nn.init.uniform_(self.v_a, -1, 1)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # input to LSTM: [seq_len, batch_size, input_size]
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2)

        _, b, _ = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor

        total_u = 0
        output_inner = []
        output4TCN = []
        for row in range(self.num_layers):
            output_inner.append([])
        for row in range(self.num_layers):
            output4TCN.append([])

        h = []
        for row in range(self.num_layers):
            h.append([])
        c = []
        for row in range(self.num_layers):
            c.append([])
        delta_u = []
        for row in range(self.num_layers):
            delta_u.append([])

        tcn_output = []  # ��tcnf&�-�H����attentionc�d
        cut_timestep = []  # �����batch(����Xc��

        for layer_idx in range(self.num_layers):  # R�PV?
            h[layer_idx], c[layer_idx] = hidden_state[layer_idx]

        once = 0

        for t in range(seq_len):
            # print(t)
            pointer = 0
            for layer_idx in range(self.num_layers):

                if layer_idx == 0:
                    cur_layer_input = input_tensor
                else:
                    cur_layer_input = self.dropout(cur_layer_input)  # dropout on each latm layer  self.dropout(cur_layer_input)

                # h[layer_idx] = self.LN(h[layer_idx])   # ==> layernorm on lstm cell
                h[layer_idx], c[layer_idx], delta_u[layer_idx] = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[t, :, :],
                    cur_state=[h[layer_idx], c[layer_idx]])

                delta_u[layer_idx] = torch.mean(delta_u[layer_idx], dim=0)
                if layer_idx == 0:
                    total_u = total_u + float(delta_u[layer_idx])
                    U = torch.Tensor(1, 1)
                    U.data.uniform_(0, 1)
                    tau = 0.001
                    # beta = inverse4sigmoid(torch.tensor(total_u))
                    beta = inverse4sigmoid(2 * torch.sigmoid(torch.tensor(total_u)) - 1)

                    pointer = torch.sigmoid((beta + torch.log(U) - torch.log(1 - U)) / tau)

                if layer_idx <= self.num_layers - 1:
                    output_inner[layer_idx].append(h[layer_idx])

                current_output = torch.stack(output_inner[layer_idx], dim=1)  # total_u >= 0.99 and

                if (pointer >= 0.99) or t == seq_len - 1:   #

                    if layer_idx == self.num_layers - 1:
                        if t == 0:
                            x = current_output.squeeze().unsqueeze(2)
                        else:
                            x = current_output.permute(0, 2, 1)  # .cuda() .squeeze()

                    if layer_idx == self.num_layers - 1:
                        cut_timestep.append(t)
                        total_u = 0

                    if layer_idx == self.num_layers - 1:  # ding ceng
                        if once == 0:
                            previous_t = t
                            x = self.dropout(x)
                            y = self.tcn(x)
                            y1 = torch.tanh(y[:, :, -1])  # .unsqueeze(2)
                            y1 = self.LN(y1)
                            tcn_output.append(y1)
                            once = 1
                        else:
                            x_seg = self.dropout(x[:, :, previous_t:t])
                            y = self.tcn(x_seg)
                            y1 = torch.tanh(y[:, :, -1])
                            y1 = self.LN(y1)
                            tcn_output.append(y1)
                            previous_t = t

                        tcn_current_output = torch.stack(tcn_output, dim=1)

                        # layer_idx = 0  # W��Iǽ|4�0Wޓ
                        repeat_h = h[layer_idx].repeat(1, tcn_current_output.shape[1]).reshape(
                            tcn_current_output.shape[0], tcn_current_output.shape[1], tcn_current_output.shape[2]).to(
                            self.device)

                        cur_x = cur_layer_input[t, :, :].to(self.device)
                        repeat_cur_x = cur_x.repeat(1, tcn_current_output.shape[1]).reshape(tcn_current_output.shape[0],
                                                                                            tcn_current_output.shape[1],
                                                                                            self.hidden_dim[
                                                                                                layer_idx]).to(
                            self.device)  # for more than 2 lstm Layers
                        # repeat_cur_x = cur_x.repeat(1, tcn_current_output.shape[1]).reshape(tcn_current_output.shape[0],
                        #                                                                     tcn_current_output.shape[1],
                        #                                                                     self.input_dim).to(
                        #     self.device)  # for 1 lstm Layers

                        att = torch.tanh(
                            F.linear(repeat_h, self.W_a) + F.linear(tcn_current_output, self.U_a) + F.linear(
                                repeat_cur_x,
                                self.V_a))  ###���NX������

                        att_temp = F.linear(att, self.v_a)
                        att_temp = F.softmax(att_temp, dim=1)  #

                        att_coef = att_temp.repeat(1, 1, tcn_current_output.shape[2])

                        att_y = torch.mul(att_coef, tcn_current_output)
                        att_y_sum = torch.sum(att_y, dim=1)

                        layer_idx = 0  # W��Iǽ|4�0Wޓ

                        h[layer_idx] = att_y_sum
                        # c[layer_idx] = torch.rand(h[layer_idx].shape[0], h[layer_idx].shape[1]) * 1e-15

                        if self.num_layers == 3:
                            layer_idx = self.num_layers - 2
                            h[layer_idx] = torch.rand(h[layer_idx].shape[0], h[layer_idx].shape[1]) * 1e-15

                        layer_idx = self.num_layers - 1  #dingceng
                        h[layer_idx] = torch.rand(h[layer_idx].shape[0], h[layer_idx].shape[1]) * 1e-15


                if layer_idx <= self.num_layers - 1:
                    layer_output = torch.stack(output_inner[layer_idx], dim=1)
                    cur_layer_input = layer_output.permute(1, 0, 2)


        # 最后对全部长度做卷积
        x = self.dropout(x)
        y = self.tcn(x)
        y1 = self.LN1(y).permute(0, 2, 1)
        #
        return y1, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
