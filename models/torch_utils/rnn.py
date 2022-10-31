import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, bidirectional=False, dropout=0.0):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True,
                           bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input, lengths):
        '''
        :param input: [batch, seq_len, input_size]
        :param lengths: [batch]
        :return: [batch, seq_len, hidden_size], [batch, hidden_size]
        '''
        # sort
        lengths_sorted, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        input_sorted = input.index_select(0, idx_sort)

        # pack
        zero_num = int(lengths_sorted[lengths_sorted <= 0].size(0))
        if zero_num != 0:
            input_pack = pack_padded_sequence(input_sorted[:-zero_num], lengths_sorted[:-zero_num], batch_first=True)
        else:
            input_pack = pack_padded_sequence(input_sorted, lengths_sorted, batch_first=True)

        # forward
        self.rnn.flatten_parameters()
        output_pack, hidden_state_sorted = self.rnn(input_pack)
        if self.bidirectional:
            hidden_state_sorted = torch.cat((hidden_state_sorted[0][-2, :, :], hidden_state_sorted[0][-1, :, :]), dim=1)
        else:
            hidden_state_sorted = hidden_state_sorted[0][-1, :, :]

        # unpack
        output_sorted = pad_packed_sequence(output_pack, batch_first=True, total_length=input.size(1))[0]

        # pad 0
        if zero_num != 0:
            zz = torch.zeros(zero_num, input.size(1),
                             self.hidden_size * 2 if self.bidirectional else self.hidden_size).to(output_sorted.device)
            output_sorted = torch.cat((output_sorted, zz), 0)

            zz = torch.zeros(zero_num, self.hidden_size * 2 if self.bidirectional else self.hidden_size).to(
                hidden_state_sorted.device)
            hidden_state_sorted = torch.cat((hidden_state_sorted, zz), 0)

        # unsort
        output = output_sorted.index_select(0, idx_unsort)
        hidden_state = hidden_state_sorted.index_select(0, idx_unsort)

        return output, hidden_state

    @staticmethod
    def test():
        model = DynamicLSTM(input_size=5, hidden_size=3, num_layers=1, bias=True, bidirectional=True, dropout=0.0)
        data_input = torch.randn((4, 4, 5))
        data_lengths = torch.tensor([2, 3, 0, 1])
        print(data_input)
        print(data_lengths)
        output, hidden_state = model(data_input, data_lengths)
        print(output)
        print(hidden_state)
        print('----------------------------------')
        data_input = data_input[:1, :, :]
        data_lengths = data_lengths[:1]
        print(data_input)
        print(data_lengths)
        output, hidden_state = model(data_input, data_lengths)
        print(output)
        print(hidden_state)


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, bidirectional=False, dropout=0.0):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True,
                          bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input, lengths):
        '''
        :param input: [batch, seq_len, input_size]
        :param lengths: [batch]
        :return: [batch, seq_len, hidden_size], [batch, hidden_size]
        '''
        # sort
        lengths_sorted, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        input_sorted = input.index_select(0, idx_sort)

        # pack
        zero_num = int(lengths_sorted[lengths_sorted <= 0].size(0))
        if zero_num != 0:
            input_pack = pack_padded_sequence(input_sorted[:-zero_num], lengths_sorted[:-zero_num], batch_first=True)
        else:
            input_pack = pack_padded_sequence(input_sorted, lengths_sorted, batch_first=True)

        # forward
        self.rnn.flatten_parameters()
        output_pack, hidden_state_sorted = self.rnn(input_pack)
        if self.bidirectional:
            hidden_state_sorted = torch.cat((hidden_state_sorted[-2, :, :], hidden_state_sorted[-1, :, :]), dim=1)
        else:
            hidden_state_sorted = hidden_state_sorted[-1, :, :]

        # unpack
        output_sorted = pad_packed_sequence(output_pack, batch_first=True, total_length=input.size(1))[0]

        # pad 0
        if zero_num != 0:
            zz = torch.zeros(zero_num, input.size(1),
                             self.hidden_size * 2 if self.bidirectional else self.hidden_size).to(output_sorted.device)
            output_sorted = torch.cat((output_sorted, zz), 0)

            zz = torch.zeros(zero_num, self.hidden_size * 2 if self.bidirectional else self.hidden_size).to(
                hidden_state_sorted.device)
            hidden_state_sorted = torch.cat((hidden_state_sorted, zz), 0)

        # unsort
        output = output_sorted.index_select(0, idx_unsort)
        hidden_state = hidden_state_sorted.index_select(0, idx_unsort)

        return output, hidden_state

    @staticmethod
    def test():
        model = DynamicGRU(input_size=5, hidden_size=3, num_layers=1, bias=True, bidirectional=True, dropout=0.0)
        data_input = torch.randn((4, 4, 5))
        data_lengths = torch.tensor([2, 3, 0, 1])
        print(data_input)
        print(data_lengths)
        output, hidden_state = model(data_input, data_lengths)
        print(output)
        print(hidden_state)
        print('----------------------------------')
        data_input = data_input[:1, :, :]
        data_lengths = data_lengths[:1]
        print(data_input)
        print(data_lengths)
        output, hidden_state = model(data_input, data_lengths)
        print(output)
        print(hidden_state)


if __name__ == '__main__':
    DynamicLSTM.test()
    DynamicGRU.test()
