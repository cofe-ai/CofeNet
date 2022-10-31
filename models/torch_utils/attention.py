import math
import torch
import torch.nn as nn
from .funs import sequence_mask_att, softmax_mask


class LocalSelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_size, attention_dropout=0.0):
        super(LocalSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.attention_probs_dropout_prob = attention_dropout

        self.att_Wh = nn.Linear(self.hidden_size, self.attention_size, bias=True)
        self.att_V = nn.Linear(self.attention_size, 1, bias=False)
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        w_bound = math.sqrt(6. / (self.hidden_size + self.attention_size))
        nn.init.uniform_(self.att_Wh.weight, -w_bound, w_bound)
        nn.init.zeros_(self.att_Wh.bias)

        nn.init.normal_(self.att_V.weight, std=0.1)

    def forward(self, hidden_states, lengths=None):
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :param lengths: [batch_size]
        :return:
            output: [batch_size, hidden_size]
            attention_probs: [batch_size, seq_len]
        """
        # Ah = [batch_size, seq_len, attention_size]
        Ah = self.att_Wh(hidden_states)

        # w = [batch_size, seq_len, attention_size]
        w = torch.tanh(Ah)

        # attention_probs = [batch_size, seq_len]
        if lengths is None:
            attention_prob = torch.softmax(self.att_V(w).squeeze(2), dim=1)
        else:
            attention_mask = sequence_mask_att(lengths, hidden_states.size(1))
            attention_prob = softmax_mask(self.att_V(w).squeeze(2), attention_mask, dim=1)

        attention_prob = self.dropout(attention_prob)

        # [batch_size, hidden_size]
        output = torch.bmm(attention_prob.unsqueeze(1), hidden_states).squeeze(1)

        return output, attention_prob


class X1SelfAttention(nn.Module):

    def __init__(self, hidden_size, x0_hidden_size, attention_size, attention_probs_dropout_prob=0.0):
        super(X1SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.x0_hidden_size = x0_hidden_size
        self.attention_size = attention_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.att_Wh = nn.Linear(self.hidden_size, self.attention_size, bias=True)
        self.att_W0 = nn.Linear(self.x0_hidden_size, self.attention_size, bias=False)
        self.att_V = nn.Linear(self.attention_size, 1, bias=False)
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        w_bound = math.sqrt(6. / (self.hidden_size + self.attention_size))
        w0_bound = math.sqrt(6. / (self.x0_hidden_size + self.attention_size))

        nn.init.uniform_(self.att_Wh.weight, -w_bound, w_bound)
        nn.init.uniform_(self.att_W0.weight, -w0_bound, w0_bound)
        nn.init.zeros_(self.att_Wh.bias)

        nn.init.normal_(self.att_V.weight, std=0.1)

    def forward(self, hidden_states, x0_hidden_states, lengths=None):
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :param x0_hidden_states: [batch_size, x0_hidden_size]
        :param lengths: [batch_size]
        :return:
            output: [batch_size, hidden_size]
            attention_probs: [batch_size, seq_len]
        """
        # Ah = [batch_size, seq_len, attention_size]
        Ah = self.att_Wh(hidden_states)

        # A0 = [batch_size, 1, attention_size]
        A0 = self.att_W0(x0_hidden_states).unsqueeze(1)

        # w = [batch_size, seq_len, attention_size]
        # w = torch.tanh(Ah + A0)
        w = torch.relu(Ah + A0)

        # attention_probs = [batch_size, seq_len]
        if lengths is None:
            attention_prob = torch.softmax(self.att_V(w).squeeze(2), dim=1)
        else:
            attention_mask = sequence_mask_att(lengths, hidden_states.size(1))
            attention_prob = softmax_mask(self.att_V(w).squeeze(2), attention_mask, dim=1)

        attention_prob = self.dropout(attention_prob)

        # [batch_size, hidden_size]
        output = torch.bmm(attention_prob.unsqueeze(1), hidden_states).squeeze(1)

        return output, attention_prob


class X2SelfAttention(nn.Module):

    def __init__(self, hidden_size, x0_hidden_size, x1_hidden_size, attention_size, attention_probs_dropout_prob=0.0):
        super(X2SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.x0_hidden_size = x0_hidden_size
        self.x1_hidden_size = x1_hidden_size
        self.attention_size = attention_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.att_Wh = nn.Linear(self.hidden_size, self.attention_size, bias=True)
        self.att_W0 = nn.Linear(self.x0_hidden_size, self.attention_size, bias=False)
        self.att_W1 = nn.Linear(self.x1_hidden_size, self.attention_size, bias=False)
        self.att_V = nn.Linear(self.attention_size, 1, bias=False)
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        w_bound = math.sqrt(6. / (self.hidden_size + self.attention_size))
        w0_bound = math.sqrt(6. / (self.x0_hidden_size + self.attention_size))
        w1_bound = math.sqrt(6. / (self.x1_hidden_size + self.attention_size))

        nn.init.uniform_(self.att_Wh.weight, -w_bound, w_bound)
        nn.init.uniform_(self.att_W0.weight, -w0_bound, w0_bound)
        nn.init.uniform_(self.att_W1.weight, -w1_bound, w1_bound)
        nn.init.zeros_(self.att_Wh.bias)

        nn.init.normal_(self.att_V.weight, std=0.1)

    def forward(self, hidden_states, x0_hidden_states, x1_hidden_states, lengths=None):
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :param x0_hidden_states: [batch_size, x0_hidden_size]
        :param x1_hidden_states: [batch_size, x1_hidden_size]
        :param lengths: [batch_size]
        :return:
            output: [batch_size, hidden_size]
            attention_probs: [batch_size, seq_len]
        """
        # Ah = [batch_size, seq_len, attention_size]
        Ah = self.att_Wh(hidden_states)

        # A0 = [batch_size, 1, attention_size]
        A0 = self.att_W0(x0_hidden_states).unsqueeze(1)

        # A1 = [batch_size, 1, attention_size]
        A1 = self.att_W1(x1_hidden_states).unsqueeze(1)

        # w = [batch_size, seq_len, attention_size]
        # w = torch.tanh(Ah + A0 + A1)
        w = torch.relu(Ah + A0 + A1)

        # attention_probs = [batch_size, seq_len]
        if lengths is None:
            attention_prob = torch.softmax(self.att_V(w).squeeze(2), dim=1)
        else:
            attention_mask = sequence_mask_att(lengths, hidden_states.size(1))
            attention_prob = softmax_mask(self.att_V(w).squeeze(2), attention_mask, dim=1)

        attention_prob = self.dropout(attention_prob)

        # [batch_size, hidden_size]
        output = torch.bmm(attention_prob.unsqueeze(1), hidden_states).squeeze(1)

        return output, attention_prob


class X3SelfAttention(nn.Module):

    def __init__(self, hidden_size, x0_hidden_size, x1_hidden_size, x2_hidden_size,
                 attention_size, attention_probs_dropout_prob=0.0):
        super(X3SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.x0_hidden_size = x0_hidden_size
        self.x1_hidden_size = x1_hidden_size
        self.x2_hidden_size = x2_hidden_size
        self.attention_size = attention_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.att_Wh = nn.Linear(self.hidden_size, self.attention_size, bias=True)
        self.att_W0 = nn.Linear(self.x0_hidden_size, self.attention_size, bias=False)
        self.att_W1 = nn.Linear(self.x1_hidden_size, self.attention_size, bias=False)
        self.att_W2 = nn.Linear(self.x2_hidden_size, self.attention_size, bias=False)
        self.att_V = nn.Linear(self.attention_size, 1, bias=False)
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        w_bound = math.sqrt(6. / (self.hidden_size + self.attention_size))
        w0_bound = math.sqrt(6. / (self.x0_hidden_size + self.attention_size))
        w1_bound = math.sqrt(6. / (self.x1_hidden_size + self.attention_size))
        w2_bound = math.sqrt(6. / (self.x2_hidden_size + self.attention_size))

        nn.init.uniform_(self.att_Wh.weight, -w_bound, w_bound)
        nn.init.uniform_(self.att_W0.weight, -w0_bound, w0_bound)
        nn.init.uniform_(self.att_W1.weight, -w1_bound, w1_bound)
        nn.init.uniform_(self.att_W2.weight, -w2_bound, w2_bound)
        nn.init.zeros_(self.att_Wh.bias)

        nn.init.normal_(self.att_V.weight, std=0.1)

    def forward(self, hidden_states, x0_hidden_states, x1_hidden_states, x2_hidden_states, lengths=None):
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :param x0_hidden_states: [batch_size, x0_hidden_size]
        :param x1_hidden_states: [batch_size, x1_hidden_size]
        :param x2_hidden_states: [batch_size, x2_hidden_size]
        :param lengths: [batch_size]
        :return:
            output: [batch_size, hidden_size]
            attention_probs: [batch_size, seq_len]
        """
        # Ah = [batch_size, seq_len, attention_size]
        Ah = self.att_Wh(hidden_states)

        # A0 = [batch_size, 1, attention_size]
        A0 = self.att_W0(x0_hidden_states).unsqueeze(1)

        # A1 = [batch_size, 1, attention_size]
        A1 = self.att_W1(x1_hidden_states).unsqueeze(1)

        # A2 = [batch_size, 1, attention_size]
        A2 = self.att_W2(x2_hidden_states).unsqueeze(1)

        # w = [batch_size, seq_len, attention_size]
        # w = torch.tanh(Ah + A0 + A1 + A2)
        w = torch.relu(Ah + A0 + A1 + A2)

        # attention_probs = [batch_size, seq_len]
        if lengths is None:
            attention_prob = torch.softmax(self.att_V(w).squeeze(2), dim=1)
        else:
            attention_mask = sequence_mask_att(lengths, hidden_states.size(1))
            attention_prob = softmax_mask(self.att_V(w).squeeze(2), attention_mask, dim=1)

        attention_prob = self.dropout(attention_prob)

        # [batch_size, hidden_size]
        output = torch.bmm(attention_prob.unsqueeze(1), hidden_states).squeeze(1)

        return output, attention_prob


class X4SelfAttention(nn.Module):

    def __init__(self, hidden_size, x0_hidden_size, x1_hidden_size, x2_hidden_size, x3_hidden_size,
                 attention_size, attention_probs_dropout_prob=0.0):
        super(X4SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.x0_hidden_size = x0_hidden_size
        self.x1_hidden_size = x1_hidden_size
        self.x2_hidden_size = x2_hidden_size
        self.x3_hidden_size = x3_hidden_size
        self.attention_size = attention_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.att_Wh = nn.Linear(self.hidden_size, self.attention_size, bias=True)
        self.att_W0 = nn.Linear(self.x0_hidden_size, self.attention_size, bias=False)
        self.att_W1 = nn.Linear(self.x1_hidden_size, self.attention_size, bias=False)
        self.att_W2 = nn.Linear(self.x2_hidden_size, self.attention_size, bias=False)
        self.att_W3 = nn.Linear(self.x3_hidden_size, self.attention_size, bias=False)
        self.att_V = nn.Linear(self.attention_size, 1, bias=False)
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        w_bound = math.sqrt(6. / (self.hidden_size + self.attention_size))
        w0_bound = math.sqrt(6. / (self.x0_hidden_size + self.attention_size))
        w1_bound = math.sqrt(6. / (self.x1_hidden_size + self.attention_size))
        w2_bound = math.sqrt(6. / (self.x2_hidden_size + self.attention_size))
        w3_bound = math.sqrt(6. / (self.x3_hidden_size + self.attention_size))

        nn.init.uniform_(self.att_Wh.weight, -w_bound, w_bound)
        nn.init.uniform_(self.att_W0.weight, -w0_bound, w0_bound)
        nn.init.uniform_(self.att_W1.weight, -w1_bound, w1_bound)
        nn.init.uniform_(self.att_W2.weight, -w2_bound, w2_bound)
        nn.init.uniform_(self.att_W3.weight, -w3_bound, w3_bound)
        nn.init.zeros_(self.att_Wh.bias)

        nn.init.normal_(self.att_V.weight, std=0.1)

    def forward(self, hidden_states, x0_hidden_states, x1_hidden_states, x2_hidden_states, x3_hidden_states, lengths=None):
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :param x0_hidden_states: [batch_size, x0_hidden_size]
        :param x1_hidden_states: [batch_size, x1_hidden_size]
        :param x2_hidden_states: [batch_size, x2_hidden_size]
        :param x3_hidden_states: [batch_size, x3_hidden_size]
        :param lengths: [batch_size]
        :return:
            output: [batch_size, hidden_size]
            attention_probs: [batch_size, seq_len]
        """
        # Ah = [batch_size, seq_len, attention_size]
        Ah = self.att_Wh(hidden_states)

        # A0 = [batch_size, 1, attention_size]
        A0 = self.att_W0(x0_hidden_states).unsqueeze(1)

        # A1 = [batch_size, 1, attention_size]
        A1 = self.att_W1(x1_hidden_states).unsqueeze(1)

        # A2 = [batch_size, 1, attention_size]
        A2 = self.att_W2(x2_hidden_states).unsqueeze(1)

        # A3 = [batch_size, 1, attention_size]
        A3 = self.att_W3(x3_hidden_states).unsqueeze(1)

        # w = [batch_size, seq_len, attention_size]
        # w = torch.tanh(Ah + A0 + A1 + A2 + A3)
        w = torch.relu(Ah + A0 + A1 + A2 + A3)

        # attention_probs = [batch_size, seq_len]
        if lengths is None:
            attention_prob = torch.softmax(self.att_V(w).squeeze(2), dim=1)
        else:
            attention_mask = sequence_mask_att(lengths, hidden_states.size(1))
            attention_prob = softmax_mask(self.att_V(w).squeeze(2), attention_mask, dim=1)

        attention_prob = self.dropout(attention_prob)

        # [batch_size, hidden_size]
        output = torch.bmm(attention_prob.unsqueeze(1), hidden_states).squeeze(1)

        return output, attention_prob


if __name__ == '__main__':
    pass
