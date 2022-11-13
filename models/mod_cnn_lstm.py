import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import DynamicLSTM
from .torch_utils import SeqCNN


class ModelCNN_LSTM(ExpModelBase):
    def __init__(self, config):
        super(ModelCNN_LSTM, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']

        self.cnn_out_channels = config['cnn_out_channels']
        self.cnn_out_hidden_size = config['cnn_out_hidden_size']
        self.cnn_kernel_sizes = config['cnn_kernel_sizes']

        self.lstm_hidden_size = config['lstm_hidden_size']
        self.lstm_bidirectional = config['lstm_bidirectional']
        self.lstm_num_layers = config['lstm_num_layers']
        self.lstm_dropout = config['lstm_dropout']
        self.lstm_out_hidden_size = self.lstm_hidden_size if not self.lstm_bidirectional else self.lstm_hidden_size * 2

        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_cnn = SeqCNN(self.word_embedding_dim, self.cnn_out_channels, self.cnn_out_hidden_size, self.cnn_kernel_sizes)
        self.layer_lstm = DynamicLSTM(input_size=self.cnn_out_hidden_size, hidden_size=self.lstm_hidden_size,
                                      num_layers=self.lstm_num_layers, bias=True, bidirectional=self.lstm_bidirectional,
                                      dropout=self.lstm_dropout)
        self.layer_output = nn.Linear(self.lstm_out_hidden_size, self.tag_size)

    def forward(self, batch_data):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        cnn_hiddens = self.layer_cnn(tk_embedding)
        lstm_hidden, _ = self.layer_lstm(cnn_hiddens, batch_data['lengths'])
        return self.layer_output(lstm_hidden)

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        probs = torch.softmax(self(batch_data), dim=-1).clamp(min=1e-9)
        loss = F.nll_loss(torch.log(probs.transpose(1, 2)), labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict):
        return torch.argmax(self(batch_data), dim=-1)

    def load_pretrained(self, pretrained_model_name_or_path):
        return self

    def fix_bert(self):
        return self

    def get_params_by_part(self):
        all_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        return [], all_params


class ModelCNN_LSTM_CRF(ModelCNN_LSTM):
    def __init__(self, config):
        super(ModelCNN_LSTM_CRF, self).__init__(config)
        self.layer_crf = CRF(self.tag_size, batch_first=True)

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        feats = self(batch_data)
        log_likelihood = self.layer_crf.forward(feats, labelss, mask=batch_data['crfmasks'].byte(), reduction='mean')
        loss = -log_likelihood
        return loss

    def predict(self, batch_data):
        feats = self(batch_data)
        b_tag_seq = self.layer_crf.decode(feats, mask=batch_data['crfmasks'].byte())
        return b_tag_seq


if __name__ == '__main__':
    pass
