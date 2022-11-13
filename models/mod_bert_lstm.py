import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import WordBert
from .torch_utils import DynamicLSTM
from .torch_utils import sequence_mask


class ModelBertLSTM(ExpModelBase):
    def __init__(self, config):
        super(ModelBertLSTM, self).__init__()
        self.tag_size = config['tag_size']
        self.lstm_hidden_size = config['lstm']['hidden_size']
        self.lstm_bidirectional = config['lstm']['bidirectional']
        self.lstm_num_layers = config['lstm']['num_layers']
        self.lstm_dropout = config['lstm']['dropout']
        self.lstm_out_hidden_size = self.lstm_hidden_size if not self.lstm_bidirectional else self.lstm_hidden_size * 2

        self.layer_bert = WordBert(config['bert'])

        self.layer_lstm = DynamicLSTM(input_size=self.layer_bert.bert_config.hidden_size, hidden_size=self.lstm_hidden_size,
                                      num_layers=self.lstm_num_layers, bias=True, bidirectional=self.lstm_bidirectional,
                                      dropout=self.lstm_dropout)

        self.layer_output = nn.Linear(self.lstm_out_hidden_size, self.tag_size)

    def forward(self, batch_data):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        lstm_hidden, _ = self.layer_lstm(words_hidden, batch_data['lengths'])
        return self.layer_output(lstm_hidden)

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        probs = torch.softmax(self(batch_data), dim=-1).clamp(min=1e-9)
        loss = F.nll_loss(torch.log(probs.transpose(1, 2)), labelss, ignore_index=ignore_idx)
        # return loss, torch.argmax(probs, dim=-1)
        return loss

    def predict(self, batch_data: dict):
        return torch.argmax(self(batch_data), dim=-1)

    def load_pretrained(self, pretrained_model_name_or_path):
        return self.layer_bert.load_pretrained(pretrained_model_name_or_path)

    def fix_bert(self):
        return self.set_layer_trainable('layer_bert', False)

    def get_params_by_part(self):
        bert_params = list(filter(lambda p: p.requires_grad, self.layer_bert.parameters()))
        base_params = list(filter(lambda p: id(p) not in list(map(id, bert_params)) and p.requires_grad, self.parameters()))
        return bert_params, base_params


class ModelBertLSTM_CRF(ModelBertLSTM):
    def __init__(self, config):
        super(ModelBertLSTM_CRF, self).__init__(config)
        self.layer_crf = CRF(self.tag_size, batch_first=True)

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        feats = self(batch_data)
        seq_mask = sequence_mask(batch_data['lengths']).float()
        log_likelihood = self.layer_crf.forward(feats, labelss, mask=seq_mask.byte(), reduction='mean')
        loss = -log_likelihood
        return loss

    def predict(self, batch_data):
        feats = self(batch_data)
        seq_mask = sequence_mask(batch_data['lengths']).float()
        b_tag_seq = self.layer_crf.decode(feats, mask=seq_mask.byte())
        return b_tag_seq


if __name__ == '__main__':
    pass
