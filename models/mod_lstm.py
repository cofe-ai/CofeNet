import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import DynamicLSTM, EnhancedCell, sequence_mask


class ModelLSTM(ExpModelBase):
    def __init__(self, config):
        super(ModelLSTM, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.lstm_bidirectional = config['lstm_bidirectional']
        self.lstm_num_layers = config['lstm_num_layers']
        self.lstm_dropout = config['lstm_dropout']
        self.lstm_out_hidden_size = self.lstm_hidden_size if not self.lstm_bidirectional else self.lstm_hidden_size * 2

        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_lstm = DynamicLSTM(input_size=self.word_embedding_dim, hidden_size=self.lstm_hidden_size,
                                      num_layers=self.lstm_num_layers, bias=True, bidirectional=self.lstm_bidirectional,
                                      dropout=self.lstm_dropout)
        self.layer_output = nn.Linear(self.lstm_out_hidden_size, self.tag_size)

    def forward(self, batch_data):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        lstm_hidden, _ = self.layer_lstm(tk_embedding, batch_data['lengths'])
        return self.layer_output(lstm_hidden)

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        probs = torch.softmax(self(batch_data), dim=-1).clamp(min=1e-9)
        loss = F.nll_loss(torch.log(probs.transpose(1, 2)), labelss, ignore_index=ignore_idx)
        # return loss, torch.argmax(probs, dim=-1)
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


class ModelLSTM_CRF(ModelLSTM):
    def __init__(self, config):
        super(ModelLSTM_CRF, self).__init__(config)
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


class ModelLSTM_Cofe(ExpModelBase):
    def __init__(self, config):
        super(ModelLSTM_Cofe, self).__init__()
        self.tag_size = config['tag_size']
        self.words_emb_dropout_prob = config['words_emb_dropout_prob']
        self.words_rep_dropout_prob = config['words_rep_dropout_prob']

        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.lstm_bidirectional = config['lstm_bidirectional']
        self.lstm_num_layers = config['lstm_num_layers']
        self.lstm_dropout_prob = config['lstm_dropout_prob']
        self.lstm_out_hidden_size = self.lstm_hidden_size if not self.lstm_bidirectional else self.lstm_hidden_size * 2

        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.words_emb_dropout = nn.Dropout(self.words_emb_dropout_prob)
        self.layer_lstm = DynamicLSTM(input_size=self.word_embedding_dim, hidden_size=self.lstm_hidden_size,
                                      num_layers=self.lstm_num_layers, bias=True, bidirectional=self.lstm_bidirectional,
                                      dropout=self.lstm_dropout_prob)
        self.words_rep_dropout = nn.Dropout(self.words_rep_dropout_prob)
        self.layer_enh = EnhancedCell(config['enh'])

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        tk_embedding = self.words_emb_dropout(self.layer_emb(batch_data['tkidss']))
        words_hidden = self.words_rep_dropout(self.layer_lstm(tk_embedding, batch_data['lengths'])[0])
        loss = self.layer_enh.forward(words_hidden, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        words_hidden = self.layer_lstm(tk_embedding, batch_data['lengths'])[0]
        list_tags = torch.argmax(self.layer_enh.predict(words_hidden, batch_data['lengths']), dim=-1)
        return list_tags

    def load_pretrained(self, pretrained_model_name_or_path):
        return self

    def fix_bert(self):
        return self

    def get_params_by_part(self):
        all_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        return [], all_params


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils import *
    from data import imp_exp_dataset

    exp_name = 'pn_lstm_cofe'

    exp_conf = ExpConfig(exp_name)
    obj = ModelLSTM_Cofe(exp_conf.mod_conf)

    ds = imp_exp_dataset(exp_name, 'TST')
    train_loader = DataLoader(dataset=ds, batch_size=2, shuffle=True, collate_fn=ds.collate)
    print(len(train_loader))
    for batch_data, batch_lab, _ in train_loader:
        loss = obj.forward_loss(batch_data, batch_lab, ds.LBID_IGN)
        pred_y_2 = obj.predict(batch_data)
        print(loss)
        print(pred_y_2)
        print(batch_lab)
        break
