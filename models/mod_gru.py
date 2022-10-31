import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from models.base import ExpModelBase
from models.torch_utils import DynamicGRU, EnhancedCell, sequence_mask


class ModelGRU(ExpModelBase):
    def __init__(self, config):
        super(ModelGRU, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.gru_hidden_size = config['gru_hidden_size']
        self.gru_bidirectional = config['gru_bidirectional']
        self.gru_num_layers = config['gru_num_layers']
        self.gru_dropout = config['gru_dropout']
        self.gru_out_hidden_size = self.gru_hidden_size if not self.gru_bidirectional else self.gru_hidden_size * 2

        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_gru = DynamicGRU(input_size=self.word_embedding_dim, hidden_size=self.gru_hidden_size,
                                    num_layers=self.gru_num_layers, bias=True, bidirectional=self.gru_bidirectional,
                                    dropout=self.gru_dropout)
        self.layer_output = nn.Linear(self.gru_out_hidden_size, self.tag_size)

    def forward(self, batch_data):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        gru_hidden, _ = self.layer_gru(tk_embedding, batch_data['lengths'])
        return self.layer_output(gru_hidden)

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


class ModelGRU_CRF(ModelGRU):
    def __init__(self, config):
        super(ModelGRU_CRF, self).__init__(config)
        self.layer_crf = CRF(self.tag_size, batch_first=True)

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):  # neg_log_likelihood
        feats = self(batch_data)
        seq_mask = sequence_mask(batch_data['lengths']).float()
        log_likelihood = self.layer_crf.forward(feats, labelss, mask=seq_mask.byte(), reduction='mean')  # mean seq loss
        loss = -log_likelihood
        return loss

    def predict(self, batch_data):
        feats = self(batch_data)
        seq_mask = sequence_mask(batch_data['lengths']).float()
        b_tag_seq = self.layer_crf.decode(feats, mask=seq_mask.byte())
        return b_tag_seq


class ModelGRU_Cofe(ExpModelBase):
    def __init__(self, config):
        super(ModelGRU_Cofe, self).__init__()
        self.tag_size = config['tag_size']
        self.words_emb_dropout_prob = config['words_emb_dropout_prob']
        self.words_rep_dropout_prob = config['words_rep_dropout_prob']

        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.gru_hidden_size = config['gru_hidden_size']
        self.gru_bidirectional = config['gru_bidirectional']
        self.gru_num_layers = config['gru_num_layers']
        self.gru_dropout = config['gru_dropout']
        self.gru_out_hidden_size = self.gru_hidden_size if not self.gru_bidirectional else self.gru_hidden_size * 2

        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.words_emb_dropout = nn.Dropout(self.words_emb_dropout_prob)
        self.layer_gru = DynamicGRU(input_size=self.word_embedding_dim, hidden_size=self.gru_hidden_size,
                                    num_layers=self.gru_num_layers, bias=True, bidirectional=self.gru_bidirectional,
                                    dropout=self.gru_dropout)
        self.words_rep_dropout = nn.Dropout(self.words_rep_dropout_prob)
        self.layer_enh = EnhancedCell(config['grdb'])

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        tk_embedding = self.words_emb_dropout(self.layer_emb(batch_data['tkidss']))
        words_hidden = self.words_rep_dropout(self.layer_gru(tk_embedding, batch_data['lengths'])[0])
        loss = self.layer_enh.forward(words_hidden, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        words_hidden = self.layer_gru(tk_embedding, batch_data['lengths'])[0]
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

    exp_name = 'en2f_gru_grdb'

    exp_conf = ExpConfig(exp_name)
    obj = ModelGRU_Cofe(exp_conf.mod_conf)

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
