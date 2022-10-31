import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from models.base import ExpModelBase
from models.torch_utils import GreedyCellB, sequence_mask


class ModelEMB(ExpModelBase):
    def __init__(self, config):
        super(ModelEMB, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_output = nn.Linear(self.word_embedding_dim, self.tag_size)

    def forward(self, batch_data):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        return self.layer_output(tk_embedding)

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


class ModelEMB_CRF(ModelEMB):
    def __init__(self, config):
        super(ModelEMB_CRF, self).__init__(config)
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


class ModelEMB_GRDB(ExpModelBase):
    def __init__(self, config):
        super(ModelEMB_GRDB, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_grd = GreedyCellB(config['grdb'])

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        loss = self.layer_grd.forward(tk_embedding, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        list_tags = torch.argmax(self.layer_grd.predict(tk_embedding, batch_data['lengths']), dim=-1)
        return list_tags

    def predict_bs(self, batch_data: dict, beam_width=None):
        beam_width = 1 if beam_width is None else beam_width
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        return self.layer_grd.predict_bs(tk_embedding, batch_data['lengths'], beam_width)

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

    exp_name = 'zh_emb'

    exp_conf = ExpConfig(exp_name)
    obj = ModelEMB(exp_conf.mod_conf)

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
