import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import EnhancedCell, sequence_mask


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


class ModelEMB_Cofe(ExpModelBase):
    def __init__(self, config):
        super(ModelEMB_Cofe, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']
        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_enh = EnhancedCell(config['enh'])

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        loss = self.layer_enh.forward(tk_embedding, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        list_tags = torch.argmax(self.layer_enh.predict(tk_embedding, batch_data['lengths']), dim=-1)
        return list_tags

    def predict_bs(self, batch_data: dict, beam_width=None):
        beam_width = 1 if beam_width is None else beam_width
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        return self.layer_enh.predict_bs(tk_embedding, batch_data['lengths'], beam_width)

    def load_pretrained(self, pretrained_model_name_or_path):
        return self

    def fix_bert(self):
        return self

    def get_params_by_part(self):
        all_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        return [], all_params


if __name__ == '__main__':
    pass
