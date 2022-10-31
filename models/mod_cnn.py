import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import SeqCNN, EnhancedCell, sequence_mask


class ModelCNN(ExpModelBase):
    def __init__(self, config):
        super(ModelCNN, self).__init__()
        self.tag_size = config['tag_size']
        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']

        self.cnn_channels = config['cnn_channels']
        self.cnn_out_hidden_size = config['cnn_out_hidden_size']
        self.cnn_pool = config['cnn_pool']
        self.cnn_kernel_sizes = config['cnn_kernel_sizes']
        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.layer_cnn = SeqCNN(self.word_embedding_dim, self.cnn_channels, self.cnn_out_hidden_size, self.cnn_kernel_sizes, self.cnn_pool)
        self.layer_output = nn.Linear(self.cnn_out_hidden_size, self.tag_size)

    def forward(self, batch_data):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        cnn_hiddens = self.layer_cnn(tk_embedding)
        return self.layer_output(cnn_hiddens)

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


class ModelCNN_CRF(ModelCNN):
    def __init__(self, config):
        super(ModelCNN_CRF, self).__init__(config)
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


class ModelCNN_Cofe(ExpModelBase):
    def __init__(self, config):
        super(ModelCNN_Cofe, self).__init__()
        self.tag_size = config['tag_size']
        self.words_emb_dropout_prob = config['words_emb_dropout_prob']
        self.words_rep_dropout_prob = config['words_rep_dropout_prob']

        self.words_size = config['words_size']
        self.word_embedding_dim = config['word_embedding_dim']

        self.cnn_channels = config['cnn_channels']
        self.cnn_out_hidden_size = config['cnn_out_hidden_size']
        self.cnn_pool = config['cnn_pool']
        self.cnn_kernel_sizes = config['cnn_kernel_sizes']
        self.layer_emb = nn.Embedding(self.words_size, self.word_embedding_dim)
        self.words_emb_dropout = nn.Dropout(self.words_emb_dropout_prob)
        self.layer_cnn = SeqCNN(self.word_embedding_dim, self.cnn_channels, self.cnn_out_hidden_size, self.cnn_kernel_sizes, self.cnn_pool)
        self.words_rep_dropout = nn.Dropout(self.words_rep_dropout_prob)
        self.layer_enh = EnhancedCell(config['grdb'])

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        cnn_hiddens = self.layer_cnn(tk_embedding)
        loss = self.layer_enh.forward(cnn_hiddens, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict):
        tk_embedding = self.layer_emb(batch_data['tkidss'])
        cnn_hiddens = self.layer_cnn(tk_embedding)
        list_tags = torch.argmax(self.layer_enh.predict(cnn_hiddens, batch_data['lengths']), dim=-1)
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

    exp_name = 'zh_cnn_grdb'

    exp_conf = ExpConfig(exp_name)
    obj = ModelCNN_Cofe(exp_conf.mod_conf)

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
