import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import WordBert
from .torch_utils import SeqCNN


class ModelBertCNN(ExpModelBase):
    def __init__(self, config):
        super(ModelBertCNN, self).__init__()
        self.tag_size = config['tag_size']

        self.cnn_out_channels = config['cnn']['out_channels']
        self.cnn_out_hidden_size = config['cnn']['out_hidden_size']
        self.cnn_pool = config['cnn']['pool']
        self.cnn_kernel_sizes = config['cnn']['kernel_sizes']

        self.layer_bert = WordBert(config['bert'])

        self.layer_cnn = SeqCNN(in_hidden_size=self.layer_bert.bert_config.hidden_size,
                                channels=self.cnn_out_channels,
                                out_hidden_size=self.cnn_out_hidden_size,
                                kernel_sizes=self.cnn_kernel_sizes,
                                pool=self.cnn_pool)

        self.layer_output = nn.Linear(self.cnn_out_hidden_size, self.tag_size)

    def forward(self, batch_data):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        lstm_hidden = self.layer_cnn(words_hidden)
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


class ModelBertCNN_CRF(ModelBertCNN):
    def __init__(self, config):
        super(ModelBertCNN_CRF, self).__init__(config)
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
