import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .base import ExpModelBase
from .torch_utils import WordBert, EnhancedCell
from .torch_utils import sequence_mask


class ModelBert(ExpModelBase):
    def __init__(self, config):
        super(ModelBert, self).__init__()
        self.tag_size = config['tag_size']
        self.layer_bert = WordBert(config['bert'])
        self.layer_output = nn.Linear(self.layer_bert.bert_config.hidden_size, self.tag_size)

    def forward(self, batch_data):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        return self.layer_output(words_hidden)

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


class ModelBert_CRF(ModelBert):
    def __init__(self, config):
        super(ModelBert_CRF, self).__init__(config)
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


class ModelBert_Cofe(ExpModelBase):
    def __init__(self, config):
        super(ModelBert_Cofe, self).__init__()
        self.tag_size = config['tag_size']
        self.words_dropout_prob = config['words_dropout_prob']

        self.layer_bert = WordBert(config['bert'])
        self.words_dropout = nn.Dropout(self.words_dropout_prob)
        self.layer_enh = EnhancedCell(config['enh'])

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        words_hidden = self.words_dropout(words_hidden)
        loss = self.layer_enh.forward(words_hidden, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict, output_weight=False, output_Z=False):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        outputs = self.layer_enh.predict(words_hidden, batch_data['lengths'], output_weight=output_weight, output_Z=output_Z)
        if isinstance(outputs, tuple):
            return (torch.argmax(outputs[0], dim=-1), ) + outputs[1:]
        else:
            return torch.argmax(outputs, dim=-1)

    def predict_bs(self, batch_data: dict, beam_width=None):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        return self.layer_enh.predict_bs(words_hidden, batch_data['lengths'], beam_width)

    def load_pretrained(self, pretrained_model_name_or_path):
        return self.layer_bert.load_pretrained(pretrained_model_name_or_path)

    def fix_bert(self):
        return self.set_layer_trainable('layer_bert', False)

    def get_params_by_part(self):
        bert_params = list(filter(lambda p: p.requires_grad, self.layer_bert.parameters()))
        base_params = list(filter(lambda p: id(p) not in list(map(id, bert_params)) and p.requires_grad, self.parameters()))
        return bert_params, base_params


if __name__ == '__main__':
    pass
