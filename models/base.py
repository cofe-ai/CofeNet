import torch.nn as nn


class ExpModelBase(nn.Module):
    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        raise NotImplemented

    def predict(self, batch_data):
        raise NotImplemented

    def load_pretrained(self, pretrained_model_name_or_path):
        raise NotImplemented

    def fix_bert(self):
        raise NotImplemented

    def get_params_by_part(self):
        raise NotImplemented

    def set_layer_trainable(self, layer_name, trainable=False):
        if hasattr(self, layer_name):
            layer = getattr(self, layer_name)
            for p in layer.parameters():
                p.requires_grad = trainable
        return self
