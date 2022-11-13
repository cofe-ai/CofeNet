import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from .funs import *


class WordBert(nn.Module):
    def __init__(self, config):
        super(WordBert, self).__init__()
        if isinstance(config, dict):
            self.bert_config = BertConfig.from_dict(config)
        elif isinstance(config, str):
            self.bert_config = BertConfig.from_pretrained(config)
        else:
            raise ValueError
        self.bert = BertModel(self.bert_config)

    def load_pretrained(self, pretrained_model_name_or_path):
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path, config=self.bert_config)
        return self

    def forward(self, input_ids, attention_mask, wdlens, output_num=1):
        """
        :param input_ids: [batch, tk_seq + 1]
        :param attention_mask: [batch, tk_seq + 1] or None
        :param wdlens: [batch, wd_seq]
        :return: [batch, wd_seq, hidden]
        """
        # bert_output = [batch, tk_seq, hidden]
        # bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        bert_output = self.bert(input_ids, attention_mask=attention_mask)

        if len(bert_output) <= 2 or output_num == 1:
            layer_hidden = bert_output[0]
            layer_hidden = layer_hidden[:, 1:, :]  # drop CLS token

            # token -> word
            output_tensor = [get_word_rep_from_subword(tensor_item, len_item)
                             for tensor_item, len_item in zip(layer_hidden, wdlens)]
            return pad_sequence_with_max_len(output_tensor, batch_first=True, max_len=-1)
        else:
            outputs = tuple()
            for layer_hidden in bert_output[2][-output_num:]:
                layer_hidden = layer_hidden[:, 1:, :]  # drop CLS token

                # token -> word
                output_tensor = [get_word_rep_from_subword(tensor_item, len_item)
                                 for tensor_item, len_item in zip(layer_hidden, wdlens)]
                outputs += (pad_sequence_with_max_len(output_tensor, batch_first=True, max_len=-1),)
            return outputs

    def forward_text_rep(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        layer_hidden = bert_output[0]
        return layer_hidden[:, 0, :]  # CLS token


if __name__ == '__main__':
    pass
