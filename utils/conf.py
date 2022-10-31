from .path import *
from .files import *


class ExpConfig(object):
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.setting_file = CONF_SET_EXP_FILE(exp_name)

        self.config = load_json_file(self.setting_file)

        self.dat_name = self.config['dat_name']
        self.mod_name = self.config['mod_name']
        self.for_bert = self.config.get('for_bert', False)
        self.max_seqs = self.config.get('max_seqs', None)
        self.mod_conf = self.config['mod_conf']

        self.dat_model = self.config['dat_model']
        self.bert_pretrained = self.config.get('bert_pretrained', None)

        self.tag_size = self.mod_conf['tag_size']
        self.words_size = self.mod_conf.get('words_size', None)

        self.model_dir = CONF_MOD_EXP_DIR(exp_name)
        self.model_param_dir = CONF_MOD_EXP_DIR_PARAM(exp_name)
        self.model_vocab_dir = CONF_MOD_EXP_DIR_VOCAB(exp_name)
        self.model_vocab = concat_path(self.model_vocab_dir, 'vocab.txt')

        if 'evalmod' in self.mod_conf and 'Bert' not in self.mod_conf['evalmod']['mod_name']:
            self.model_vocab_dir_e = CONF_MOD_EXP_DIR_VOCAB_E(exp_name)
            self.model_vocab_e = concat_path(self.model_vocab_dir_e, 'vocab.txt')
            self.words_size_e = self.mod_conf['evalmod']['config'].get('words_size', None)

        self.dat_trn = RES_DATA_FILE(self.dat_name, 'TRN')
        self.dat_tst = RES_DATA_FILE(self.dat_name, 'TST')
        self.dat_val = RES_DATA_FILE(self.dat_name, 'VAL')
        self.dat_tag = RES_DATA_FILE(self.dat_name, 'TAG')
        self.dat_voc = RES_DATA_FILE(self.dat_name, 'VOC')

    def dat_file(self, file_type):
        return RES_DATA_FILE(self.dat_name, file_type)


if __name__ == '__main__':
    obj = ExpConfig('zh_lstm')
