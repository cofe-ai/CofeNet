from abc import ABCMeta
from abc import abstractmethod
from pytorch_transformers import BertTokenizer
from utils import load_text_file_by_line, save_text_file_by_line, ExpConfig


class VocabularyBase(metaclass=ABCMeta):
    TK_PAD = '[PAD]'
    TK_UNK = '[UNK]'

    @abstractmethod
    def wd2ids(self, word):
        raise NotImplemented


class VocabularyNormal(VocabularyBase):
    def __init__(self, words: list):
        self.words = words
        self.map_wd2id = {wd: id for id, wd in enumerate(self.words)}
        self.ID_PAD = self.map_wd2id[self.TK_PAD]
        self.ID_UNK = self.map_wd2id[self.TK_UNK]

    def wd2ids(self, word):
        return [self.map_wd2id.get(word, self.ID_UNK)]

    def wd2id(self, word):
        return self.map_wd2id.get(word, self.ID_UNK)

    @classmethod
    def load_vocabulary(cls, exp_conf, eval=False):
        if isinstance(exp_conf, str):
            exp_conf = ExpConfig(exp_conf)

        if eval:
            try:
                words = load_text_file_by_line(exp_conf.model_vocab_e)
            except FileNotFoundError as e:
                words = load_text_file_by_line(exp_conf.dat_voc)
                words = [cls.TK_PAD, cls.TK_UNK] + words[:exp_conf.words_size_e - 2]
                save_text_file_by_line(exp_conf.model_vocab_e, words)

            assert len(words) == exp_conf.words_size_e
            return cls(words)
        else:
            try:
                words = load_text_file_by_line(exp_conf.model_vocab)
            except FileNotFoundError as e:
                words = load_text_file_by_line(exp_conf.dat_voc)
                words = [cls.TK_PAD, cls.TK_UNK] + words[:exp_conf.words_size - 2]
                save_text_file_by_line(exp_conf.model_vocab, words)

            assert len(words) == exp_conf.words_size
            return cls(words)


class VocabularyBert(VocabularyBase):
    TK_CLS = '[CLS]'
    TK_MSK = '[MASK]'
    TK_SEP = '[SEP]'

    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.ID_PAD, self.ID_UNK, self.ID_CLS, self.ID_MSK, self.ID_SEP = \
            self.tokenizer.convert_tokens_to_ids([self.TK_PAD, self.TK_UNK, self.TK_CLS, self.TK_MSK, self.TK_SEP])

    def wd2ids(self, word):
        if not word:
            ret = [self.ID_UNK]
        else:
            ret = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            if not ret:
                ret = [self.ID_UNK]
        ret = [x if x not in [self.ID_PAD, self.ID_CLS, self.ID_MSK, self.ID_SEP] else self.ID_UNK for x in ret]
        return ret

    @classmethod
    def load_vocabulary(cls, exp_conf):
        if isinstance(exp_conf, str):
            exp_conf = ExpConfig(exp_conf)

        tokenizer = BertTokenizer.from_pretrained(exp_conf.model_vocab_dir)
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(exp_conf.bert_pretrained)
            assert tokenizer is not None
            tokenizer.save_pretrained(exp_conf.model_vocab_dir)

        return cls(tokenizer)


if __name__ == '__main__':
    # obj = VocabularyNormal.load_vocabulary('zh_lstm')
    # print('我 -> ' + str(obj.wd2ids('我')))

    # obj = VocabularyBert.load_vocabulary('en2f_bert')
    # print('VocabularyNormal -> ' + str(obj.wd2ids('[CLS]VocabularyNormal')))

    obj = VocabularyBert.load_vocabulary('zh_bert')
    print('我 -> ' + str(obj.wd2ids('[CLS]我')))
    print('\'\' -> ' + str(obj.wd2ids('')))
    print('\'\' -> ' + str(obj.wd2ids('')))
    print('\'\' -> ' + str(obj.wd2ids('')))
    print('\'\' -> ' + str(obj.wd2ids('')))
    print('\'\' -> ' + str(obj.wd2ids('')))
    print('VocabularyNormal -> ' + str(obj.wd2ids('VocabularyNormal')))
