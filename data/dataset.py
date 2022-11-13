import torch
from tqdm import tqdm

from torch.utils.data import Dataset
from utils import *
from .vocab import VocabularyNormal, VocabularyBert


class ExpDatasetBase(Dataset):
    LBID_IGN = -1

    def __init__(self, exp_conf, file_type, device=None):
        self.device = torch.device('cpu') if device is None else device

        self.exp_conf = ExpConfig(exp_conf) if isinstance(exp_conf, str) else exp_conf

        self.map_tg2tgid = {tag: idx for idx, tag in enumerate(load_text_file_by_line(self.exp_conf.dat_tag))}
        self.map_tgid2tg = {idx: tag for tag, idx in self.map_tg2tgid.items()}
        assert len(self.map_tg2tgid) == self.exp_conf.tag_size

        # assert file_type in ['TRN', 'TST', 'VAL']
        self.file_type = file_type
        if self.file_type in ['TRN', 'TST', 'VAL']:
            self.org_data = load_data_from_file(self.exp_conf.dat_file(self.file_type), self.exp_conf.max_seqs)
        else:
            self.org_data = load_data_from_file(self.file_type, self.exp_conf.max_seqs)

    def set_device(self, device):
        self.device = device
        return self

    def collate(self, batch):
        raise NotImplementedError

    def _calc_dis_weight(self, lbids, dis_type: str):
        ret = np.array([1.0 if dis_type in self.map_tgid2tg[lbid] else 0.0 for lbid in lbids])
        n_ret = (ret + 1) % 2

        ret /= ret.sum()
        ret[np.isnan(ret)] = 0.0

        n_ret /= n_ret.sum()
        n_ret[np.isnan(n_ret)] = 0.0
        return list(ret), list(n_ret)

    def _calc_dis_mask(self, lbids, dis_type: str):
        ret = np.array([1 if dis_type in self.map_tgid2tg[lbid] else 0 for lbid in lbids])
        n_ret = (ret + 1) % 2
        return list(ret), list(n_ret)


class DatasetBert(ExpDatasetBase):
    def __init__(self, exp_conf, file_type, device=None):
        super().__init__(exp_conf, file_type, device)

        self.vocab = VocabularyBert.load_vocabulary(self.exp_conf)
        self.tkidss, self.wdlenss, self.lbidss, self.tk_lengths, self.wd_lengths = [], [], [], [], []
        for item in tqdm(self.org_data, desc='Data \'%s\' loading ...' % file_type):
            tkids, wdlens, lbids = [self.vocab.ID_CLS], [], []

            for wd, tg in zip(item['tokens'], item['labels']):
                wd_tkids = self.vocab.wd2ids(wd)
                tkids.extend(wd_tkids)
                wdlens.append(len(wd_tkids))
                lbids.append(self.map_tg2tgid[tg])

            self.tkidss.append(tkids)
            self.wdlenss.append(wdlens)
            self.lbidss.append(lbids)

            self.tk_lengths.append(len(tkids))
            self.wd_lengths.append(len(wdlens))

    def __len__(self):
        return len(self.tkidss)

    def __getitem__(self, idx):
        return {
            'tkids': self.tkidss[idx],
            'lbids': self.lbidss[idx],
            'wdlens': self.wdlenss[idx],
            'tk_length': self.tk_lengths[idx],
            'wd_length': self.wd_lengths[idx],
            'lbstrs': self.org_data[idx]['labels']
        }

    def collate(self, batch):
        """
        And for DataLoader `collate_fn`.
        :param batch: list of {
                'tkids': [tkid, tkid, ...],
                'lbids': [lbid, lbid, ...],
                'wdlens': [wdlen, wdlen, ...],
                'tk_length': len('tkids'),
                'wd_length': len('lbids') or len('wdlens'),
                'lbstrs': len('lbids')
            }
        :return: (
                    {
                        'tkidss': tensor[batch, seq],
                        'attention_mask': tensor[batch, seq],
                        'wdlens': tensor[batch, seq],
                        'lengths': tensor[batch],
                    }
                    ,
                    lbidss: tensor[batch, seq]
                    ,
                    lbstrss: list[list[string]]
            )
        """
        tk_lengths = [item['tk_length'] for item in batch]
        wd_lengths = [item['wd_length'] for item in batch]
        tk_max_length = max(tk_lengths)
        wd_max_length = max(wd_lengths)

        tkidss, attention_mask, wdlens, lbidss, lbstrss = [], [], [], [], []
        for item in batch:
            tk_num_pad = tk_max_length - item['tk_length']
            wd_num_pad = wd_max_length - item['wd_length']

            tkidss.append(item['tkids'] + [self.vocab.ID_PAD] * tk_num_pad)
            attention_mask.append([1] * item['tk_length'] + [0] * tk_num_pad)
            wdlens.append(item['wdlens'] + [0] * wd_num_pad)
            lbidss.append(item['lbids'] + [self.LBID_IGN] * wd_num_pad)
            lbstrss.append(item['lbstrs'])

        output = {
            'tkidss': torch.tensor(tkidss).to(self.device),
            'attention_mask': torch.tensor(attention_mask).to(self.device),
            'wdlens': torch.tensor(wdlens).to(self.device),
            'lengths': torch.tensor(wd_lengths).to(self.device)
        }

        lbidss = torch.tensor(lbidss).to(self.device)
        return output, lbidss, lbstrss


class DatasetLSTM(ExpDatasetBase):
    def __init__(self, exp_conf, file_type, device=None):
        super().__init__(exp_conf, file_type, device)

        self.vocab = VocabularyNormal.load_vocabulary(self.exp_conf)
        self.tkidss, self.lbidss, self.lengths = [], [], []
        for item in tqdm(self.org_data, desc='Data \'%s\' loading ...' % file_type):
            tkids, lbids = [], []
            for tk, tg in zip(item['tokens'], item['labels']):
                tkids.append(self.vocab.wd2id(tk))
                lbids.append(self.map_tg2tgid[tg])

            self.tkidss.append(tkids)
            self.lbidss.append(lbids)
            self.lengths.append(len(tkids))

    def __len__(self):
        return len(self.tkidss)

    def __getitem__(self, idx):
        return {
            'tkids': self.tkidss[idx],
            'lbids': self.lbidss[idx],
            'length': self.lengths[idx],
            'lbstrs': self.org_data[idx]['labels']
        }

    def collate(self, batch):
        """
        And for DataLoader `collate_fn`.
        :param batch: list of {
                'tkids': [tkid, tkid, ...],
                'lbids': [lbid, lbid, ...],
                'length': len('tkids') or len('lbids'),
                'lbstrs': len('lbids')
            }
        :return: (
                    {
                        'tkidss': tensor[batch, seq],
                        'lengths': tensor[batch],
                    }
                    ,
                    lbidss: tensor[batch, seq]
                    ,
                    lbstrss: list[list[string]]
            )
        """
        lengths = [item['length'] for item in batch]
        max_length = max(lengths)

        tkidss, lbidss, lbstrs = [], [], []
        for item in batch:
            num_pad = max_length - item['length']
            tkidss.append(item['tkids'] + [self.vocab.ID_PAD] * num_pad)
            lbidss.append(item['lbids'] + [self.LBID_IGN] * num_pad)
            lbstrs.append(item['lbstrs'])

        output = {
            'tkidss': torch.tensor(tkidss).to(self.device),
            'lengths': torch.tensor(lengths).to(self.device)
        }

        lbidss = torch.tensor(lbidss).to(self.device)
        return output, lbidss, lbstrs


if __name__ == '__main__':
    pass
