from data.dataset import *
from data.loader import SingleDataLoader

def imp_exp_dataset(exp_conf, file_type, device=None) -> ExpDatasetBase:
    exp_conf = ExpConfig(exp_conf) if isinstance(exp_conf, str) else exp_conf
    cls = globals()[exp_conf.dat_model]
    return cls(exp_conf, file_type, device)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = imp_exp_dataset('zh_lstm', 'TST')
    # ds = imp_exp_dataset('zh_bert', 'TST')
    # ds = imp_exp_dataset('en_bert', 'TST')
    #
    train_loader = DataLoader(dataset=ds, batch_size=5, shuffle=True, collate_fn=ds.collate)
    print(len(train_loader))
    for data in train_loader:
        print(data)
        break
