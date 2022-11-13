from .dataset import *
from .loader import SingleDataLoader


def imp_exp_dataset(exp_conf, file_type, device=None) -> ExpDatasetBase:
    exp_conf = ExpConfig(exp_conf) if isinstance(exp_conf, str) else exp_conf
    cls = globals()[exp_conf.dat_model]
    return cls(exp_conf, file_type, device)


if __name__ == '__main__':
    pass
