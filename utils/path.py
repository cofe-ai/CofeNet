import os

from .data import create_vocabulary_from_files


def define_dir(*args):
    folder = os.path.abspath(os.path.join(*args))
    if not os.path.exists(folder):
        os.makedirs(folder)
    assert os.path.isdir(folder), '[ERROR] \'%s\' is not a folder path!' % folder
    return folder


def concat_path(*args):
    return os.path.abspath(os.path.join(*args))


RES_DIR = define_dir(os.path.dirname(__file__), os.pardir, 'res')

MAP_TYPE_NAME = {
    'TRN': 'train.txt',
    'TST': 'test.txt',
    'VAL': 'valid.txt',
    'TAG': 'tag.txt',
    'VOC': 'voc.txt'
}


def RES_DATA_DIR(dataset):
    return concat_path(RES_DIR, dataset)


def RES_DATA_FILE(dataset, file_type):
    try:
        file_path = concat_path(RES_DATA_DIR(dataset), MAP_TYPE_NAME[file_type])
    except KeyError as e:
        raise KeyError('file_type = \'%s\' not in %s' % (file_type, str(list(MAP_TYPE_NAME.keys()))))

    if not os.path.exists(file_path):
        if file_type != 'VOC':
            raise FileNotFoundError(file_path)

        create_vocabulary_from_files([RES_DATA_FILE(dataset, 'TRN'), RES_DATA_FILE(dataset, 'VAL')], file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

    return file_path


CONF_DIR = define_dir(os.path.dirname(__file__), os.pardir, 'conf')
CONF_SET_DIR = concat_path(CONF_DIR, 'setting')
CONF_MOD_DIR = concat_path(CONF_DIR, 'models')


def CONF_SET_EXP_FILE(exp_name: str):
    file_path = concat_path(CONF_SET_DIR, exp_name + '.json')
    assert os.path.exists(file_path), 'Can not find Experiment setting \'%s\'' % exp_name
    return file_path


def CONF_MOD_EXP_DIR(exp_name: str):
    return define_dir(CONF_MOD_DIR, exp_name)


def CONF_MOD_EXP_DIR_PARAM(exp_name: str):
    return define_dir(CONF_MOD_EXP_DIR(exp_name), 'param')


def CONF_MOD_EXP_DIR_VOCAB(exp_name: str):
    return define_dir(CONF_MOD_EXP_DIR(exp_name), 'vocab')


def CONF_MOD_EXP_DIR_VOCAB_E(exp_name: str):
    return define_dir(CONF_MOD_EXP_DIR(exp_name), 'vocab_e')


LOG_DIR = define_dir(os.path.dirname(__file__), os.pardir, 'log')

if __name__ == '__main__':
    pass
