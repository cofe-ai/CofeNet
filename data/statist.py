from utils.path import *
from utils.data import *


def statist_dataset(dataset_name):
    data_trn = load_data_from_file(RES_DATA_FILE(dataset_name, 'TRN'), max_length_text=-1)
    data_tst = load_data_from_file(RES_DATA_FILE(dataset_name, 'TST'), max_length_text=-1)
    data_dev = load_data_from_file(RES_DATA_FILE(dataset_name, 'VAL'), max_length_text=-1)
    data_all = data_trn + data_tst + data_dev

    ret_titles = '| %-20s |' % 'data'
    ret_splits = '| %-20s |' % ':----'
    ret_values = '| %-20s |' % dataset_name

    # all items num
    ret_titles += ' %10s |' % 'all docs'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10d |' % len(data_all)

    # trn items num
    ret_titles += ' %10s |' % 'trn docs'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10d |' % len(data_trn)

    # dev items num
    ret_titles += ' %10s |' % 'dev docs'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10d |' % len(data_dev)

    # tst items num
    ret_titles += ' %10s |' % 'tst docs'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10d |' % len(data_tst)

    # doc length
    size_all = len(data_all)
    ret_titles += ' %10s |' % 'doc length'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(sum([len(it['tokens']) for it in data_all])) / size_all)

    num_src, num_cue, num_cnt = 0, 0, 0
    len_src, len_cue, len_cnt = 0, 0, 0
    for item in data_all:
        for tag in item['labels']:
            if tag.endswith('source'):
                len_src += 1
                if tag[0] == 'B':
                    num_src += 1
            elif tag.endswith('cue'):
                len_cue += 1
                if tag[0] == 'B':
                    num_cue += 1
            elif tag.endswith('content'):
                len_cnt += 1
                if tag[0] == 'B':
                    num_cnt += 1

    # src/doc
    ret_titles += ' %10s |' % 'src/doc'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(num_src) / size_all)

    # cue/doc
    ret_titles += ' %10s |' % 'cue/doc'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(num_cue) / size_all)

    # cnt/doc
    ret_titles += ' %10s |' % 'cnt/doc'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(num_cnt) / size_all)

    # src length
    ret_titles += ' %10s |' % 'src length'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(len_src) / num_src)

    # cue length
    ret_titles += ' %10s |' % 'cue length'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(len_cue) / num_cue)

    # cnt length
    ret_titles += ' %10s |' % 'cnt length'
    ret_splits += ' %10s |' % ':----:'
    ret_values += ' %10.2f |' % (float(len_cnt) / num_cnt)

    print(ret_titles)
    print(ret_splits)
    print(ret_values)
    print('')


statist_dataset('polnear-v2-fixed')
statist_dataset('riqua2')
statist_dataset('zh')