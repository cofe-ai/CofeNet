#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from utils import get_gpus_meminfo, get_best_device, cuda_is_available
from exe import Executor

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='pn_bert_cofe')
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--use_cpu", default=False, action='store_true')
parser.add_argument("--mod_index", type=int, default=-1)
params = parser.parse_args().__dict__

# select the good gpu/cpu
if not params['use_cpu'] and cuda_is_available():
    if (params['gpu'] is None) or (params['gpu'] not in get_gpus_meminfo()[0]):
        params['gpu'] = get_best_device()
else:
    params['gpu'] = None
params.pop('use_cpu')

if params['gpu'] is None:
    print('device: CPU')
else:
    print('device: GPU %d' % params['gpu'])

mod = Executor(params['exp_name'], params['gpu']).load_model(mod_index=params['mod_index'])
ret = Executor.format_result(mod.eval_dataset_tst(), markdown_table=True)
print(ret)
