#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from utils import get_gpus_meminfo, get_best_device, cuda_is_available
from exe import Trainer
from utils import set_global_rand_seed

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='zh_lstm')
parser.add_argument("--trn_name", type=str, default='v1')

parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--use_cpu", default=False, action='store_true')

parser.add_argument('--show_per_step', type=int, default=10)
parser.add_argument("--eval_per_step", type=int, default=250)
parser.add_argument("--min_eval_step", type=int, default=100)
parser.add_argument("--eval_type", type=str, default='bio_f1', choices=["bio_f1", "exact_f1_avg"])
parser.add_argument('--max_mod_saved_num', type=int, default=2)
parser.add_argument("--do_not_save_mod", default=False, action='store_true')

parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--optim', type=str, default="Adam", choices=["Adam", "AdamW", "Adadelta", "RMSprop", "Adagrad"])

parser.add_argument("--max_epoch", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument("--bert_learning_rate", type=float, default=1e-4) # 5e-5
parser.add_argument('--bert_weight_decay', type=float, default=0.0)
parser.add_argument("--fix_bert", default=False, action='store_true')
parser.add_argument("--batch_size", type=int, default=32)
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

Trainer(params).train()
