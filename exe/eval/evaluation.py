# coding: utf-8
#
# Copyright 2021 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# model evaluation

from __future__ import unicode_literals, print_function, division

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import List

#from .extraction import get_element, get_spans
from exe.eval.extraction import get_element, get_spans


def eva_classifier(list_t, list_p, labels=None, average='binary'):
    c_m = confusion_matrix(list_t, list_p, labels=labels)
    acc = accuracy_score(list_t, list_p)
    rec = recall_score(list_t, list_p, labels=labels, average=average)
    pre = precision_score(list_t, list_p, labels=labels, average=average)
    f1 = f1_score(list_t, list_p, labels=labels, average=average)
    f1_micro = f1_score(list_t, list_p, labels=labels, average='micro')
    f1_macro = f1_score(list_t, list_p, labels=labels, average='macro')

    return {'c_m': c_m, 'acc': acc, 'f1': f1, 'pre': pre, 'rec': rec, 'f1_macro': f1_macro, 'f1_micro': f1_micro}


def evaluate_bio_multiple_classification(y_true, y_pred):
    dict_eva = dict()
    bio_y, bio_p = [], []

    for list_t1, list_t2 in zip(y_true, y_pred):
        for t1, t2 in zip(list_t1, list_t2):
            if t1 == '<IGN>':
                break
            bio_y.append(t1)
            bio_p.append(t2)
    dict_eva['bio'] = eva_classifier(
        bio_y, bio_p, labels=['O', 'B-source', 'I-source', 'B-cue', 'I-cue', 'B-content', 'I-content'], average='macro')
    dict_eva['begin_multiple'] = eva_classifier(
        bio_y, bio_p, labels=['B-source', 'B-cue', 'B-content'], average='macro')
    return dict_eva


def evaluate_extraction(y_true_bio: List[List[str]], y_pred_bio: List[List[str]]):
    dict_y_true = {
        'begin': {'source': [], 'cue': [], 'content': []},
        'exact_match': {'source': [], 'cue': [], 'content': []},
        'jaccard': {'source': [], 'cue': [], 'content': []}
    }
    dict_y_pred = {
        'begin': {'source': [], 'cue': [], 'content': []},
        'exact_match': {'source': [], 'cue': [], 'content': []},
        'jaccard': {'source': [], 'cue': [], 'content': []}
    }
    for inst_true, inst_pred in zip(y_true_bio, y_pred_bio):
        dict_tmp_true = get_element(inst_true)
        dict_tmp_pred = get_element(inst_pred)

        for k in ['source', 'cue', 'content']:
            v_tmp_true = dict_tmp_true[k] if k in dict_tmp_true else []
            v_tmp_pred = dict_tmp_pred[k] if k in dict_tmp_pred else []
            for type_match in ['begin', 'exact_match', 'jaccard']:
                tmp_true, tmp_pred = cal_evaluation_detail(v_tmp_true, v_tmp_pred, type_match)
                dict_y_true[type_match][k].extend(tmp_true)
                dict_y_pred[type_match][k].extend(tmp_pred)

    dict_eva = evaluate_bio_multiple_classification(y_true_bio, y_pred_bio)
    for k1 in ['begin', 'exact_match', 'jaccard']:
        if k1 not in dict_eva:
            dict_eva[k1] = dict()
        for k2 in ['source', 'cue', 'content']:
            if k1 != 'jaccard':
                dict_eva[k1][k2] = eva_classifier(dict_y_true[k1][k2], dict_y_pred[k1][k2], average='binary')
            else:
                dict_eva[k1][k2] = np.mean(np.array(dict_y_true[k1][k2]) * np.array(dict_y_pred[k1][k2]))
    return dict_eva


def cal_evaluation_detail(list_idx_true, list_idx_pred, type_match: str):
    y_exact_match_true, y_exact_match_pred = [], []
    if type_match == 'begin':
        list_idx_true = [tmp[0] for tmp in list_idx_true]
        list_idx_pred = [tmp[0] for tmp in list_idx_pred]
    for tmp in list_idx_true:
        y_exact_match_true.append(1)
        if tmp in list_idx_pred:
            y_exact_match_pred.append(1)
        else:
            if type_match == 'exact_match':
                y_exact_match_pred.append(0)
            elif type_match == 'begin':
                y_exact_match_pred.append(0)
            else:
                y_exact_match_pred.append(
                    max([cal_overlap(tmp, _tmp) for _tmp in list_idx_pred]) if len(list_idx_pred) > 0 else 0)
    for tmp in list_idx_pred:
        if tmp in list_idx_true:
            continue
        else:
            if type_match == 'begin':
                y_exact_match_true.append(0)
                y_exact_match_pred.append(1)
            elif (type_match == 'jaccard') and (len(list_idx_true) > 0) and (max([cal_overlap(tmp, _tmp) for _tmp in list_idx_true]) > 0):
                # do nothing, avoid the
                pass
            else:
                y_exact_match_true.append(0)
                y_exact_match_pred.append(1)
    assert len(y_exact_match_true) == len(y_exact_match_pred)
    return y_exact_match_true, y_exact_match_pred


def cal_overlap(a: [int, int], b: [int, int]):
    assert len(a) == len(b) == 2
    assert a[1] > a[0]
    assert b[1] > b[0]
    tmp = min(a[1] - b[0], b[1] - a[0], b[1] - b[0], a[1] - a[0]) / (max(a[1], b[1]) - min(a[0], b[0]))
    return max(0, tmp)


if __name__ == '__main__':
    # tmp1 = [[1, 3], [5, 7], [8, 9], [9, 19]]
    # tmp2 = [[1, 3], [5, 7], [12, 13], [20, 24]]
    # print(cal_evaluation_detail(tmp1, tmp2, 'begin'))
    # print(cal_evaluation_detail(tmp1, tmp2, 'exact_match'))
    # print(cal_evaluation_detail(tmp1, tmp2, 'jaccard'))
    # exit()
    '''
    np.random.seed(2019)
    y_true_M = np.random.randint(0, 3, 10)
    # y_pred_M = np.array([1, 0, 2, 0, 1, 1, 0, 1, 0, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0], dtype='float64')
    y_pred_prob = []
    for i in range(len(y_true_M)):
        y_pred_prob.append(np.random.random(3))
    print(y_true_M, y_pred_prob)
    '''
    # true_test = ["B-source", "I-cue", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
    #              "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
    #              "O", "O", "O", "O", "O", "O", "O", "B-content", "O", "O", "O", "O", "O", "O", "O"]
    # pred_test = ["B-source", "I-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
    #              "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
    #              "O", "O", "O", "O", "O", "O", "O", "B-content", "O", "O", "O", "O", "O", "O", "O"]
    # pred_test = [["O", "O", "O", "O", "O", "O", "O"]]
    # report!
    # print(evaluate_extraction(true_test, pred_test))

    # true_test = ["O", "O", "B-cue", "I-cue", "I-cue", "O", "O", "O", "O", "O",
    #              "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
    #              "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-cue", "I-cue", "O"]
    # pred_test = ["O", "O", "B-cue", "I-cue", "I-cue", "O", "O", "O", "O", "O",
    #              "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
    #              "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-cue", "B-cue", "B-cue"]

    true_test = ["B-cue", "I-cue", "O", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
                 "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
                 "O", "O", "O", "O", "O", "O", "O", "B-content", "O", "O", "O", "O", "O", "O", "O"]
    pred_test = ["O", "B-cue", "O", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
                 "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
                 "O", "O", "O", "O", "O", "O", "O", "B-content", "O", "O", "O", "O", "O", "O", "O"]

    aaa = evaluate_extraction([true_test], [pred_test])
    print(evaluate_extraction([true_test], [pred_test]))
    print('begin:', aaa['begin']['cue']['f1'])
    print('exact:', aaa['exact_match']['cue']['f1'])

    # pred_test_2 = ["O"] * len(true_test)
    #
    # print(evaluate_extraction([true_test], [pred_test_2]))

