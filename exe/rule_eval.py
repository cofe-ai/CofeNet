from utils import *
from exe.eval import evaluate_extraction
from exe.executor import Executor


def match(tokens: list, target_str: str, lang: str, bng: int):
    # space = ' ' if lang == 'en' else ''
    space = ''
    all_length = len(tokens)
    tgt_length = len(target_str) if lang == 'zh' else len(target_str.split(' '))
    if lang == 'en':
        target_str = target_str.replace(' ', '')
    assert tgt_length > 0
    for ii in range(bng, all_length - tgt_length, 1):
        if target_str.startswith(tokens[ii]):
            if space.join(tokens[ii: ii + tgt_length]) == target_str:
                return ii, ii + tgt_length
    return None, None


MAP_TAG_TP = {
    'name': 'source',
    'trigger': 'cue',
    'content': 'content'
}


def entity_bio_seq(length: int, tp: str):
    tp = MAP_TAG_TP[tp]
    return ['B-' + tp] + ['I-' + tp] * (length - 1)


def get_pred_tag_seq(tokens, tmp_pred, lang):
    point = 0
    preds = ['O'] * len(tokens)
    for _, item in tmp_pred.items():
        if item:
            nxt_point = -1
            for name, val in item[0].items():
                if val is not None:
                    bng, end = match(tokens, val, lang, point)

                    if bng is not None:  # get
                        nxt_point = max(nxt_point, end)
                        tp = MAP_TAG_TP[name]
                        preds[bng] = 'B-' + tp
                        for ix in range(bng + 1, end, 1):
                            preds[ix] = 'I-' + tp
            if nxt_point > 0:
                point = nxt_point
    return preds


def eval_dataset(dataset: str, lang: str = 'en'):
    tst_datas_file = RES_DATA_FILE(dataset, "TST")
    tmp_preds_file = concat_path(RES_DATA_DIR('rule_tmp'), dataset + '_tmp.txt')

    tst_datas = load_json_file_by_line(tst_datas_file)
    tmp_preds = load_json_file_by_line(tmp_preds_file)

    assert len(tst_datas) == len(tmp_preds)

    labels = [item['labels'] for item in tst_datas]
    preds = []
    for tst_data, tmp_pred in zip(tst_datas, tmp_preds):
        preds.append(get_pred_tag_seq(tst_data['tokens'], tmp_pred, lang))

    return evaluate_extraction(labels, preds)

# polnear-v2-fixed
print('polnear-v2-fixed')
print(Executor.format_result(eval_dataset('polnear-v2-fixed', 'en'), markdown_table=True))

# riqua
print('riqua2')
print(Executor.format_result(eval_dataset('riqua2', 'en'), markdown_table=True))

# zh
print('zh')
print(Executor.format_result(eval_dataset('zh', 'zh'), markdown_table=True))
