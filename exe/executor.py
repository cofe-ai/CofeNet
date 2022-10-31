import copy
import torch
import pandas as pd
from torch.utils.data import SequentialSampler
from sklearn.metrics import accuracy_score
from data import SingleDataLoader

from .eval import evaluate_extraction
from models import imp_exp_model
from data import imp_exp_dataset
from utils import *


class Executor(object):
    def __init__(self, exp_conf, gpu):
        self.exp_conf = ExpConfig(exp_conf) if isinstance(exp_conf, str) else exp_conf
        self.device = torch.device('cpu') if gpu is None else torch.device("cuda", gpu)
        self.model_param_dir = self.exp_conf.model_param_dir

        # model
        self.model = imp_exp_model(self.exp_conf.mod_name, self.exp_conf.mod_conf).to(self.device)

        # load tst data
        self.data_tst = imp_exp_dataset(self.exp_conf, 'TST', self.device)
        self.map_tgid2tg = self.data_tst.map_tgid2tg

        # log dir
        self.log_root_dir = LOG_DIR

        self.fun_rule = None

    def _get_params_file_path(self, mod_index=-1):
        select_index = mod_index
        if mod_index < 0:
            bng_idx, end_idx = len('model_'), len('.bin')
            indexs = [int(name[bng_idx: -end_idx]) for name in os.listdir(self.model_param_dir) if
                      len(name) > (bng_idx + end_idx)]
            assert indexs, 'no model file in %s' % self.model_param_dir
            select_index = max(indexs)

        return concat_path(self.model_param_dir, 'model_%d.bin' % select_index)

    def load_model(self, mod_index=-1):
        file_path = self._get_params_file_path(mod_index)
        print('Load %d -> \'%s\' ..' % (mod_index, file_path))
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        return self

    def save_model(self, mod_index):
        file_path = self._get_params_file_path(mod_index)
        print('Save %d -> \'%s\' ..' % (mod_index, file_path))
        torch.save(self.model.state_dict(), file_path)
        return self

    def get_preds_trues(self, dataset, batch_size=32, beam_width=None):
        dataloder = SingleDataLoader(dataset=dataset, batch_size=batch_size,
                                     sampler=SequentialSampler(dataset), collate_fn=dataset.collate)

        preds, labels = [], []
        for batch_data, _, lbstrss in dataloder:
            self.model.eval()
            with torch.no_grad():
                if beam_width is None:
                    batch_preds = self.model.predict(batch_data)
                else:
                    batch_preds = self.model.predict_bs(batch_data, beam_width)

            batch_pred_strs = self.tgidss2tgstrss(
                batch_preds.data.cpu().numpy() if not isinstance(batch_preds, list) else batch_preds,
                batch_data['lengths'].cpu().numpy())

            if self.fun_rule is not None:
                batch_pred_strs = self.fun_rule(batch_pred_strs)

            preds.extend(batch_pred_strs)
            labels.extend(lbstrss)
        return preds, labels

    def eval_dataset(self, dataset, batch_size=32, beam_width=None):
        """
        :return:    {
                'bio':              {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},

                'begin_multiple':   {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},

                'begin': {
                    'source':   {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},
                    'cue':      {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},
                    'content':  {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},
                },

                'exact_match': {
                    'source':   {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},
                    'cue':      {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},
                    'content':  {'c_m': m, 'acc': x, 'f1': x, 'pre': x, 'rec': x, 'f1_macro': x, 'f1_micro': x},
                },

                'jaccard': {
                    'source':   x,
                    'cue':      x,
                    'content':  x
            }
        """
        preds, labels = self.get_preds_trues(dataset, batch_size, beam_width)
        return evaluate_extraction(labels, preds)

    def log_worst_items(self, dataset, batch_size=32, max_log_items=0.3, beam_width=None):
        preds, labels = self.get_preds_trues(dataset, batch_size, beam_width)
        tuple_idx_accs = [(idx, accuracy_score(label, pred)) for idx, (pred, label) in enumerate(zip(preds, labels))]
        tuple_idx_accs = sorted(tuple_idx_accs, key=lambda x: x[1], reverse=False)

        max_log_items = len(tuple_idx_accs) * max_log_items if max_log_items < 1 else int(max_log_items)
        worst_items = []
        for (idx, acc) in tuple_idx_accs:
            if len(worst_items) >= max_log_items or acc >= 1.0:
                break
            log_item = {
                'idx': idx,
                'acc': acc,
                'preds': preds[idx],
                'trues': labels[idx],
                'tokens': dataset.org_data[idx]['tokens']
            }
            worst_items.append(log_item)
        output_file = concat_path(self.log_root_dir, '%s-worst_items.txt' % self.exp_conf.exp_name)
        save_json_file_by_line(output_file, worst_items)

        output_file_2 = concat_path(self.log_root_dir, '%s-worst_items_compare.txt' % self.exp_conf.exp_name)
        with open(output_file_2, 'w') as f:
            for item in worst_items:
                f.write('\n' + ('=' * 120) + '\n')
                f.write('idx: %d acc: %.4f\n' % (item['idx'], item['acc']))
                tk_lengths = [max(max(len(pred), 6), max(len(true), 6), len(token)) for (pred, true, token) in
                              zip(item['preds'], item['trues'], item['tokens'])]
                for title in ['tokens', 'trues', 'preds']:
                    f.write("%s: " % title.center(8))
                    f.write(" ".join([pred.center(tk_lengths[ii]) for ii, pred in enumerate(item[title])]))
                    f.write("\n")

                f.write("%s  " % ' '.center(8))
                f.write(" ".join([(' ' if true == pred else '*').center(tk_lengths[ii]) for ii, (true, pred) in
                                  enumerate(zip(item['trues'], item['preds']))]))
                f.write("\n")

        return worst_items

    def log_worst_items_by_tst(self, batch_size=32, max_log_items=0.3):
        return self.log_worst_items(self.data_tst, batch_size, max_log_items)

    @staticmethod
    def format_result(eval_result: dict, markdown_table=False) -> str:
        def format_single_result(single):
            sret = ''
            sret += str(single['c_m'])
            single.pop('c_m')
            sret += '\n' + str(single) + '\n'
            return sret

        ret = ''
        result = copy.deepcopy(eval_result)

        # bio
        ret += '# bio\n' + format_single_result(result['bio'])
        ret += '\n'

        # bio
        ret += '# begin_multiple\n' + format_single_result(result['begin_multiple'])
        ret += '\n'

        # begin->source
        ret += '# begin / source\n' + format_single_result(result['begin']['source'])
        # begin->cue
        ret += '# begin / cue\n' + format_single_result(result['begin']['cue'])
        # begin->content
        ret += '# begin / content\n' + format_single_result(result['begin']['content'])
        ret += '\n'

        # exact_match->source
        ret += '# exact_match / source\n' + format_single_result(result['exact_match']['source'])
        # exact_match->cue
        ret += '# exact_match / cue\n' + format_single_result(result['exact_match']['cue'])
        # exact_match->content
        ret += '# exact_match / content\n' + format_single_result(result['exact_match']['content'])
        ret += '\n'

        # jaccard
        ret += '# jaccard\n' + str(result['jaccard']) + '\n'

        if markdown_table:
            # for markdown
            ret += '\n'
            ret += ('|' + ' ' * 16 + '|')
            for n in ['src', 'cue', 'cnt']:
                for s in ['pre', 'rec', 'f1', 'acc']:
                    ret += (n + '-' + s).center(8) + '|'
            ret += '\n'

            ret += '|' + ':----'.center(16) + '|' + (':----:'.center(8) + '|') * 12
            ret += '\n'

            # exact_match
            ret += '|' + 'exact match'.center(16) + '|'
            for n in ['source', 'cue', 'content']:
                for s in ['pre', 'rec', 'f1', 'acc']:
                    ret += ('%.4f' % eval_result['exact_match'][n][s]).center(8) + '|'
            ret += '\n'

            # begin
            ret += '|' + 'only begin'.center(16) + '|'
            for n in ['source', 'cue', 'content']:
                for s in ['pre', 'rec', 'f1', 'acc']:
                    ret += ('%.4f' % eval_result['begin'][n][s]).center(8) + '|'
            ret += '\n'

            # jaccard
            ret += '\n'
            ret += '|%16s|%8s|%8s|%8s|' % (' ', 'source', 'cue', 'content') + '\n'
            ret += '|%16s|%8s|%8s|%8s|' % (':----', ':----:', ':----:', ':----:') + '\n'
            ret += '|%16s|%8.4f|%8.4f|%8.4f|' % \
                   ('jaccard', eval_result['jaccard']['source'], eval_result['jaccard']['cue'],
                    eval_result['jaccard']['content']) + '\n'

            # all
            ret += '\n'
            ret += '|exa#src |exa#cue |exa#cnt |bng#src |bng#cue |bng#cnt |jac#src |jac#cue |jac#cnt |\n'
            ret += '| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |\n'
            ret += '| %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n' % (
                eval_result['exact_match']['source']['f1'], eval_result['exact_match']['cue']['f1'],
                eval_result['exact_match']['content']['f1'],
                eval_result['begin']['source']['f1'], eval_result['begin']['cue']['f1'],
                eval_result['begin']['content']['f1'],
                eval_result['jaccard']['source'], eval_result['jaccard']['cue'], eval_result['jaccard']['content']
            )

        return ret

    def eval_dataset_tst(self, batch_size=32, beam_width=None):
        return self.eval_dataset(self.data_tst, batch_size, beam_width)

    def tgidss2tgstrss(self, tgidss, lengths=None):
        tgstrss = []
        if lengths is None:
            for tgids in tgidss:
                tgstrss.append([self.map_tgid2tg[tgid] for tgid in tgids])
        else:
            for tgids, length in zip(tgidss, lengths):
                tgstrss.append([self.map_tgid2tg[tgid] for tgid in tgids[:length]])
        return tgstrss

    def save_single_case_weight_by_cofe(self, dataset_file, output_file, batch_size=32):
        dataset = imp_exp_dataset(self.exp_conf, dataset_file, self.device)
        dataloder = SingleDataLoader(dataset=dataset, batch_size=batch_size,
                                     sampler=SequentialSampler(dataset), collate_fn=dataset.collate)

        preds, labels, all_weights, lengthss = [], [], [], []
        for batch_data, _, lbstrss in dataloder:
            self.model.eval()
            with torch.no_grad():
                batch_preds, weights = self.model.predict(batch_data, output_weight=True)
                all_weights.extend(weights.cpu().numpy())

            batch_pred_strs = self.tgidss2tgstrss(
                batch_preds.data.cpu().numpy() if not isinstance(batch_preds, list) else batch_preds,
                batch_data['lengths'].cpu().numpy())

            if self.fun_rule is not None:
                batch_pred_strs = self.fun_rule(batch_pred_strs)

            preds.extend(batch_pred_strs)
            labels.extend(lbstrss)
            lengthss.extend(batch_data['lengths'].cpu().numpy())

        for it in range(len(dataset.org_data)):
            saver = {}
            # token
            for seq in range(lengthss[it]):
                seq_name = '%d' % seq
                saver[seq_name] = []
                saver[seq_name].append(dataset.org_data[it]['tokens'][seq])
                saver[seq_name].append(labels[it][seq])
                saver[seq_name].append(all_weights[it][seq][0])
                saver[seq_name].append(all_weights[it][seq][1])
                saver[seq_name].append(all_weights[it][seq][2])
                saver[seq_name].append(all_weights[it][seq][3])
            df = pd.DataFrame(saver)
            df.to_excel(output_file)
            break

        return preds, labels, all_weights

    def save_mean_weight_by_cofe(self, output_dir, batch_size=32):
        dataloder = SingleDataLoader(dataset=self.data_tst, batch_size=batch_size,
                                     sampler=SequentialSampler(self.data_tst), collate_fn=self.data_tst.collate)

        weight_pre_y = [[[] for j in range(8)] for i in range(8)]
        weight_pre_r = [[[] for j in range(8)] for i in range(8)]
        weight_cur_r = [[[] for j in range(8)] for i in range(8)]
        weight_nxt_r = [[[] for j in range(8)] for i in range(8)]

        for batch_data, _, lbstrss in dataloder:
            self.model.eval()
            with torch.no_grad():
                batch_preds, weights = self.model.predict(batch_data, output_weight=True)

            for tags, weight, length in zip(batch_preds.data.cpu().numpy(), weights.cpu().numpy(),
                                            batch_data['lengths'].cpu().numpy()):
                for seq in range(length):
                    cur_tag = tags[seq] + 1
                    pre_tag = 0 if seq == 0 else tags[seq - 1] + 1

                    weight_pre_y[pre_tag][cur_tag].append(weight[seq][0])
                    weight_pre_r[pre_tag][cur_tag].append(weight[seq][1])
                    weight_cur_r[pre_tag][cur_tag].append(weight[seq][2])
                    weight_nxt_r[pre_tag][cur_tag].append(weight[seq][3])

        for i in range(8):
            for j in range(8):
                weight_pre_y[i][j] = np.mean(weight_pre_y[i][j]) if weight_pre_y[i][j] else -1
                weight_pre_r[i][j] = np.mean(weight_pre_r[i][j]) if weight_pre_r[i][j] else -1
                weight_cur_r[i][j] = np.mean(weight_cur_r[i][j]) if weight_cur_r[i][j] else -1
                weight_nxt_r[i][j] = np.mean(weight_nxt_r[i][j]) if weight_nxt_r[i][j] else -1
        weight_pre_y = pd.DataFrame(np.array(weight_pre_y))
        weight_pre_r = pd.DataFrame(np.array(weight_pre_r))
        weight_cur_r = pd.DataFrame(np.array(weight_cur_r))
        weight_nxt_r = pd.DataFrame(np.array(weight_nxt_r))

        weight_pre_y.to_excel(concat_path(output_dir, 'pre_y.xlsx'))
        weight_pre_r.to_excel(concat_path(output_dir, 'pre_r.xlsx'))
        weight_cur_r.to_excel(concat_path(output_dir, 'cur_r.xlsx'))
        weight_nxt_r.to_excel(concat_path(output_dir, 'nxt_r.xlsx'))

        print('weight_pre_y')
        print(weight_pre_y)
        print('weight_pre_r')
        print(weight_pre_r)
        print('weight_cur_r')
        print(weight_cur_r)
        print('weight_nxt_r')
        print(weight_nxt_r)

    def save_con_prob_by_cofe(self, output_dir, batch_size=32):
        dataloder = SingleDataLoader(dataset=self.data_tst, batch_size=batch_size,
                                     sampler=SequentialSampler(self.data_tst), collate_fn=self.data_tst.collate)

        prob_true = np.zeros((8, 8), dtype=np.int32)
        prob_pred = np.zeros((8, 8), dtype=np.int32)

        for batch_data, labels, _ in dataloder:
            self.model.eval()
            with torch.no_grad():
                batch_preds, weights = self.model.predict(batch_data, output_weight=True)

            for t_pred, t_true, length in zip(batch_preds.data.cpu().numpy(), labels.cpu().numpy(),
                                              batch_data['lengths'].cpu().numpy()):
                for seq in range(length):
                    pre_tag = 0 if seq == 0 else t_pred[seq - 1] + 1
                    cur_tag = t_pred[seq] + 1
                    prob_pred[pre_tag][cur_tag] += 1

                    pre_tag = 0 if seq == 0 else t_true[seq - 1] + 1
                    cur_tag = t_true[seq] + 1
                    prob_true[pre_tag][cur_tag] += 1

        prob_true = prob_true.astype(np.float32)
        for i in range(prob_true.shape[0]):
            prob_true[i] = prob_true[i] / prob_true[i].sum()

        prob_pred = prob_pred.astype(np.float32)
        for i in range(prob_pred.shape[0]):
            prob_pred[i] = prob_pred[i] / prob_pred[i].sum()

        prob_diff = prob_true - prob_pred
        prob_diff_abs = np.abs(prob_diff)

        prob_true = pd.DataFrame(prob_true)
        prob_pred = pd.DataFrame(prob_pred)
        prob_diff = pd.DataFrame(prob_diff)
        prob_diff_abs = pd.DataFrame(prob_diff_abs)

        prob_true.to_excel(concat_path(output_dir, 'true.xlsx'))
        prob_pred.to_excel(concat_path(output_dir, 'pred.xlsx'))
        prob_diff.to_excel(concat_path(output_dir, 'diff.xlsx'))
        prob_diff_abs.to_excel(concat_path(output_dir, 'abs.xlsx'))

        print('prob_true')
        print(prob_true)
        print('prob_pred')
        print(prob_pred)
        print('prob_diff')
        print(prob_diff)
        print('prob_diff_abs')
        print(prob_diff_abs)

    def save_Zs_by_cofe(self, output_dir, batch_size=32):
        dataloder = SingleDataLoader(dataset=self.data_tst, batch_size=batch_size,
                                     sampler=SequentialSampler(self.data_tst), collate_fn=self.data_tst.collate)

        saver1, saver2 = {}, {}
        for n in ['py', 'pr', 'cr', 'nr']:
            for p in range(8):
                saver1['%s-%d' % (n, p)] = []

        for p in range(8):
            for n in ['py', 'pr', 'cr', 'nr']:
                saver2['%s-%d' % (n, p)] = []

        for batch_data, labels, _ in dataloder:
            self.model.eval()
            with torch.no_grad():
                batch_preds, Zpes, Zhps, Zhcs, Zhns = self.model.predict(batch_data, output_Z=True)

            for tags, Zpe, Zhp, Zhc, Zhn, length in zip(labels,
                                                        Zpes.cpu().numpy(), Zhps.cpu().numpy(), Zhcs.cpu().numpy(),
                                                        Zhns.cpu().numpy(),
                                                        batch_data['lengths'].cpu().numpy()):
                for seq in range(length):
                    p = 0 if seq == 0 else tags[seq - 1] + 1
                    saver1['%s-%d' % ('py', p)].append(Zpe[seq])
                    saver1['%s-%d' % ('pr', p)].append(Zhp[seq])
                    saver1['%s-%d' % ('cr', p)].append(Zhc[seq])
                    saver1['%s-%d' % ('nr', p)].append(Zhn[seq])

                    saver2['%s-%d' % ('py', p)].append(Zpe[seq])
                    saver2['%s-%d' % ('pr', p)].append(Zhp[seq])
                    saver2['%s-%d' % ('cr', p)].append(Zhc[seq])
                    saver2['%s-%d' % ('nr', p)].append(Zhn[seq])

        for key, val in saver1.items():
            saver1[key] = np.mean(np.array(val), axis=0)
        for key, val in saver2.items():
            saver2[key] = np.mean(np.array(val), axis=0)

        df1 = pd.DataFrame(saver1)
        df2 = pd.DataFrame(saver2)

        df1.to_excel(concat_path(output_dir, 'all1.xlsx'))
        df2.to_excel(concat_path(output_dir, 'all2.xlsx'))

        print('df1')
        print(df1)
        print('df2')
        print(df2)


if __name__ == '__main__':
    # Executor('en_bert_crf', None).load_model().log_worst_items_by_tst(max_log_items=3000)
    # Executor('en2f_bert_d3_0.5', None).load_model().log_worst_items_by_tst(max_log_items=3000)

    # print result
    # mod = Executor('en2f_emb_grdb', None).load_model()
    # ret = Executor.format_result(mod.eval_dataset_tst(), markdown_table=True)
    # print(ret)

    # mod = Executor('en2f_emb_grdb', None).load_model()
    # ret = Executor.format_result(mod.eval_dataset_tst(beam_width=1), markdown_table=True)
    # print(ret)

    exp_name = 'en2f_bert'

    mod = Executor(exp_name, None).load_model()
    ret = Executor.format_result(mod.eval_dataset_tst(), markdown_table=True)
    print(ret)

    # mod = Executor('en2f_bert_grdb', None).load_model()
    # ret = Executor.format_result(mod.eval_dataset_tst(beam_width=7), markdown_table=True)
    # print(ret)

    # log worst items
    # mod.log_worst_items_by_tst(max_log_items=3000)

    # WEIGHT en2f 1 case
    # mod = Executor('en2f_bert_grdb', None).load_model()
    # preds, labels, all_weights = mod.save_single_case_weight_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/res/polnear-v2-fixed/oth.txt',
    #     '/Users/lixiang/PycharmProjects/discourse/log/en2f_single_weight.xlsx'
    # )

    # mean WEIGHT zh
    # mod = Executor('zh_bert_grdb', None).load_model()
    # mod.save_mean_weight_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/log/zh_weight'
    # )

    # mean WEIGHT en2f
    # mod = Executor('en2f_bert_grdb', None).load_model()
    # mod.save_mean_weight_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/log/en2f_weight'
    # )

    # mean WEIGHT zh
    # mod = Executor('zh_bert_grdb', None).load_model()
    # mod.save_con_prob_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/log/zh_prob'
    # )

    # mean WEIGHT en2f
    # mod = Executor('en2f_bert_grdb', None).load_model()
    # mod.save_con_prob_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/log/en2f_prob'
    # )

    # mean Zs en2f
    # mod = Executor('en2f_bert_grdb', None).load_model()
    # mod.save_Zs_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/log/en2f_Zs'
    # )

    # mean Zs zh
    # mod = Executor('zh_bert_grdb', None).load_model()
    # mod.save_Zs_by_cofe(
    #     '/Users/lixiang/PycharmProjects/discourse/log/zh_Zs'
    # )
