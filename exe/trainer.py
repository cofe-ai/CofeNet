import time
from torch import optim
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .executor import Executor
from data import imp_exp_dataset
from utils import *


def get_current_time_str():
    t = time.gmtime()
    return '%04d%02d%02d_%02d%02d%02d' % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


class Trainer(Executor):
    def __init__(self, params: dict):
        self.params = params
        self.random_seed = params['random_seed']
        set_global_rand_seed(self.random_seed)

        super().__init__(params['exp_name'], params['gpu'])

        # freq param
        self.show_per_step = self.params['show_per_step']
        self.eval_per_step = self.params['eval_per_step']
        self.min_eval_step = self.params['min_eval_step']
        self.eval_type = self.params['eval_type']
        self.max_mod_saved_num = self.params['max_mod_saved_num']
        self.do_not_save_mod = self.params['do_not_save_mod']

        self.max_epoch = self.params['max_epoch']
        self.batch_size = self.params['batch_size']

        # load data
        self.data_trn = imp_exp_dataset(self.exp_conf, 'TRN', self.device)
        self.data_val = imp_exp_dataset(self.exp_conf, 'VAL', self.device)

        # load pretrained
        if self.exp_conf.bert_pretrained is not None:
            layers = self.model.load_pretrained(self.exp_conf.bert_pretrained)
            if isinstance(layers, tuple):
                for l in layers:
                    l.to(self.device)
            else:
                layers.to(self.device)

        # fix bert
        if params['fix_bert']:
            self.model.fix_bert()

        # optim
        self.cls_optim = getattr(optim, self.params['optim'])
        bert_params, base_params = self.model.get_params_by_part()
        self.optimizer = self.cls_optim([
            {'params': bert_params, 'weight_decay': self.params['bert_weight_decay'], 'lr': self.params['bert_learning_rate']},
            {'params': base_params, 'weight_decay': self.params['weight_decay'], 'lr': self.params['learning_rate']}
        ],
            lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])

        # log
        self.record_name = '%s_%s_%s' % (self.params['exp_name'], self.params['trn_name'], get_current_time_str())
        self.log_dir = define_dir(self.log_root_dir, self.record_name)
        self.log_config_file = concat_path(self.log_root_dir, self.record_name + '.json')
        self.log_record_file = concat_path(self.log_root_dir, self.record_name + '.txt')
        save_json_file(self.log_config_file, self.params, 2)
        self.writer = SummaryWriter(self.log_dir, flush_secs=6)

        # other
        self.mod_saved_path_cache = []

    def eval_dataset_val(self, batch_size=32):
        return self.eval_dataset(self.data_val, batch_size)

    def save_model(self, mod_index):
        file_path = self._get_params_file_path(mod_index)
        if self.max_mod_saved_num is not None:
            while len(self.mod_saved_path_cache) >= self.max_mod_saved_num:
                if os.path.isfile(self.mod_saved_path_cache[0]):
                    os.remove(self.mod_saved_path_cache[0])
                self.mod_saved_path_cache = self.mod_saved_path_cache[1:]
            self.mod_saved_path_cache.append(file_path)
        print('Save %d -> \'%s\' ..' % (mod_index, file_path))
        torch.save(self.model.state_dict(), file_path)
        return self

    def train_step(self, batch_data, batch_labels):
        self.model.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        with torch.enable_grad():
            batch_loss = self.model.forward_loss(batch_data, batch_labels, self.data_trn.LBID_IGN)

        if isinstance(batch_loss, tuple):
            for b_loss in batch_loss:
                b_loss.backward()
            self.optimizer.step()
            return sum(batch_loss).data.cpu().numpy()
        else:
            batch_loss.backward()
            self.optimizer.step()
            return batch_loss.data.cpu().numpy()

    def train(self):
        epoch, step = 0, 0
        all_step_in_epoch = len(self.data_trn) // self.batch_size + (1 if len(self.data_trn) % self.batch_size != 0 else 0)

        best_step, best_val_score, best_tst_score = -1, 0.0, 0.0

        while self.max_epoch < 0 or epoch < self.max_epoch:
            dataloder = DataLoader(dataset=self.data_trn, batch_size=self.batch_size,
                                   sampler=RandomSampler(self.data_trn), collate_fn=self.data_trn.collate)

            for step_in_epoch, (batch_data, batch_labels, _) in enumerate(dataloder):
                batch_loss = self.train_step(batch_data, batch_labels)

                # show loss
                if step % self.show_per_step == 0:
                    print('step:%d epoch:%d [%d / %d] loss: %.4f best@%d[val: %.4f tst: %.4f]' %
                          (step, epoch, step_in_epoch, all_step_in_epoch, batch_loss, best_step, best_val_score, best_tst_score))
                    self.writer.add_scalar('trn/loss', batch_loss, step)

                # eval val
                if step % self.eval_per_step == 0 and step > self.min_eval_step:
                    val_eval_result = self.eval_dataset_val(self.batch_size)
                    if self.eval_type == 'bio_f1':
                        val_score = val_eval_result['bio']['f1']
                    elif self.eval_type == 'exact_f1_avg':
                        val_score = sum([val_eval_result['exact_match'][x]['f1'] for x in ['source', 'cue', 'content']])

                    # write val eval log
                    self.write_summary('val', val_eval_result, step)
                    print('EVAL step:%d epoch:%d [%d / %d] val=[score: %.4f -> %.4f]' %
                          (step, epoch, step_in_epoch, all_step_in_epoch, best_val_score, val_score), end='')

                    if val_score > best_val_score:
                        # eval tst
                        tst_eval_result = self.eval_dataset_tst(self.batch_size)
                        if self.eval_type == 'bio_f1':
                            tst_score = tst_eval_result['bio']['f1']
                        elif self.eval_type == 'exact_f1_avg':
                            tst_score = sum([tst_eval_result['exact_match'][x]['f1'] for x in ['source', 'cue', 'content']])

                        # write tst eval log
                        self.write_summary('tst', tst_eval_result, step)
                        print('+ tst=[score: %.4f -> %.4f]%s' %
                              (best_tst_score, tst_score, '+' if tst_score > best_tst_score else '-'))

                        # update best
                        best_step, best_val_score, best_tst_score = step, val_score, tst_score

                        # record result
                        self.record_result(tst_eval_result, val_eval_result, step, epoch)

                        # save model
                        if not self.do_not_save_mod:
                            self.save_model(step)
                    else:
                        print('-')

                # next step
                step += 1

            # next epoch
            epoch += 1

    def write_summary(self, data_type, eval_result, step):
        """
        :param eval_result: {
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
        self.writer.add_scalar('%s/bio-f1' % data_type, eval_result['bio']['f1'], step)
        # self.writer.add_scalar('%s/begin_multiple-f1' % data_type, eval_result['begin_multiple']['f1'], step)

        self.writer.add_scalar('%s/begin-src-f1' % data_type, eval_result['begin']['source']['f1'], step)
        self.writer.add_scalar('%s/begin-cue-f1' % data_type, eval_result['begin']['cue']['f1'], step)
        self.writer.add_scalar('%s/begin-cnt-f1' % data_type, eval_result['begin']['content']['f1'], step)

        self.writer.add_scalar('%s/exact_match-src-f1' % data_type, eval_result['exact_match']['source']['f1'], step)
        self.writer.add_scalar('%s/exact_match-cue-f1' % data_type, eval_result['exact_match']['cue']['f1'], step)
        self.writer.add_scalar('%s/exact_match-cnt-f1' % data_type, eval_result['exact_match']['content']['f1'], step)

        self.writer.add_scalar('%s/jaccard-src' % data_type, eval_result['jaccard']['source'], step)
        self.writer.add_scalar('%s/jaccard-cue' % data_type, eval_result['jaccard']['cue'], step)
        self.writer.add_scalar('%s/jaccard-cnt' % data_type, eval_result['jaccard']['content'], step)

    def record_result(self, tst_eval_result, val_eval_result, step, epoch):
        with open(self.log_record_file, "a") as f:
            # step title
            f.write('\n' + ('=' * 100) + '\n')
            f.write(('=' * 35) + ('%d in %d' % (step, epoch)).center(30) + ('=' * 35) + '\n')

            # VAL
            f.write(('>' * 20) + 'valid'.center(10) + ('<' * 20) + '\n')
            f.write(self.format_result(val_eval_result))

            f.write('\n')

            # TST
            f.write(('>' * 20) + 'test'.center(10) + ('<' * 20) + '\n')
            f.write(self.format_result(tst_eval_result, True))

            # end
            f.write('\n' + ('=' * 100) + '\n')


if __name__ == '__main__':
    pass
