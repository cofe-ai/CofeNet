import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.torch_utils.rnn import DynamicLSTM
from models.torch_utils.funs import Gelu, sequence_mask


class EnhancedCell(nn.Module):
    def __init__(self, config):
        super(EnhancedCell, self).__init__()
        self.tag_size = config['tag_size']
        self.random_seed = config['random_seed']
        self.true_pred_rand_prob = config['true_pred_rand_prob']
        self.true_pred_rand_prob[1] += self.true_pred_rand_prob[0]
        self.true_pred_rand_prob[2] += self.true_pred_rand_prob[1]
        self.act_fn = config.get('hidden_act', 'relu')
        assert self.act_fn in ['relu', 'gelu']
        self.num_pre_preds = config['num_pre_preds']
        assert self.num_pre_preds > 0
        self.num_pre_tokens = config['num_pre_tokens']
        assert self.num_pre_tokens > 0
        self.num_nxt_tokens = config['num_nxt_tokens']
        assert self.num_nxt_tokens > 0
        self.in_hidden_size = config['in_hidden_size']
        self.hidden_size = config['hidden_size']
        self.pred_embedding_dim = config['pred_embedding_dim']
        self.decay_ctrl = config.get('decay_ctrl', True)
        self.attention_ctrl = config.get('attention_ctrl', True)
        self.y_enable = config.get('y_enable', True)
        self.p_enable = config.get('p_enable', True)
        self.c_enable = config.get('c_enable', True)
        self.n_enable = config.get('n_enable', True)

        self.fc_pe_dropout_prob = config['fc_pe_dropout_prob']
        self.fc_hc_dropout_prob = config['fc_hc_dropout_prob']
        self.fc_hp_dropout_prob = config['fc_hp_dropout_prob']
        self.fc_hn_dropout_prob = config['fc_hn_dropout_prob']

        self.rand = random.Random(self.random_seed)

        self.layer_pred_emb = nn.Embedding(self.tag_size + self.num_pre_preds, self.pred_embedding_dim)
        self.feats_pad_bng = nn.Parameter(torch.empty(1, 1, self.in_hidden_size))
        self.feats_pad_end = nn.Parameter(torch.empty(1, 1, self.in_hidden_size))

        self.fc_pe = nn.Sequential(
            nn.Linear(self.pred_embedding_dim * self.num_pre_preds, self.hidden_size),
            nn.ReLU(True) if self.act_fn == 'relu' else Gelu(),
            nn.Dropout(self.fc_pe_dropout_prob)
        )

        self.fc_hc = nn.Sequential(
            nn.Linear(self.in_hidden_size, self.hidden_size),
            nn.ReLU(True) if self.act_fn == 'relu' else Gelu(),
            nn.Dropout(self.fc_hc_dropout_prob)
        )

        self.fc_hp = nn.Sequential(
            nn.Linear(self.in_hidden_size * self.num_pre_tokens, self.hidden_size),
            nn.ReLU(True) if self.act_fn == 'relu' else Gelu(),
            nn.Dropout(self.fc_hp_dropout_prob)
        )

        self.fc_hn = nn.Sequential(
            nn.Linear(self.in_hidden_size * self.num_nxt_tokens, self.hidden_size),
            nn.ReLU(True) if self.act_fn == 'relu' else Gelu(),
            nn.Dropout(self.fc_hn_dropout_prob)
        )

        self.Z_pe = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())
        self.Z_hc = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())
        self.Z_hp = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())
        self.Z_hn = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())

        self.Z_att = nn.Sequential(nn.Linear(self.hidden_size * 2, 4), nn.Softmax())

        self.layer_out = nn.Linear(self.hidden_size, self.tag_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.feats_pad_bng, -0.1, 0.1)
        nn.init.uniform_(self.feats_pad_end, -0.1, 0.1)

    def padding_feats(self, feats, lengths):
        # [batch_size, seq_length, hidden_size]
        batch_size, seq_length, _ = feats.shape
        pad_feats = None
        for bat in range(batch_size):
            b_leng = lengths[bat]
            b_feat = feats[bat: bat + 1, :b_leng, :]
            b_feat = torch.cat(
                (
                    self.feats_pad_bng.repeat((1, self.num_pre_tokens, 1)),
                    b_feat,
                    self.feats_pad_end.repeat((1, seq_length - b_leng + self.num_nxt_tokens, 1))
                ), dim=1
            )
            pad_feats = b_feat if pad_feats is None else torch.cat((pad_feats, b_feat), dim=0)
        return pad_feats

    def tag_id_mod(self, embs):
        for b in range(embs.size(0)):
            for s in range(embs.size(1)):
                if embs[b][s] < 0:
                    embs[b][s] += self.tag_size + self.num_pre_preds
        return embs

    def forward(self, feats, lengths, labelss, ignore_index=-1):
        device = feats.device
        batch_size, seq_length, _ = feats.shape

        bng_tags = torch.range(start=-self.num_pre_preds, end=-1, step=1, dtype=torch.long, device=device).unsqueeze(0).repeat(
            (batch_size, 1))

        # [batch_size, seq_length, pred_embedding_dim]
        true_embs = self.tag_id_mod(torch.cat((bng_tags, labelss), dim=1))
        true_embs = self.layer_pred_emb(true_embs)

        # [batch_size, seq_length, pred_embedding_dim]
        pred_embs = self.tag_id_mod(bng_tags)
        pred_embs = self.layer_pred_emb(pred_embs)

        # [batch_size, num_pre_tokens + seq_length + num_nxt_tokens, in_hidden_size]
        feats = self.padding_feats(feats, lengths)

        probs = None
        for seq in range(seq_length):
            feat_seq = seq + self.num_pre_tokens
            pred_seq = seq + self.num_pre_preds

            # pe
            rand = self.rand.random()
            if rand < self.true_pred_rand_prob[0]:  # true
                pred_e = true_embs[:, pred_seq - self.num_pre_preds: pred_seq, :].reshape((batch_size, -1))
            elif rand < self.true_pred_rand_prob[1]:  # pred
                pred_e = pred_embs[:, pred_seq - self.num_pre_preds: pred_seq, :].reshape((batch_size, -1))
            else:  # rand
                rand_tags = torch.randint(high=self.tag_size, size=(batch_size, self.num_pre_preds), device=device)
                pred_e = self.layer_pred_emb(rand_tags).reshape((batch_size, -1))
            pe = self.fc_pe(pred_e)

            # hp
            hp = self.fc_hp(feats[:, feat_seq - self.num_pre_tokens: feat_seq, :].reshape((batch_size, -1)))

            # hc
            hc = self.fc_hc(feats[:, feat_seq, :])

            # hn
            hn = self.fc_hn(feats[:, feat_seq + 1: feat_seq + 1 + self.num_nxt_tokens, :].reshape((batch_size, -1)))

            if not self.y_enable:
                pe = torch.zeros_like(pe, dtype=pe.dtype, device=device)
            if not self.p_enable:
                hp = torch.zeros_like(hp, dtype=hp.dtype, device=device)
            if not self.c_enable:
                hc = torch.zeros_like(hc, dtype=hc.dtype, device=device)
            if not self.n_enable:
                hn = torch.zeros_like(hn, dtype=hn.dtype, device=device)

            cat_pe_hc = torch.cat((pe, hc), dim=-1)

            # att
            att = self.Z_att(cat_pe_hc)
            if not self.attention_ctrl:
                att = torch.ones_like(att, dtype=att.dtype, device=device)

            # Z
            if self.decay_ctrl:
                hidden = pe * self.Z_pe(cat_pe_hc) * att[:, 0: 1] + hp * self.Z_hp(cat_pe_hc) * att[:, 1: 2] + \
                         hc * self.Z_hc(cat_pe_hc) * att[:, 2: 3] + hn * self.Z_hn(cat_pe_hc) * att[:, 3: 4]
            else:
                hidden = pe * att[:, 0: 1] + hp * att[:, 1: 2] + hc * att[:, 2: 3] + hn * att[:, 3: 4]
            cur_prob = self.layer_out(hidden)
            probs = cur_prob.unsqueeze(1) if probs is None else torch.cat((probs, cur_prob.unsqueeze(1)), dim=1)

            cur_pred = torch.argmax(cur_prob, dim=-1)
            pred_embs = torch.cat((pred_embs, self.layer_pred_emb(cur_pred).unsqueeze(1)), dim=1)

        loss = F.nll_loss(torch.log(torch.softmax(probs, dim=-1).clamp(min=1e-9).transpose(1, 2)), labelss, ignore_index=ignore_index)
        return loss

    def predict(self, feats, lengths, output_weight=False, output_Z=False):
        device = feats.device
        batch_size, seq_length, _ = feats.shape

        # [batch_size, seq_length, pred_embedding_dim]
        pred_embs = torch.range(start=-self.num_pre_preds, end=-1, step=1, dtype=torch.long, device=device).unsqueeze(0).repeat(
            (batch_size, 1))
        pred_embs = self.layer_pred_emb(self.tag_id_mod(pred_embs))

        # [batch_size, num_pre_tokens + seq_length + num_nxt_tokens, in_hidden_size]
        feats = self.padding_feats(feats, lengths)

        probs = None
        if output_weight:
            weigths = None
        if self.decay_ctrl and output_Z:
            Zpes, Zhps, Zhcs, Zhns = None, None, None, None

        for seq in range(seq_length):
            feat_seq = seq + self.num_pre_tokens
            pred_seq = seq + self.num_pre_preds

            # pe
            pe = self.fc_pe(pred_embs[:, pred_seq - self.num_pre_preds: pred_seq, :].reshape((batch_size, -1)))

            # hp
            hp = self.fc_hp(feats[:, feat_seq - self.num_pre_tokens: feat_seq, :].reshape((batch_size, -1)))

            # hc
            hc = self.fc_hc(feats[:, feat_seq, :])

            # hn
            hn = self.fc_hn(feats[:, feat_seq + 1: feat_seq + 1 + self.num_nxt_tokens, :].reshape((batch_size, -1)))

            if not self.y_enable:
                pe = torch.zeros_like(pe, dtype=pe.dtype, device=device)
            if not self.p_enable:
                hp = torch.zeros_like(hp, dtype=hp.dtype, device=device)
            if not self.c_enable:
                hc = torch.zeros_like(hc, dtype=hc.dtype, device=device)
            if not self.n_enable:
                hn = torch.zeros_like(hn, dtype=hn.dtype, device=device)

            cat_pe_hc = torch.cat((pe, hc), dim=-1)

            # att
            att = self.Z_att(cat_pe_hc)
            if not self.attention_ctrl:
                att = torch.ones_like(att, dtype=att.dtype, device=device)

            if output_weight:
                weigths = att.unsqueeze(1) if weigths is None else torch.cat((weigths, att.unsqueeze(1)), dim=1)

            # Z
            if self.decay_ctrl:
                Zpe = self.Z_pe(cat_pe_hc)
                Zhp = self.Z_hp(cat_pe_hc)
                Zhc = self.Z_hc(cat_pe_hc)
                Zhn = self.Z_hn(cat_pe_hc)
                hidden = pe * Zpe * att[:, 0: 1] + hp * Zhp * att[:, 1: 2] + hc * Zhc * att[:, 2: 3] + hn * Zhn * att[:, 3: 4]
                if output_Z:
                    Zpes = Zpe.unsqueeze(1) if Zpes is None else torch.cat((Zpes, Zpe.unsqueeze(1)), dim=1)
                    Zhps = Zhp.unsqueeze(1) if Zhps is None else torch.cat((Zhps, Zhp.unsqueeze(1)), dim=1)
                    Zhcs = Zhc.unsqueeze(1) if Zhcs is None else torch.cat((Zhcs, Zhc.unsqueeze(1)), dim=1)
                    Zhns = Zhn.unsqueeze(1) if Zhns is None else torch.cat((Zhns, Zhn.unsqueeze(1)), dim=1)
            else:
                hidden = pe * att[:, 0: 1] + hp * att[:, 1: 2] + hc * att[:, 2: 3] + hn * att[:, 3: 4]
            cur_prob = self.layer_out(hidden)
            probs = cur_prob.unsqueeze(1) if probs is None else torch.cat((probs, cur_prob.unsqueeze(1)), dim=1)

            cur_pred = torch.argmax(cur_prob, dim=-1)
            pred_embs = torch.cat((pred_embs, self.layer_pred_emb(cur_pred).unsqueeze(1)), dim=1)

        outputs = probs

        if output_weight:
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
            outputs += (weigths, )

        if self.decay_ctrl and output_Z:
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
            outputs += (Zpes, Zhps, Zhcs, Zhns, )

        return outputs

    def predict_bs(self, feats, lengths, beam_width):
        device = feats.device
        batch_size, seq_length, _ = feats.shape

        # [batch_size, num_pre_tokens + seq_length + num_nxt_tokens, in_hidden_size]
        feats = self.padding_feats(feats, lengths)

        preds = []

        for bat in range(batch_size):
            # [beam_width, seq_length, pred_embedding_dim]
            bat_pred_embs_bs = torch.range(start=-self.num_pre_preds, end=-1, step=1, dtype=torch.long, device=device).unsqueeze(0).repeat(
                (beam_width, 1))
            bat_pred_embs_bs = self.layer_pred_emb(self.tag_id_mod(bat_pred_embs_bs))

            bat_preds_bs = [[] for _ in range(beam_width)]
            bat_score_bs = [0.0 for _ in range(beam_width)]

            for seq in range(lengths[bat]):
                seq_f = seq + self.num_pre_tokens
                seq_p = seq + self.num_pre_preds

                ft = feats[bat: bat + 1, :, :]
                # hp
                hp = self.fc_hp(ft[:, seq_f - self.num_pre_tokens: seq_f, :].reshape((1, -1))).repeat((beam_width, 1))
                # hc
                hc = self.fc_hc(ft[:, seq_f, :]).repeat((beam_width, 1))
                # hn
                hn = self.fc_hn(ft[:, seq_f + 1: seq_f + 1 + self.num_nxt_tokens, :].reshape((1, -1))).repeat((beam_width, 1))

                # pe
                pe = self.fc_pe(bat_pred_embs_bs[:, seq_p - self.num_pre_preds: seq_p, :].reshape((beam_width, -1)))
                cat_pe_hc = torch.cat((pe, hc), dim=-1)

                # att
                att = self.Z_att(cat_pe_hc)

                # Z
                hidden = pe * self.Z_pe(cat_pe_hc) * att[:, 0: 1] + hp * self.Z_hp(cat_pe_hc) * att[:, 1: 2] + \
                         hc * self.Z_hc(cat_pe_hc) * att[:, 2: 3] + hn * self.Z_hn(cat_pe_hc) * att[:, 3: 4]
                bat_cur_prob_bs = torch.softmax(self.layer_out(hidden), dim=-1)
                if seq == 0:
                    bat_cur_prob_bs = bat_cur_prob_bs[0: 1, ...]
                bat_cur_prob_bs = bat_cur_prob_bs.reshape((-1))

                topk_probs, topk_idxs = bat_cur_prob_bs.topk(beam_width, 0)

                new_bat_pred_embs_bs = torch.zeros((beam_width, bat_pred_embs_bs.shape[1] + 1, self.pred_embedding_dim), device=device)
                new_bat_preds_bs = []
                new_bat_score_bs = []
                for save_bs_idx, (tp_prob, tp_idx) in enumerate(zip(topk_probs, topk_idxs)):
                    last_bs_idx = tp_idx / self.tag_size
                    this_pred = tp_idx % self.tag_size
                    new_bat_pred_embs_bs[save_bs_idx, : -1, :] = bat_pred_embs_bs[last_bs_idx]
                    new_bat_pred_embs_bs[save_bs_idx, -1, :] = self.layer_pred_emb(this_pred)

                    new_bat_preds_bs.append(copy.deepcopy(bat_preds_bs[last_bs_idx]))
                    new_bat_preds_bs[save_bs_idx].append(this_pred.item())

                    new_bat_score_bs.append(copy.deepcopy(bat_score_bs[last_bs_idx]))
                    new_bat_score_bs[save_bs_idx] += tp_prob.item()

                bat_pred_embs_bs = new_bat_pred_embs_bs
                bat_preds_bs = new_bat_preds_bs
                bat_score_bs = new_bat_score_bs
            # seq end
            best_bs_idx = np.argmax(bat_score_bs)
            preds.append(bat_preds_bs[best_bs_idx])
        # bat end

        return preds
