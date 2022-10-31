import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_gen(input_lengths):
    """ Forward pass.
    # Arguments:
        input_lengths (torch.LongTensor): Lengths of the effective each row
    # Return:
        mask: mask results
    """
    max_len = torch.max(input_lengths)
    indices = torch.arange(0, max_len, device=input_lengths.device).unsqueeze(0)
    # mask = Variable((indices < input_lengths.unsqueeze(1)).float())
    mask = (indices < input_lengths.unsqueeze(1)).float()

    return mask


def dynamic_softmax(input, input_lengths):
    """ Forward pass.
    # Arguments:
        inputs (Torch.Variable): Tensor of input matrix
        input_lengths (torch.LongTensor): Lengths of the effective each row
    # Return:
        attentions: dynamic softmax results
    """
    mask = mask_gen(input_lengths)

    # apply mask and renormalize attention scores (weights)
    masked_weights = input * mask
    att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
    dyn_softmax = masked_weights.div(att_sums + 1e-12)

    return dyn_softmax


def softmax_mask(input, mask, dim=-1):
    e = torch.exp(input) * mask
    ss = torch.sum(e, dim=dim).unsqueeze(1)
    ss[ss == 0] = 1
    # ss = ss + 1e-9
    return e / ss


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def sequence_mask_att(lengths, max_len):
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0)
            .expand(lengths.numel(), max_len)
            .lt(lengths.unsqueeze(1))
            .type(torch.float32))


def pad_sequence_with_max_len(sequences, batch_first=False, padding_value=0, max_len=-1):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences]) if max_len < 0 else max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def get_word_rep_from_subword(vectors, lengths):
    embedded_ = []
    idx_begin, idx_end = 0, 0
    for len_current in lengths:
        cur_len = len_current.item()
        if cur_len <= 0:
            break
        idx_begin, idx_end = idx_end, idx_end + cur_len
        embedded_tmp = vectors[idx_begin: idx_end]
        # fix the bug that some word are ignored by bert, then the len_current is 0, then cause nan error
        # Please notice that had better use those when the input of Bert is array.
        if len(embedded_tmp) < 1:
            embedded_tmp = vectors[idx_begin: idx_begin + 1]
        embedded_.append(torch.mean(embedded_tmp, dim=0))
    return torch.stack(embedded_)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # def forward(self, euclidean_distance, label):
    #     loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
    #                                   (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
    #                                                           2))
    #     return loss_contrastive

    def forward(self, anchor, positive, negative):
        euclidean_distance_1 = F.pairwise_distance(anchor, positive)
        euclidean_distance_0 = F.pairwise_distance(anchor, negative)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance_1, 2) +
                                      torch.pow(torch.clamp(self.margin - euclidean_distance_0, min=0.0), 2))
        return loss_contrastive


class MyCosineLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(MyCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        loss_entropy_1 = 1.0 - F.cosine_similarity(anchor, positive, dim=-1).mean()
        loss_entropy_2 = max(0, F.cosine_similarity(anchor, negative, dim=-1).mean() - self.margin)
        return loss_entropy_1 + loss_entropy_2


class Gelu(nn.Module):

    def __init__(self):
        super(Gelu, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
