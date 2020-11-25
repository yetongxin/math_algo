import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import DataLoader
from config import *
args = get_args()
data_loader = DataLoader(args)

# 按逗号分割，相同的子句中设置为同一个token，遇到新逗号时token+1, 问题句子的token设置为1000
def src_to_mask(src):
    # src shape: [batch_size, seq_len]
    src = src.cpu().numpy()
    batch_data_mask_tok = []
    for encode_sen_idx in src:

        token = 1
        mask = [0] * len(encode_sen_idx)
        for num in range(len(encode_sen_idx)):
            mask[num] = token
            if (encode_sen_idx[num] == data_loader.vocab_dict["．"] or encode_sen_idx[num] == data_loader.vocab_dict["，"]) \
                    and num != len(encode_sen_idx) - 1:
                token += 1
            if encode_sen_idx[num]==0:mask[num] = 0
        for num in range(len(encode_sen_idx)):
            if mask[num] == token and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask)
    # [[1, 1, 1, 2, 2, 2, 3, 3,1000, 1000, 0, 0,...], [1, 1, 1, 2, 2, 2, 3, 3,1000, 1000, 0, 0,...],  ... ]
    return np.array(batch_data_mask_tok)


# batch: 已mask为1，2，3, (1000),0,0,0....的batch
def group_mask(batch,type="self",pad=0):
    length = batch.shape[1]
    lis = []
    # 当前子句的设置为1， 其他设置为0
    if type=="self":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            # print('mask:', mask, mask.shape, tok.shape)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                    #print("self copy",copy)
                '''
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)  # 将copy按列叠加到Mask中
            mask = mask[:,1:]  # 第一列是全0
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)  # mask shape: [1, seq_len, seq_len]
            lis.append(mask)
        res = np.concatenate(tuple(lis)) # res shape: [batch_size, seq_len, seq_len]
    # 当前子句和问题句子设置为0，其他子句设置为1
    elif type=="between":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy==1000] = 0
                    copy[copy ==ele] = 0
                    copy[copy!= 0] = 1
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == "question":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele==1000:  # question部分设置为0, 其他部分设置为1
                	copy[copy==0] = -1
                	copy[copy==1] = 0
                	copy[copy==-1] = 1
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)
            mask = mask[:,1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask,0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:return "error"
    return res


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    add & norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))