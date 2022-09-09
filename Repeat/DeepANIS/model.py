#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/03/01 15:16:19
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from einops import rearrange, repeat

######################################## function area ########################################

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def swish_function(x):
    return x * F.sigmoid(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * F.tanh(F.softplus(x))


def mish_function(x):
    return x * F.tanh(F.softplus(x))


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class Position_Wise_Feed_Forward(nn.Module):
    def __init__(self, hidden_features, num_layers, dropout, output_act_fn='mish'):
        super(Position_Wise_Feed_Forward, self).__init__()
        self.num_layers = num_layers
        self.linears = clones(nn.Linear(hidden_features, hidden_features), num_layers)
        self.dropout = nn.Dropout(dropout)
        
        if output_act_fn == 'relu':
            self.output_act = lambda x: F.leaky_relu(x, negative_slope=0.1, inplace=True)
        elif output_act_fn == 'tanh':
            self.output_act = lambda x: F.tanh(x)
        elif output_act_fn == 'swish':
            self.output_act = lambda x: x * F.sigmoid(x)
        elif output_act_fn == 'mish':
            self.output_act = lambda x: x * F.tanh(F.softplus(x))
        else:
            self.output_act = lambda x: x
    
    def forward(self, x):
        if self.num_layers == 0:
            return x
        for i in range(self.num_layers - 1):
            x = self.dropout(mish_function(self.linears[i](x)))
        return self.dropout(self.output_act(self.linears[-1](x)))


def attention(query, key, value, batch_masks, dropout):
    # query, key, value.shape   = [batch, h, max_length, d]
    # batch_masks.shape         = [batch, max_length]
    h, d = query.shape[1], query.shape[-1]
    
    attention_scores = torch.einsum('b h m d, b h d n -> b h m n', query, rearrange(key, 'b h m d -> b h d m'))
    
    # mask padding, shape with = [batch, h, max_length, max_length]
    if batch_masks is not None:
        batch_masks = repeat(batch_masks, 'b m -> b h c m', h=h, c=batch_masks.shape[-1])
        attention_scores = attention_scores.masked_fill(batch_masks==0, -1e12)
    
    attention_scores = F.softmax(attention_scores / math.sqrt(d), dim=-1)
    
    if dropout is not None:
        attention_scores = dropout(attention_scores)
        
    node_hidden = torch.einsum('b h m n, b h n d -> b h m d', attention_scores, value)
    
    return node_hidden


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, hidden_features, num_attention_heads, dropout, output_act_fn):
        super(Multi_Head_Self_Attention, self).__init__()
        if hidden_features % num_attention_heads != 0:
            print("hidden features dimensions not divided by attention heads!")
            assert False
        
        self.num_attention_heads = num_attention_heads
        self.attention_hidden_features = hidden_features // num_attention_heads
        self.layers = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        
        if output_act_fn == 'relu':
            self.output_act = lambda x: F.leaky_relu(x)
        elif output_act_fn == 'tanh':
            self.output_act = lambda x: F.tanh(x)
        elif output_act_fn == 'mish':
            self.output_act = lambda x: x * F.tanh(F.softplus(x))
        else:
            self.output_act = lambda x: x
    
    def forward(self, batch_node_features, batch_masks):
        # batch_node_features.shape = [batch, max_length, hidden_features]
        # batch_masks.shape         = [batch, max_length]
        
        # 1) Do all the linear projections in batch
        query = rearrange(self.layers[0](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        key   = rearrange(self.layers[1](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        value = rearrange(self.layers[2](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        
        # 2) Apply attention on all the projected vectors in batch
        batch_node_features = attention(query, key, value, batch_masks, self.dropout)
        
        # 3) Concatenate
        batch_node_features = rearrange(batch_node_features, 'b h m d -> b m (h d)')
        
        return mish_function(self.layers[3](batch_node_features))
    
    
class Encoder(nn.Module):
    def __init__(self, hidden_features, num_MHSA_layers, num_attention_heads, num_FFN_layers, scale_norm, dropout):
        super(Encoder, self).__init__()
        
        self.num_layers = num_MHSA_layers
        self.MHSAs = clones(Multi_Head_Self_Attention(hidden_features, num_attention_heads, dropout, output_act_fn='mish'), num_MHSA_layers)
        self.FFNs = clones(Position_Wise_Feed_Forward(hidden_features, num_layers=num_FFN_layers, dropout=dropout, output_act_fn='mish'), num_MHSA_layers)
        self.norm = ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_node_features, batch_masks):
        
        for i in range(self.num_layers):
            # MHSAs
            batch_node_features = self.dropout(self.norm(batch_node_features))
            batch_node_features = batch_node_features + self.MHSAs[i](batch_node_features, batch_masks)
            # FFNs
            batch_node_features = self.dropout(self.norm(batch_node_features))
            batch_node_features = batch_node_features + self.FFNs[i](batch_node_features)
        
        return batch_node_features

class DeepANIS(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, num_layers=2, dropout=0.2, bidirectional=True):
        super(DeepANIS, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # input
        self.cdr_emb = nn.Embedding(num_embeddings=22, embedding_dim=hidden_features // 2, padding_idx=0)
        self.node_emb = nn.Linear(input_features, hidden_features // 2)
        # transformer
        self.transformer = Encoder(hidden_features, num_MHSA_layers=2, num_attention_heads=4, num_FFN_layers=1, scale_norm=False, dropout=dropout)
        # lstm
        self.lstm = nn.LSTM(hidden_features, hidden_features, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.lstm_fc = nn.Linear(hidden_features * 2, hidden_features)
        else:
            self.lstm_fc = nn.Linear(hidden_features, hidden_features)
        # output
        self.output_fc = nn.Linear(hidden_features * 2, output_features)
        self.elu = nn.ELU(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False)

        
    def forward(self, cdrs, node_features, feature_masks, label_masks, device):
        # [batch, max_length] -> [batch, max_length, hidden_features // 2]
        cdrs = self.cdr_emb(cdrs)
        # [batch, max_length, input_features] -> [batch, max_length, hidden_features // 2]
        node_features = self.node_emb(node_features)
        # [batch, max_length, hidden_features]
        total_features = torch.cat((cdrs, node_features), dim=-1)
        
        # transformer output with shape = [batch_size, max_length, hidden_features]
        transformer_outputs = self.transformer(total_features, feature_masks)
        
        # lstm output with shape = [batch_size, max_length, hidden_features]
        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, total_features.shape[0], self.hidden_features).to(device)
            c0 = torch.zeros(2 * self.num_layers, total_features.shape[0], self.hidden_features).to(device)
        else:
            h0 = torch.zeros(self.num_layers, total_features.shape[0], self.hidden_features).to(device)
            c0 = torch.zeros(self.num_layers, total_features.shape[0], self.hidden_features).to(device)
        lstm_pads = pack_padded_sequence(total_features, lengths=[mask[1] for mask in label_masks], batch_first=True)
        lstm_outputs, _ = self.lstm(lstm_pads, (h0, c0))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        lstm_outputs = self.dropout(self.elu(self.lstm_fc(lstm_outputs)))
        
        # [batch_size, max_length, 2 * hidden_features] -> [batch_size, max_length, output_features]
        total_outputs = self.dropout(self.elu(self.output_fc(torch.cat((transformer_outputs, lstm_outputs), dim=-1))))
        
        output = list()
        for total_output, (idx, mask) in zip(total_outputs, label_masks):
            output.append(total_output[:mask, :])
        output = torch.cat(output, dim=0)
        
        return output