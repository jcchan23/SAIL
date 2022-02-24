#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/02/24 14:39:24
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Node_Embedding(nn.Module):
    
    def __init__(self, in_features, hidden_features, num_layers, dropout):
        super(Node_Embedding, self).__init__()
        self.proj = list()
        if num_layers == 1:
            self.proj.extend([
                nn.LayerNorm(in_features, elementwise_affine=True),
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_features),
                Mish()
            ])
        else:
            for i in range(num_layers):
                if i == 0:
                    self.proj.extend([
                        nn.LayerNorm(in_features, elementwise_affine=True),
                        nn.Dropout(dropout),
                        nn.Linear(in_features, hidden_features),
                        Mish()
                    ])
                else:
                    self.proj.extend([
                        nn.LayerNorm(hidden_features, elementwise_affine=True),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_features, hidden_features),
                        Mish()
                    ])
        self.proj.append(nn.LayerNorm(hidden_features, elementwise_affine=True))
        self.proj = nn.Sequential(*self.proj)
        
    def forward(self, x):
        return self.proj(x)


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


def attention(query, key, value, batch_distance_matrix, batch_mask, dropout):
    pass


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, hidden_features, num_attention_heads, num_neighbors, dropout, output_act_fn):
        super(Multi_Head_Self_Attention, self).__init__()
        if hidden_features % num_attention_heads != 0:
            print("hidden features dimensions not divided by attention heads!")
            assert False
        
        self.num_attention_heads = num_attention_heads
        self.attention_hidden_features = hidden_features // num_attention_heads
        self.num_neighbors = num_neighbors
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
    
    def forward(self):
        pass


class Encoder(nn.Module):
    def __init__(self, hidden_features, num_MHSA_layers, num_attention_heads, num_FFN_layers, dropout, scale_norm):
        super(Encoder, self).__init__()
        
        self.num_layers = num_MHSA_layers
        self.MHSAs = clones(Multi_Head_Self_Attention(hidden_features, num_attention_heads, dropout, output_act_fn='mish'), num_MHSA_layers)
        self.FFNs = clones(Position_Wise_Feed_Forward(hidden_features, num_layers=num_FFN_layers, dropout=dropout, output_act_fn='mish'), num_MHSA_layers)
        self.norm = ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_node_features, batch_edge_features, batch_distance_matrix, batch_mask):
        
        for i in range(self.num_layers):
            batch_node_features = self.dropout(self.norm(batch_node_features))
            batch_node_hidden, batch_edge_hidden = self.MHSAs[i](batch_node_features, batch_edge_features, batch_distance_matrix, batch_mask)
            
            batch_node_features = batch_node_features + batch_node_hidden
            batch_edge_features = batch_edge_features + batch_edge_hidden
            
            batch_node_features = self.dropout(self.norm(batch_node_features))
            batch_node_features = batch_node_features + self.FFNs[i](batch_node_features)
        
        return batch_node_features


class Generator(nn.Module):
    def __init__(self, hidden_features, output_features, num_layers, dropout):
        super(Generator, self).__init__()
        self.gru = nn.GRU(hidden_features, hidden_features, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_features, hidden_features)
        self.bias = nn.Parameter(torch.Tensor(hidden_features), requires_grad=True)
        self.bias.data.uniform_(-1.0 / math.sqrt(hidden_features), 1.0 / math.sqrt(hidden_features))
        
        if num_layers == 1:
            self.proj = nn.Linear(hidden_features, output_features)
        else:
            self.proj = list()
            for i in range(num_layers):
                self.proj.extend([
                    nn.Linear(hidden_features, hidden_features),
                    Mish(),
                    nn.LayerNorm(hidden_features, elementwise_affine=True),
                    nn.Dropout(dropout)
                ])
                if i == num_layers - 1:
                    self.proj.append(nn.Linear(hidden_features, output_features))

            self.proj = nn.Sequential(*self.proj)

    def forward(self, x, mask):
        mask = repeat(mask, 'b m -> b m c', c=x.shape[-1])
        # [batch, max_length, hidden_features]
        out_masked = x * mask.float()
        # [batch, max_length, hidden_features]
        out_hidden = mish_function(out_masked + self.bias)
        # [2, batch, hidden_features]
        out_hidden = repeat(torch.max(out_hidden, dim=1)[0], 'b d -> h b d', h=2)
        
        # [batch, max_length, 2 * hidden_features]
        cur_message, _ = self.gru(out_masked.contiguous(), out_hidden.contiguous())
        # [batch, max_length, hidden_features]
        cur_message = mish_function((self.linear(cur_message)))
        
        # [batch, hidden_features]
        pooling = torch.sum(cur_message, dim=1) / torch.sum(mask.float(), dim=1)

        return self.proj(pooling)


class GraphSite(nn.Module):
    def __init__(self, in_features, hidden_features=64, output_features=2,
                 num_Emb_layers=2, dropout1=0.0,
                 num_MHSA_layers=2, num_FFN_layers=2, num_attention_heads=4, num_neighbors=30, dropout2=0.0,
                 num_Generator_layers=2, dropout3=0.0):
        super(GraphSite, self).__init__()
        
        self.input_block = Node_Embedding(in_features, hidden_features, num_Emb_layers, dropout1)
        self.hidden_block = Encoder(hidden_features, num_MHSA_layers, num_attention_heads, num_neighbors, num_FFN_layers, dropout2)
        self.output_block = Generator(hidden_features, output_features, num_layers=num_Generator_layers, dropout=dropout3)

    def forward(self):
        pass
