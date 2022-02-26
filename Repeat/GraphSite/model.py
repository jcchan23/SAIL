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
    
    def __init__(self, in_features, hidden_features, num_layers, scale_norm, dropout):
        super(Node_Embedding, self).__init__()
        self.proj = list()
        if num_layers == 1:
            self.proj.extend([
                ScaleNorm(in_features) if scale_norm else LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_features),
                Mish()
            ])
        else:
            for i in range(num_layers):
                if i == 0:
                    self.proj.extend([
                        ScaleNorm(in_features) if scale_norm else LayerNorm(in_features),
                        nn.Dropout(dropout),
                        nn.Linear(in_features, hidden_features),
                        Mish()
                    ])
                else:
                    self.proj.extend([
                        ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_features, hidden_features),
                        Mish()
                    ])
        self.proj.append(ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features))
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


def attention(query, key, value, batch_edge_features, batch_masks, num_neighbors, dropout):
    # query, key, value.shape   = [batch, h, max_length, d]
    # batch_edge_features.shape = [batch, max_length, max_length]
    # batch_masks.shape         = [batch, max_length]
    h, d = query.shape[1], query.shape[-1]
    
    attention_scores = torch.einsum('b h m d, b h d n -> b h m n', query, rearrange(key, 'b h m d -> b h d m'))
    
    # mask padding, shape with = [batch, h, max_length, max_length]
    if batch_masks is not None:
        batch_masks = repeat(batch_masks, 'b m -> b h c m', h=h, c=batch_masks.shape[-1])
        attention_scores = attention_scores.masked_fill(batch_masks==0, -1e12)
    
    # find nearest neighbors, shape with = [batch, max_length, max_length]
    edge_weight_sorted_index = torch.argsort(torch.argsort(-batch_edge_features, axis=-1), axis=-1)
    edge_weight_sorted_mask  = (edge_weight_sorted_index < num_neighbors)
    edge_weight = batch_edge_features * edge_weight_sorted_mask
    edge_weight = edge_weight / (torch.sum(edge_weight, dim=-1, keepdim=True) + 1e-5)
    
    attention_scores = F.softmax(attention_scores * repeat(edge_weight, 'b m1 m2 -> b h m1 m2', h=h) / math.sqrt(d), dim=-1)
    
    if dropout is not None:
        attention_scores = dropout(attention_scores)
        
    node_hidden = torch.einsum('b h m n, b h n d -> b h m d', attention_scores, value)
    
    return node_hidden

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
    
    def forward(self, batch_node_features, batch_edge_features, batch_masks):
        # batch_node_features.shape = [batch, max_length, hidden_features]
        # batch_edge_features.shape = [batch, max_length, max_length]
        # batch_masks.shape         = [batch, max_length]
        
        # 1) Do all the linear projections in batch
        query = rearrange(self.layers[0](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        key   = rearrange(self.layers[1](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        value = rearrange(self.layers[2](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        
        # 2) Apply attention on all the projected vectors in batch
        batch_node_features = attention(query, key, value, batch_edge_features, batch_masks, self.num_neighbors, self.dropout)
        
        # 3) Concatenate
        batch_node_features = rearrange(batch_node_features, 'b h m d -> b m (h d)')
        
        return mish_function(self.layers[3](batch_node_features))
    

class Encoder(nn.Module):
    def __init__(self, hidden_features, num_MHSA_layers, num_attention_heads, num_neighbors, num_FFN_layers, scale_norm, dropout):
        super(Encoder, self).__init__()
        
        self.num_layers = num_MHSA_layers
        self.MHSAs = clones(Multi_Head_Self_Attention(hidden_features, num_attention_heads, num_neighbors, dropout, output_act_fn='mish'), num_MHSA_layers)
        self.FFNs = clones(Position_Wise_Feed_Forward(hidden_features, num_layers=num_FFN_layers, dropout=dropout, output_act_fn='mish'), num_MHSA_layers)
        self.norm = ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_node_features, batch_edge_features, batch_masks):
        
        for i in range(self.num_layers):
            # MHSAs
            batch_node_features = self.dropout(self.norm(batch_node_features))
            batch_node_features = batch_node_features + self.MHSAs[i](batch_node_features, batch_edge_features, batch_masks)
            # FFNs
            batch_node_features = self.dropout(self.norm(batch_node_features))
            batch_node_features = batch_node_features + self.FFNs[i](batch_node_features)
        
        return batch_node_features


class Generator(nn.Module):
    
    def __init__(self, hidden_features, output_features, num_layers, scale_norm, dropout): 
        super(Generator, self).__init__()
        self.proj = list()
        if num_layers == 1:
            self.proj.extend([
                ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, output_features),
            ])
        else:
            for i in range(num_layers):
                if i != num_layers - 1:
                    self.proj.extend([
                        ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_features, hidden_features),
                        Mish()
                    ])
                else:
                    self.proj.extend([
                        ScaleNorm(hidden_features) if scale_norm else LayerNorm(hidden_features),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_features, output_features),
                    ])
        self.proj = nn.Sequential(*self.proj)

    def forward(self, x, masks):
        # x.shape = [batch, max_length, hidden_features]
        # masks.shape = [batch, max_length]
        
        # [batch, max_length, hidden_features] -> [batch_num_nodes, hidden_features]
        x = self.proj(x)
        
        # unpadding by masks
        x, masks = rearrange(x, 'b m h -> (b m) h'), rearrange(masks, 'b m -> (b m)')
        output = torch.index_select(x, dim=0, index=torch.where(masks == True)[0])
        
        return output


class GraphSite(nn.Module):
    def __init__(self, in_features, hidden_features=64, output_features=2,
                 num_Emb_layers=2, dropout1=0.0,
                 num_MHSA_layers=2, num_FFN_layers=2, num_attention_heads=4, num_neighbors=30, dropout2=0.0,
                 num_Generator_layers=2, dropout3=0.0,
                 scale_norm=False):
        super(GraphSite, self).__init__()
        
        self.input_block = Node_Embedding(in_features, hidden_features, num_Emb_layers, scale_norm, dropout1)
        self.hidden_block = Encoder(hidden_features, num_MHSA_layers, num_attention_heads, num_neighbors, num_FFN_layers, scale_norm, dropout2)
        self.output_block = Generator(hidden_features, output_features, num_Generator_layers, scale_norm, dropout3)

    def forward(self, batch_node_features, batch_edge_features, batch_masks, device):
        
        # [batch, max_length, node_dim] -> [batch, max_length, hidden_features]
        batch_node_features = self.input_block(batch_node_features)
        
        # [batch, max_length, hidden_features]
        output = self.hidden_block(batch_node_features, batch_edge_features, batch_masks)
        
        # [batch_num_nodes, output_features]
        output = self.output_block(output, batch_masks)
        
        return output
        
        
