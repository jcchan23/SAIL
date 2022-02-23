#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_graph.py
@Time    :   2022/02/17 09:17:52
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import math
import copy
from re import L
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
    def __init__(self, embedding_features):
        super(Node_Embedding, self).__init__()
        self.embedding = nn.ModuleList([
            # AtomicNum
            nn.Embedding(120, embedding_features, padding_idx=0),
            # Hybridization
            nn.Embedding(20, embedding_features, padding_idx=0),
            # TotalDegree
            nn.Embedding(20, embedding_features, padding_idx=0),
            # TotalNumHs
            nn.Embedding(20, embedding_features, padding_idx=0),
            # FormalCharge
            nn.Embedding(20, embedding_features, padding_idx=0),
            # TotalValence
            nn.Embedding(20, embedding_features, padding_idx=0),
            # NumRadicalElectrons
            nn.Embedding(20, embedding_features, padding_idx=0),
            # IsAromatic
            nn.Embedding(10, embedding_features, padding_idx=0),
            # IsInRing
            nn.Embedding(10, embedding_features, padding_idx=0)
        ])
        
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.embedding[i](x[:, :, i])
        return x_embedding


class Edge_Embedding(nn.Module):
    def __init__(self, embedding_features):
        super(Edge_Embedding, self).__init__()
        
        self.embedding = nn.ModuleList([
            # BondType
            nn.Embedding(20, embedding_features, padding_idx=0),
            # Stereo
            nn.Embedding(20, embedding_features, padding_idx=0),
            # IsConjugated
            nn.Embedding(10, embedding_features, padding_idx=0),
            # IsAromatic
            nn.Embedding(10, embedding_features, padding_idx=0),
            # InRing
            nn.Embedding(10, embedding_features, padding_idx=0)
        ])
    
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.embedding[i](x[:, :, :, i])
        return x_embedding


class Position_Encoding(nn.Module):
    def __init__(self, embedding_features):
        super(Position_Encoding, self).__init__()
        self.pe = nn.Embedding(500, embedding_features, padding_idx=0)
    
    def forward(self, x):
        return self.pe(x)


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
    # query.shape = [batch, h, max_length, d]
    # key.shape = [batch, h, max_length, max_length, d]
    # value.shape = [batch, h, max_length, d]
    # batch_distance_matrix.shape = [batch, max_length, max_length]
    # batch_mask.shape = [batch, max_length]
    
    h, d = query.shape[1], query.shape[-1]
    
    out_scores = torch.einsum('b h m d, b h m n d -> b h m n', query, key) / math.sqrt(d)
    in_scores = torch.einsum('b h m d, b h m n d -> b h n m', query, key) / math.sqrt(d)
    
    if batch_mask is not None:
        batch_mask = repeat(batch_mask, 'b m -> b h c m', h=h, c=batch_mask.shape[-1])
        out_scores = out_scores.masked_fill(batch_mask==0, -np.inf)
        in_scores = in_scores.masked_fill(batch_mask==0, -np.inf)
            
    out_attn = F.softmax(out_scores, dim=-1)
    in_attn = F.softmax(in_scores, dim=-1)
    diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    message = out_attn + in_attn - diag_attn
    message = message * repeat(batch_distance_matrix, 'b m1 m2 -> b h m1 m2', h=h)
    
    if dropout is not None:
        message = dropout(message)
    
    node_hidden = torch.einsum('b h m n, b h n d -> b h m d', message, value)
    
    if torch.isnan(node_hidden).any():
        print("after update")
        assert False

    edge_hidden = repeat(message, 'b h m1 m2 -> b h m1 m2 c', c=key.shape[-1]) * key
    
    return node_hidden, edge_hidden
    

class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, hidden_features, num_attention_heads, dropout, output_act_fn):
        super(Multi_Head_Self_Attention, self).__init__()
        if hidden_features % num_attention_heads != 0:
            print("hidden features dimensions not divided by attention heads!")
            assert False
        
        self.num_attention_heads = num_attention_heads
        self.attention_hidden_features = hidden_features // num_attention_heads
        self.layers = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(5)])
        self.dropout = nn.Dropout(dropout)
        self.attenuation_lambda = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        
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

    def forward(self, batch_node_features, batch_edge_features, batch_distance_matrix, batch_mask):
        
        # 1) prepare distance matrix with shape (batch, max_length, max_length)
        
        torch.clamp(self.attenuation_lambda, min=0, max=1)
        
        batch_distance_matrix = self.attenuation_lambda * batch_distance_matrix
        batch_distance_matrix = batch_distance_matrix.masked_fill(repeat(batch_mask, 'b m -> b c m', c=batch_mask.shape[-1]) == 0, np.inf)
        batch_distance_matrix = torch.exp(-batch_distance_matrix)
        
        # 2) Do all the linear projections in batch
        query = rearrange(self.layers[0](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        key = rearrange(self.layers[1](batch_edge_features), 'b m1 m2 (h d) -> b h m1 m2 d', h=self.num_attention_heads, d=self.attention_hidden_features)
        value = rearrange(self.layers[2](batch_node_features), 'b m (h d) -> b h m d', h=self.num_attention_heads, d=self.attention_hidden_features)
        
        # 3) Apply attention on all the projected vectors in batch
        batch_node_features, batch_edge_features = attention(query, key, value, batch_distance_matrix, batch_mask, self.dropout)
                
        # 4) Concatenate
        batch_node_features = rearrange(batch_node_features, 'b h m d -> b m (h d)')
        batch_edge_features = rearrange(batch_edge_features, 'b h m1 m2 d -> b m1 m2 (h d)')

        return mish_function(self.layers[3](batch_node_features)), mish_function(self.layers[4](batch_edge_features))


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


class CoMPT(nn.Module):
    def __init__(self, hidden_features=64, output_features=2,
                 num_MHSA_layers=2, num_attention_heads=4,
                 num_FFN_layers=2, num_Generator_layers=2, dropout=0.0, scale_norm=False):
        super(CoMPT, self).__init__()
        
        self.input_node_block = Node_Embedding(hidden_features)
        self.input_edge_block = Edge_Embedding(hidden_features)
        self.position_block = Position_Encoding(hidden_features)
        self.hidden_block = Encoder(hidden_features, num_MHSA_layers, num_attention_heads, num_FFN_layers, dropout, scale_norm)
        self.output_block = Generator(hidden_features, output_features, num_layers=num_Generator_layers, dropout=dropout)

    def forward(self, batch_node_features, batch_edge_features, batch_distance_matrix, batch_mask, device):
        
        # [batch, max_length, node_dim] -> [batch, max_length, hidden_features]
        batch_node_features = self.input_node_block(batch_node_features[:, :, :-1]) + self.position_block(batch_node_features[:, :, -1])
        
        # [batch, max_length, max_length, edge_dim] -> [batch, max_length, max_length, hidden_features]
        batch_edge_features = self.input_edge_block(batch_edge_features)
        
        # [batch, max_length, hidden_features]
        output = self.hidden_block(batch_node_features, batch_edge_features, batch_distance_matrix, batch_mask)
        
        # [batch, max_length, hidden_features] -> [batch, hidden_features] -> [batch_ output_features]
        output = self.output_block(output, batch_mask)
        
        return output
