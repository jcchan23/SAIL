#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/01/18 09:22:35
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

######################################## function area ########################################

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * F.tanh(F.softplus(x))


def mish_function(x):
    return x * F.tanh(F.softplus(x))


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
    def __init__(self, hidden_features, num_layers, dropout, output_act_fn):
        super(Position_Wise_Feed_Forward, self).__init__()
        self.num_layers = num_layers
        self.linears = clones(nn.Linear(hidden_features, hidden_features), num_layers)
        self.dropouts = clones(nn.Dropout(dropout), num_layers)
        
        if output_act_fn == 'relu':
            self.output_act = lambda x: F.leaky_relu(x)
        elif output_act_fn == 'tanh':
            self.output_act = lambda x: F.tanh(x)
        elif output_act_fn == 'mish':
            self.output_act = lambda x: x * F.tanh(F.softplus(x))
        else:
            self.output_act = lambda x: x

    def forward(self, graphs):
        if self.num_layers == 0:
            return graphs.ndata['h']
        for i in range(self.num_layers - 1):
            graphs.ndata['h'] = self.dropouts[i](mish_function(self.linears[i](graphs.ndata['h'])))
        return self.dropouts[-1](self.output_act(self.linears[-1](graphs.ndata['h'])))
   

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
        
    def message_func(self, edges):
        # score.shape: [num_edges, num_attention_heads, attention_hidden_features] -> [num_edges, num_attention_heads, 1]
        return {'score':torch.sum(edges.src['k'] * edges.dst['q'], dim=-1, keepdim=True), 'v': edges.src['v'], 'weight': edges.data['x']}
    
    def reduce_func(self, nodes):
        # mailbox['weight'].shape = [num_same_degree_nodes, degrees, 1]
        edge_weight_sorted_index = torch.argsort(torch.argsort(-nodes.mailbox['weight'], axis=1), axis=1)
        edge_weight_sorted_mask = (edge_weight_sorted_index < self.num_neighbors)
        edge_weight = nodes.mailbox['weight'] * edge_weight_sorted_mask
        edge_weight = edge_weight / (torch.sum(edge_weight, dim=1, keepdim=True) + 1e-5)
        # mailbox['score', 'v'].shape = [num_same_degree_nodes, degrees, num_attention_heads, 1]
        attn = F.softmax(nodes.mailbox['score'] * edge_weight.unsqueeze(-2) / math.sqrt(self.attention_hidden_features), dim=1)
        return {'h': torch.sum(attn * nodes.mailbox['v'], dim=1)}
    
    def forward(self, graphs):
        graphs.ndata['q'] = self.layers[0](graphs.ndata['h']).reshape(-1, self.num_attention_heads, self.attention_hidden_features)
        graphs.ndata['k'] = self.layers[1](graphs.ndata['h']).reshape(-1, self.num_attention_heads, self.attention_hidden_features)
        graphs.ndata['v'] = self.layers[2](graphs.ndata['h']).reshape(-1, self.num_attention_heads, self.attention_hidden_features)
        
        graphs.update_all(self.message_func, self.reduce_func)
        graphs.ndata['h'] = self.layers[3](graphs.ndata['h'].reshape(-1, self.num_attention_heads * self.attention_hidden_features))
        graphs.ndata.pop('q')
        graphs.ndata.pop('k')
        graphs.ndata.pop('v')
        return self.dropout(self.output_act(graphs.ndata['h']))


class Encoder(nn.Module):
    def __init__(self, hidden_features, num_MHSA_layers, num_attention_heads, num_neighbors, num_FFN_layers, dropout):
        super(Encoder, self).__init__()
        self.num_layers = num_MHSA_layers
        self.MHSAs = clones(Multi_Head_Self_Attention(hidden_features, num_attention_heads, num_neighbors, dropout, output_act_fn='mish'), num_MHSA_layers)
        self.FFNs = clones(Position_Wise_Feed_Forward(hidden_features, num_layers=num_FFN_layers, dropout=dropout, output_act_fn='mish'), num_MHSA_layers)
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graphs):
        graphs.ndata['h'] = self.dropout(self.norm(graphs.ndata['h']))
        for i in range(self.num_layers):
            graphs.ndata['h'] = graphs.ndata['h'] + self.dropout(self.norm(self.MHSAs[i](graphs)))
            graphs.ndata['h'] = graphs.ndata['h'] + self.dropout(self.norm(self.FFNs[i](graphs)))
        return self.dropout(self.norm(graphs.ndata['h']))


class Generator(nn.Module):
    
    def __init__(self, hidden_features, output_features, num_layers, dropout): 
        super(Generator, self).__init__()
        self.proj = list()
        if num_layers == 1:
            self.proj.extend([
                nn.LayerNorm(hidden_features, elementwise_affine=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, output_features),
            ])
        else:
            for i in range(num_layers):
                if i != num_layers - 1:
                    self.proj.extend([
                        nn.LayerNorm(hidden_features, elementwise_affine=True),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_features, hidden_features),
                        Mish()
                    ])
                else:
                    self.proj.extend([
                        nn.LayerNorm(hidden_features, elementwise_affine=True),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_features, output_features),
                    ])
        self.proj = nn.Sequential(*self.proj)

    def forward(self, x):
        return self.proj(x)


class GraphSite(nn.Module):
    def __init__(self, in_features, hidden_features=64, output_features=2,
                 num_Emb_layers=2, dropout1=0.0,
                 num_MHSA_layers=2, num_FFN_layers=2, num_attention_heads=4, num_neighbors=30, dropout2=0.0,
                 num_Generator_layers=2, dropout3=0.0):
        super(GraphSite, self).__init__()
        
        self.input_block = Node_Embedding(in_features, hidden_features, num_Emb_layers, dropout1)
        self.hidden_block = Encoder(hidden_features, num_MHSA_layers, num_attention_heads, num_neighbors, num_FFN_layers, dropout2)
        self.output_block = Generator(hidden_features, output_features, num_layers=num_Generator_layers, dropout=dropout3)
        
    def forward(self, graphs, device):
        graphs.ndata['h'] = self.input_block(graphs.ndata['x'])
        graphs.ndata['h'] = self.hidden_block(graphs)
        output = self.output_block(graphs.ndata['h'])
        # [batch_num_nodes, 2]
        return output
