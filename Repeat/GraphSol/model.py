#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/01/17 16:00:19
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn.parameter import Parameter

######################################## function area ########################################

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, graphs):
        graphs.update_all(fn.u_mul_e('h', 'x', 'm'), fn.sum('m', 'h'))
        graphs.ndata['h'] = graphs.ndata['h'] @ self.weight
        if self.bias is not None:
            return graphs.ndata['h'] + self.bias
        else:
            return graphs.ndata['h']


class AttnPooling(nn.Module):
    def __init__(self, in_features, dense_features, n_heads):
        super(AttnPooling, self).__init__()
        self.in_features = in_features
        self.dense_features = dense_features
        self.n_heads = n_heads
        self.fc1 = nn.Linear(in_features, dense_features)
        self.fc2 = nn.Linear(dense_features, n_heads)
    
    def forward(self, graphs):
        with graphs.local_scope():
            graphs.ndata['heads'] = torch.tanh(self.fc1(graphs.ndata['h']))
            # (num_nodes, n_heads)
            graphs.ndata['heads'] = self.fc2(graphs.ndata['heads'])
            attns = dgl.softmax_nodes(graphs, 'heads')
            for i in range(self.n_heads):
                graphs.ndata[f'head_{i}'] = attns[:,i].reshape(-1, 1)
            result = []
            for i in range(self.n_heads):
                result.append(dgl.sum_nodes(graphs, 'h', f'head_{i}').unsqueeze(0))
            output = torch.mean(torch.cat(result, dim=0), dim=0)
            return output
        
class GraphSol(nn.Module):
    
    def __init__(self, in_features, hidden_features, output_features, attention_features, attention_heads):
        super(GraphSol, self).__init__()
        
        self.fc = nn.Linear(in_features, hidden_features)
        self.conv1 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.conv2 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.pooling = AttnPooling(hidden_features, dense_features=attention_features, n_heads=attention_heads)
        self.fc_final = nn.Linear(hidden_features, output_features)
        
    def forward(self, graphs, device):
        
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        h0 = graphs.ndata['h'].clone()
        
        graphs.ndata['h'] = h0 + self.ln1(self.relu(self.conv1(graphs)))
        h1 = graphs.ndata['h'].clone()
        
        graphs.ndata['h'] = h0 + h1 + self.ln2(self.relu(self.conv2(graphs)))
        
        output = self.pooling(graphs)
        output = F.sigmoid(self.fc_final(output))
        graphs.ndata.pop('h')
        
        return output
    




