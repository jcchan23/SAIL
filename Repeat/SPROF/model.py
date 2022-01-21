#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/01/20 17:27:18
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
import dgl.function as fn
from torch.nn.parameter import Parameter

######################################## function area ########################################


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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


class DenseGCN(nn.Module):
    
    def __init__(self, in_features, hidden_features, output_features, num_layers=3, dropout=0.0):
        super(DenseGCN, self).__init__()
        
        # input block
        self.fc_first = nn.Sequential(
            nn.LayerNorm(in_features, elementwise_affine=True),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features),
            nn.ELU(inplace=False)
        )
        
        # hidden block
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU(inplace=False)
        self.convs = clones(GraphConvolution(hidden_features, hidden_features), self.num_layers)
        
        # output block
        self.fc_final = nn.Sequential(
            nn.LayerNorm(hidden_features, elementwise_affine=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ELU(inplace=False),
            nn.Linear(hidden_features, output_features),
        )
        
    def forward(self, graphs, device):
        
        graphs.ndata['h'] = self.fc_first(graphs.ndata['x'])
        graphs.ndata['h'] = self.dropout(self.norm(graphs.ndata['h']))
        hiddens = [graphs.ndata['h'].clone()]
        
        for i in range(self.num_layers):
            graphs.ndata['h'] = self.dropout(self.norm(self.convs[i](graphs)))
            cur_hidden = graphs.ndata['h'].clone()
            
            for hidden in hiddens:
                graphs.ndata['h'] = graphs.ndata['h'] + hidden
             
            hiddens.append(cur_hidden)  
        
        output = self.fc_final(graphs.ndata['h'])
        graphs.ndata.pop('h')
        
        return output
