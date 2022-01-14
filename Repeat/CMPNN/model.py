#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/01/14 08:50:43
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

######################################## function area ########################################

class BatchGRU(nn.Module):
    def __init__(self, hidden_features=300):
        super(BatchGRU, self).__init__()
        self.hidden_features = hidden_features
        self.gru = nn.GRU(hidden_features, hidden_features, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_features))
        self.bias.data.uniform_(-1.0 / math.sqrt(hidden_features), 1.0 / math.sqrt(hidden_features))

    def forward(self, graphs):
        hidden = graphs.ndata['h']
        message = F.relu(graphs.ndata['h'] + self.bias)
        
        a_scope, temp_idx = list(), 0
        for num_nodes in graphs.batch_num_nodes():
            a_scope.append((temp_idx, num_nodes.item()))
            temp_idx = temp_idx + num_nodes.item()
        
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        
        # padding
        message_lst, hidden_lst = list(), list()
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            # [a_size, hidden_features]
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            # append a tensor with shape [1, 1, hidden_features]
            hidden_lst.append(torch.max(cur_hidden, dim=0)[0].unsqueeze(0).unsqueeze(0))
            # [a_size, hidden_features] -> [MAX_atom_len, hidden_features]
            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len-cur_message.shape[0]))(cur_message)
            # [MAX_atom_len, hidden_features] -> [1, MAX_atom_len, hidden_features]
            message_lst.append(cur_message.unsqueeze(0))
        
        # [batch, MAX_atom_len, hidden]
        message_lst = torch.cat(message_lst, 0)
        # [1, batch, hidden]
        hidden_lst = torch.cat(hidden_lst, dim=1)
        # [2, batch, hidden]
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        # cur_message = [batch, MAX_atom_len, 2 * hidden], cur_hidden = [2, batch, hidden]
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = list()
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size, :].view(-1, 2 * self.hidden_features))
        graphs.ndata['h'] = torch.cat(cur_message_unpadding, dim=0)
        
        return dgl.mean_nodes(graphs, 'h')

class CMPNN(nn.Module):
    """
        reference from 
        https://github.com/SY575/CMPNN/
        https://github.com/jcchan23/CrystalNet
    """
    def __init__(self, node_features, edge_features, hidden_features, output_features, num_step_message_passing=3):
        super(CMPNN, self).__init__()
        
        self.num_step_message_passing = num_step_message_passing

        self.node_emb = nn.Sequential(
            nn.Linear(in_features=node_features, out_features=hidden_features),
            nn.ReLU(),
        )
        
        self.edge_emb = nn.Sequential(
            nn.Linear(in_features=edge_features, out_features=hidden_features),
            nn.ReLU(),
        )
        
        for step in range(self.num_step_message_passing - 1):
            self._modules[f'message_{step}'] = nn.Linear(in_features=hidden_features, out_features=hidden_features)
            
        self.lr = nn.Linear(hidden_features * 3, hidden_features)
        
        self.pooling = BatchGRU(hidden_features)
        
        self.ffn = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features), 
            nn.ReLU(),
            nn.Linear(hidden_features, output_features), 
        )
    
    def message_func(self, edges):
        return {'m': edges.data['h']}
    
    def reduce_func(self, nodes):
        return {'h': nodes.data['h'] + torch.sum(nodes.mailbox['m'], dim=1) * torch.max(nodes.mailbox['m'], dim=1)[0]}
    
    def reduce_final_func(self, nodes):
        return {'m_final': torch.sum(nodes.mailbox['m'], dim=1) * torch.max(nodes.mailbox['m'], dim=1)[0]}
    
    def apply_func(self, edges):
        return {'h': edges.src['h']}
 
    def forward(self, graphs, devices):
        node_features, edge_features = graphs.ndata['x'], graphs.edata['x']
        
        # [batch_num_nodes, node_features] -> [batch_num_nodes, hidden_features]
        graphs.ndata['i'] = self.node_emb(node_features)
        
        # [batch_num_edges, edge_features] -> [batch_num_edges, hidden_features]
        graphs.edata['i'] = self.edge_emb(edge_features)
        
        graphs.ndata['h'], graphs.edata['h'] = graphs.ndata['i'].clone(), graphs.edata['i'].clone()
        
        # message passing
        for step in range(self.num_step_message_passing - 1):
            graphs.update_all(self.message_func, self.reduce_func)
            graphs.apply_edges(self.apply_func)
            message_bond = self._modules[f'message_{step}'](graphs.edata['h'])
            graphs.edata['h'] = F.dropout(F.relu(graphs.edata['i'] + message_bond), p=0.0, training=self.training)
        
        # final layer message passing
        graphs.update_all(self.message_func, self.reduce_final_func)
        # [batch_num_nodes, node_features * 3] -> [batch_num_nodes, node_features]
        graphs.ndata['h'] = self.lr(torch.cat([graphs.ndata['m_final'], graphs.ndata['h'], graphs.ndata['i']], dim=1))
        # Readout step: [batch_num_nodes, node_features] -> [batch_size, node_hidden_feats * 2]
        batch_hiddens = self.pooling(graphs)
        # [batch_size, node_hidden_feats * 2] -> [batch_size, output_features]
        output = self.ffn(batch_hiddens)
        
        # clear all tensors
        graphs.ndata.pop('m_final')
        graphs.ndata.pop('h')
        graphs.ndata.pop('i')
        graphs.edata.pop('h')
        
        return output


