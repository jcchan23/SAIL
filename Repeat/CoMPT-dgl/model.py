#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graphsite.py
@Time    :   2022/01/14 08:59:51
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import dgl
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

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
            x_embedding += self.embedding[i](x[:, i])
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
            x_embedding += self.embedding[i](x[:, i])
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

    def forward(self, graphs):
        if self.num_layers == 0:
            return graphs.ndata['h']
        for i in range(self.num_layers - 1):
            graphs.ndata['h'] = self.dropout(mish_function(self.linears[i](graphs.ndata['h'])))
        return self.dropout(self.output_act(self.linears[-1](graphs.ndata['h'])))
   

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
        
    def message_func(self, edges):
        # q,k.shape: [num_edges, num_attention_heads, attention_hidden_features] -> [num_edges, num_attention_heads, 1]
        in_node_edge_interaction = torch.sum(edges.src['q'] * edges.data['k'], dim=-1, keepdims=True)
        out_node_edge_interaction = torch.sum(edges.dst['q'] * edges.data['k'], dim=-1, keepdims=True)
        diag_node_node_interaction = torch.sum(edges.src['q'] * edges.dst['q'], dim=-1, keepdims=True)
        return {'in_message':in_node_edge_interaction, 'out_message': out_node_edge_interaction, 'diag_message': diag_node_node_interaction,
                'distance':edges.data['distance'], 'v': edges.dst['v']}
    
    def reduce_func(self, nodes):
        # in/out/diag_score.shape = [batch_same_degrees, degrees, num_attention_heads, 1], distance.shape = [batch_same_degrees, degrees, 1]
        in_score = F.softmax(nodes.mailbox['in_message'] / math.sqrt(self.attention_hidden_features), dim=1) * (nodes.mailbox['distance'].unsqueeze(-2) ** self.attenuation_lambda)
        out_score = F.softmax(nodes.mailbox['out_message'] / math.sqrt(self.attention_hidden_features), dim=1) * (nodes.mailbox['distance'].unsqueeze(-2) ** self.attenuation_lambda)
        diag_score = F.softmax(nodes.mailbox['diag_message'] / math.sqrt(self.attention_hidden_features), dim=1) * (nodes.mailbox['distance'].unsqueeze(-2) ** self.attenuation_lambda) 
        score = in_score + out_score + diag_score
        return {'h': torch.sum(score * nodes.mailbox['v'], dim=1)}
    
    def forward(self, graphs):
        graphs.ndata['q'] = self.layers[0](graphs.ndata['h']).reshape(-1, self.num_attention_heads, self.attention_hidden_features)
        graphs.edata['k'] = self.layers[1](graphs.edata['h']).reshape(-1, self.num_attention_heads, self.attention_hidden_features)
        graphs.ndata['v'] = self.layers[2](graphs.ndata['h']).reshape(-1, self.num_attention_heads, self.attention_hidden_features)
        torch.clamp(self.attenuation_lambda, min=0, max=1)
        
        graphs.update_all(self.message_func, self.reduce_func)
        graphs.ndata['h'] = self.layers[3](graphs.ndata['h'].reshape(-1, self.num_attention_heads * self.attention_hidden_features))
        graphs.ndata.pop('q')
        graphs.edata.pop('k')
        graphs.ndata.pop('v')
        return self.dropout(self.output_act(graphs.ndata['h']))


class Encoder(nn.Module):
    def __init__(self, hidden_features, num_MHSA_layers, num_attention_heads, num_FFN_layers, dropout):
        super(Encoder, self).__init__()
        self.num_layers = num_MHSA_layers
        self.MHSAs = clones(Multi_Head_Self_Attention(hidden_features, num_attention_heads, dropout, output_act_fn='mish'), num_MHSA_layers)
        self.FFNs = clones(Position_Wise_Feed_Forward(hidden_features, num_layers=num_FFN_layers, dropout=dropout, output_act_fn='mish'), num_MHSA_layers)
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graphs):
        for i in range(self.num_layers):
            # post LN and Add
            graphs.ndata['h'] = self.dropout(self.norm(graphs.ndata['h']))
            graphs.ndata['h'] = graphs.ndata['h'] + self.MHSAs[i](graphs)
            
            # post LN and Add
            graphs.ndata['h'] = self.dropout(self.norm(graphs.ndata['h']))
            graphs.ndata['h'] = graphs.ndata['h'] + self.FFNs[i](graphs)
            
        return graphs


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


class Generator(nn.Module):
    def __init__(self, hidden_features, output_features, num_layers, dropout):
        super(Generator, self).__init__()
        self.gru = BatchGRU(hidden_features=hidden_features)
        self.linear = nn.Linear(2 * hidden_features, hidden_features)
        
        if num_layers == 1:
            self.proj = nn.Linear(hidden_features, output_features)
        else:
            self.proj = list()
            for _ in range(num_layers - 1):
                self.proj.extend([
                    nn.LayerNorm(hidden_features, elementwise_affine=True),
                    nn.Linear(hidden_features, hidden_features),
                    Mish(),
                    nn.Dropout(dropout)
                ])
            self.proj.append(nn.Linear(hidden_features, output_features))
            self.proj = nn.Sequential(*self.proj)

    def forward(self, graphs):
        x = self.linear(self.gru(graphs))
        return self.proj(x)


class CoMPT(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_features=64, output_features=2,
                 num_MHSA_layers=2, num_attention_heads=4,
                 num_FFN_layers=1, num_Generator_layers=2, dropout=0.0):
        super(CoMPT, self).__init__()
        
        self.input_node_block = Node_Embedding(hidden_features)
        self.input_edge_block = Edge_Embedding(hidden_features)
        self.position_block = Position_Encoding(hidden_features)
        self.hidden_block = Encoder(hidden_features, num_MHSA_layers, num_attention_heads, num_FFN_layers, dropout)
        self.output_block = Generator(hidden_features, output_features, num_layers=num_Generator_layers, dropout=dropout)
        
    def forward(self, graphs, device):
        # input
        graphs.ndata['h'] = self.input_node_block(graphs.ndata['x'])
        graphs.edata['h'] = self.input_edge_block(graphs.edata['x'])
        
        # add node position
        graphs.apply_nodes(lambda nodes:{'h': nodes.data['h'] + self.position_block(nodes.data['index']).squeeze(-2)})
        # add edge direction
        graphs.apply_edges(fn.u_add_e('h', 'h', 'h'))
        
        # hidden
        graphs = self.hidden_block(graphs)
        
        # output
        output = self.output_block(graphs)
        
        return output
