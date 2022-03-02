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
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

######################################## function area ########################################

class DeepANIS(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, num_layers=2, dropout=0.2, bidirectional=True):
        super(DeepANIS, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.cdr_emb = nn.Embedding(num_embeddings=22, embedding_dim=hidden_features // 2, padding_idx=0)
        self.node_emb = nn.Linear(input_features, hidden_features // 2)
        self.lstm = nn.LSTM(hidden_features, hidden_features, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_features * 2, output_features)
        else:
            self.fc = nn.Linear(hidden_features, output_features)
        self.elu = nn.ELU(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False)

        
    def forward(self, cdrs, node_features, masks, device):
        # [batch, max_length] -> [batch, max_length, hidden_features // 2]
        cdrs = self.cdr_emb(cdrs)
        # [batch, max_length, input_features] -> [batch, max_length, hidden_features // 2]
        node_features = self.node_emb(node_features)
        # [batch, max_length, hidden_features]
        total_features = torch.cat((cdrs, node_features), axis=-1)
        
        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, total_features.shape[0], self.hidden_features).to(device)
            c0 = torch.zeros(2 * self.num_layers, total_features.shape[0], self.hidden_features).to(device)
        else:
            h0 = torch.zeros(self.num_layers, total_features.shape[0], self.hidden_features).to(device)
            c0 = torch.zeros(self.num_layers, total_features.shape[0], self.hidden_features).to(device)
        
        graphs_pad = pack_padded_sequence(total_features, lengths=[mask[1] for mask in masks], batch_first=True)
        graphs_output, _ = self.lstm(graphs_pad, (h0, c0))
        graphs_pack, _ = pad_packed_sequence(graphs_output, batch_first=True)
        
        # [batch_size, max_length, 2 * hidden_features] -> [batch_size, max_length, output_features]
        graphs_pack = self.dropout(self.elu(graphs_pack))
        graphs_pack = self.dropout(self.elu(self.fc(graphs_pack)))
        
        output = list()
        for graph_pack, (idx, mask) in zip(graphs_pack, masks):
            output.append(graph_pack[:mask, :])
        output = torch.cat(output, dim=0)
        
        return output