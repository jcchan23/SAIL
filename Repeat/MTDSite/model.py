#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/01/24 11:31:31
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


class MTDSite(nn.Module):
    def __init__(self, input_features=54, hidden_features=128, output_features=2, num_layers=2, dropout=0.1, bidirectional=True):
        super(MTDSite, self).__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.num_layers= num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_features * 2, output_features)
        else:
            self.fc = nn.Linear(hidden_features, output_features)
        self.elu = nn.ELU(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False)

        
    def forward(self, graphs, masks, device):

        if self.bidirectional:
            h0 = torch.zeros(2 * self.num_layers, graphs.shape[0], self.hidden_features).to(device)
            c0 = torch.zeros(2 * self.num_layers, graphs.shape[0], self.hidden_features).to(device)
        else:
            h0 = torch.zeros(self.num_layers, graphs.shape[0], self.hidden_features).to(device)
            c0 = torch.zeros(self.num_layers, graphs.shape[0], self.hidden_features).to(device)
        
        graphs_pad = pack_padded_sequence(graphs, lengths=[mask[1] for mask in masks], batch_first=True)
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
