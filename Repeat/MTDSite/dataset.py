#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/01/24 09:33:23
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

######################################## function area ########################################


def get_loader(names_list, sequences_dict, graphs_dict, labels_dict, batch_size, shuffle, num_workers):
    dataset = ProteinDataset(names_list=names_list, sequences_dict=sequences_dict, graphs_dict=graphs_dict, labels_dict=labels_dict)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset.get_features_dim()


def collate_fn(samples):
    # sorted samples according to the length of graphs
    names, sequences, graphs, labels = map(np.array, zip(*samples))
    
    sorted_order = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)
    sorted_idx = [i[0] for i in sorted_order]
    names, sequences, graphs, labels = names[sorted_idx].tolist(), sequences[sorted_idx].tolist(), graphs[sorted_idx], labels[sorted_idx]
    
    # [batch_size, batch_longest_sequence_length, features_dim]
    graphs = pad_sequence([torch.from_numpy(graph).float() for graph in graphs], batch_first=True, padding_value=0.0)
    
    idx, masks = 0, list()
    for i in sorted_order:
        masks.append([idx, len(i[1])])
        idx += len(i[1])
    
    return names, sequences, graphs, torch.from_numpy(np.concatenate(labels, axis=-1).astype(np.int64)), torch.from_numpy(np.array(masks, dtype=np.int64))


class ProteinDataset(Dataset):
    def __init__(self, names_list, sequences_dict, graphs_dict, labels_dict):
        super(ProteinDataset, self).__init__()
        self.names = names_list
        self.sequences = [sequences_dict[name] for name in names_list]
        self.graphs = [graphs_dict[name] for name in names_list]
        self.labels = [labels_dict[name] for name in names_list]
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        return self.names[idx], self.sequences[idx], self.graphs[idx], self.labels[idx]
    
    def get_features_dim(self):
        # use an example will meet a single atom without edges
        return max([graph.shape[1] for graph in self.graphs]), None


def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    names_list, sequences_dict, labels_dict = list(), dict(), dict()
    temp_name = ""
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == "":
            continue
        elif idx % 3 == 0:
            temp_name = line[1:].split()[0]
            names_list.append(temp_name)
        elif idx % 3 == 1:
            sequences_dict[temp_name] = line
        else:
            # The third line is the enriched binding annotation: 1=binding, 0=non-binding, 2=missing structure
            labels_dict[temp_name] = [1 if num == "1" else 0 for num in line]
            temp_name = ""
    return names_list, sequences_dict, labels_dict


######################################## main area ########################################

if __name__ == "__main__":

    data_path = './data/source'
    result_path = './data/preprocess'
    
    for dataset_name in ['carbohydrate', 'dna', 'peptide', 'rna']:
        
        for mode in ['train', 'test']:
            print(f'build {dataset_name}_{mode} dataset')
            names_list, sequences_dict, labels_dict = load_dataset(f'{data_path}/{dataset_name}/{mode}.txt')
            
            graphs_dict = dict()
            for name in tqdm(names_list):
                features_df = pd.read_csv(f'{data_path}/{dataset_name}/features/{name}.csv', sep=',', index_col=0)
                # [L, 54]
                features = features_df.values.T
                
                if not features.shape[0] == len(sequences_dict[name]) == len(labels_dict[name]):
                    print(f'{name} sequence length incorrect!')
                    assert False
                
                if features.shape[1] != 54:
                    print(f'{name} features dimension incorrect!')
                    assert False
                
                graphs_dict[name] = features
        
            # save pickle files
            with open(f'{result_path}/{dataset_name}/{mode}.pickle', 'wb') as fw:
                pickle.dump([names_list, sequences_dict, graphs_dict, labels_dict], fw)
