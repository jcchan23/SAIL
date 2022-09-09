#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/03/01 13:09:25
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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

######################################## function area ########################################

amino2id = {
    'U': 0, 'X': 21,
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
    'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 
}

id2amino = {value:key for key, value in amino2id.items()}

def get_loader(names_list, sequences_dict, graphs_dict, labels_dict, batch_size, shuffle, num_workers):
    dataset = ProteinDataset(names_list=names_list, sequences_dict=sequences_dict, graphs_dict=graphs_dict, labels_dict=labels_dict)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset.get_features_dim()


def collate_fn(samples):
    names, sequences, graphs, labels = map(list, zip(*samples))
    
    sorted_order = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)
    sorted_idx = [i[0] for i in sorted_order]
    
    batch_names, batch_sequences, batch_cdrs, batch_node_features, batch_labels = list(), list(), list(), list(), list()
    
    for idx in sorted_idx:
        batch_names.append(names[idx])
        batch_sequences.append(sequences[idx])
        batch_cdrs.append(graphs[idx][0])
        batch_node_features.append(graphs[idx][1])
        batch_labels.append(labels[idx]) 
    
    # [batch_size, batch_longest_sequence_length, features_dim]
    batch_cdrs = pad_sequence(torch.from_numpy(np.array(batch_cdrs)).long(), batch_first=True, padding_value=0)
    batch_node_features = pad_sequence(torch.from_numpy(np.array(batch_node_features)).float(), batch_first=True, padding_value=0.0)
    
    idx, batch_masks = 0, list()
    for i in sorted_order:
        batch_masks.append([idx, len(i[1])])
        idx += len(i[1])
    
    return batch_names, batch_sequences, batch_cdrs, batch_node_features, \
        torch.from_numpy(np.concatenate(batch_labels, axis=-1).astype(np.int64)).long(), torch.from_numpy(np.array(batch_masks, dtype=np.int64)).long()


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
        return max([node_features.shape[1] for _, node_features in self.graphs]), None
    

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
    
    names_list, sequence_dict, labels_dict = load_dataset(f'{data_path}/total_277.fasta')
    
    # [pssms, hmms, spd33s] = 20 + 30 + 14 = 64
    min_array = [float('inf') for _ in range(64)]
    max_array = [float('-inf') for _ in range(64)]
    
    # calculate min max
    print(f'calculate features min max dimensions')
    for name in tqdm(names_list):
        node_features = np.load(f'{result_path}/features/node_features/{name}.npy')
    
        # calculate min max
        temp_min_array = np.min(node_features, axis=0)
        temp_max_array = np.max(node_features, axis=0)
        for i in range(64):
            min_array[i] = min(min_array[i], temp_min_array[i])
            max_array[i] = max(max_array[i], temp_max_array[i])
    
    # save min max
    np.save(f'{result_path}/features/min_array.npy', np.array(min_array))
    np.save(f'{result_path}/features/max_array.npy', np.array(max_array))
    
    # load min max
    min_array = np.load(f'{result_path}/features/min_array.npy')
    max_array = np.load(f'{result_path}/features/max_array.npy')
    
    # construct graph
    print(f'construct graph')
    graph_dict = dict()
    for name in tqdm(names_list):
        cdrs = np.load(f'{result_path}/features/embedding/{name}.npy')
        node_features = np.load(f'{result_path}/features/node_features/{name}.npy')
        labels = np.load(f'{result_path}/features/label/{name}.npy')
        
        # check cdrs and labels
        if cdrs.tolist() != [amino2id[amino] for amino in sequence_dict[name]]:
            print(name)
            print(sequence_dict[name])
            print("".join([id2amino[num] for num in cdrs]))
            print(cdrs.tolist())
            print([amino2id[amino] for amino in sequence_dict[name]])
            assert False
        if labels.tolist() != labels_dict[name]:
            print(name)
            print(labels_dict[name])
            print(labels)
            assert False
        
        node_features = (node_features - min_array) / (max_array - min_array)
        graph_dict[name] = (cdrs, node_features)

    # save pickle files
    with open(f'{result_path}/total_277.pickle', 'wb') as fw:
        pickle.dump([names_list, sequence_dict, graph_dict, labels_dict], fw)
