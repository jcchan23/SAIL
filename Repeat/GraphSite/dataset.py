#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/02/24 10:29:05
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

######################################## function area ########################################

def get_loader(names_list, sequences_dict, graphs_dict, labels_dict, batch_size, shuffle, num_workers):
    dataset = ProteinDataset(names_list=names_list, sequences_dict=sequences_dict, graphs_dict=graphs_dict, labels_dict=labels_dict)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset.get_features_dim()


def collate_fn(samples):
    names, sequences, graphs, labels = map(list, zip(*samples))
    
    # padding features
    batch_node_features, batch_edge_features = list(), list()
    max_length = max([len(sequence) for sequence in sequences])
    for (node_features, edge_features) in graphs:
        batch_node_features.append(pad_array(node_features, (max_length, node_features.shape[-1])))
        batch_edge_features.append(pad_array(edge_features, (max_length, max_length)))
    
    # mask labels
    idx, masks = 0, list()
    for label in labels:
        masks.append([idx, len(label)])
        idx += len(label)
    
    return names, sequences, \
        torch.from_numpy(np.array(batch_node_features)).float(),\
        torch.from_numpy(np.array(batch_edge_features)).float(),\
        torch.from_numpy(np.concatenate(labels, axis=-1)).long(),\
        torch.from_numpy(np.array(masks)).long()
    

def pad_array(array, shape):
    padded_array = np.zeros(shape, dtype=np.float64)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def normalize_dis(mx):
    mx = 2 / (1 + np.maximum(mx/4, 1))
    mx[np.isinf(mx)] = 0
    mx[np.isnan(mx)] = 0
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


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
        return max([node_features.shape[1] for (node_features, _) in self.graphs]), None
    

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
            temp_name = line[1:]
            names_list.append(line[1:])
        elif idx % 3 == 1:
            sequences_dict[temp_name] = line
        else:
            labels_dict[temp_name] = [int(num) for num in line]
            temp_name = ""
    return names_list, sequences_dict, labels_dict


######################################## main area ########################################

if __name__ == '__main__':
    # build the dgl graph cache
    data_path = './data/source'
    result_path = './data/preprocess'
    
    for dataset_name in ['train_569', 'test_129', 'test_181']:
        print(f'build {dataset_name} dgl graph')
        names_list, sequences_dict, labels_dict = load_dataset(f'{data_path}/{dataset_name}.fasta')

        graphs_dict = dict()
        
        for name in tqdm(names_list):
            sequence, label = sequences_dict[name], labels_dict[name]
            af2_features = np.load(f'{result_path}/features/af2_node_features/{name}.npy')
            pssm_features = np.load(f'{result_path}/features/pssm/{name}.npy')
            hmm_features = np.load(f'{result_path}/features/hmm/{name}.npy')
            dssp_features = np.load(f'{result_path}/features/dssp/{name}.npy')
            
            # [L, 384 + 20 + 20 + 14 = 438]
            node_features = np.concatenate([af2_features, pssm_features, hmm_features, dssp_features], axis=-1)

            # [L, L]
            distance_map = np.load(f'{result_path}/features/af2_edge_features/{name}.npy')
            # mask the -1's rows and columns
            distance_map = np.where(distance_map >= 0, distance_map, float('inf'))
            distance_weight = normalize_dis(distance_map)
            edge_features = distance_weight / (np.sum(distance_weight, axis=-1, keepdims=True) + 1e-5)
            
            if not len(sequence) == len(label) == node_features.shape[0]:
                print(f"{dataset_name} {name} sequence, label, node features error!")
                assert False
            if not len(sequence) == edge_features.shape[0] == edge_features.shape[1]:
                print(f"{dataset_name} {name} edge features error!")
                assert False
            
            graphs_dict[name] = (node_features, edge_features)
            
        # save graphs
        with open(f'{result_path}/{dataset_name}.pickle', 'wb') as fw:
            pickle.dump([names_list, sequences_dict, graphs_dict, labels_dict], fw)
