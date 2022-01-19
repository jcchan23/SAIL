#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/01/17 16:00:25
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import dgl
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

######################################## function area ########################################

def get_loader(names_list, sequences_dict, graphs_dict, labels_dict, batch_size, shuffle, num_workers):
    dataset = ProteinDataset(names_list=names_list, sequences_dict=sequences_dict, graphs_dict=graphs_dict, labels_dict=labels_dict)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset.get_features_dim()


def collate_fn(samples):
    names, sequences, graphs, labels = map(list, zip(*samples))
    return names, sequences, dgl.batch(graphs), torch.from_numpy(np.array(labels))


def normalize_dis(mx): # from SPROF
    return 2 / (1 + np.maximum(mx/4, 1))


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
        return self.graphs[0].ndata['x'].shape[1], None
    

def build_dgl_graph(names, sequences, labels, data_path, result_path, dataset_name, mode):
    
    names_list, sequences_dict, graphs_dict, labels_dict = list(), dict(), dict(), dict()
    mean, std = np.load(f'{result_path}/esol/features/train_mean.npy'), np.load(f'{result_path}/esol/features/train_std.npy')

    for name, sequence, label in tqdm(zip(names, sequences, labels), total=len(names)):
        # [L, 20 + 20 + 30 + 14 + 7 = 91], [blosum, pssm, hmm, spd33, aaphy7]
        node_features = np.load(f'{result_path}/{dataset_name}/features/node_features/{name}.npy')        
        node_features = (node_features - mean) / std
        # [L, L]
        edge_features = np.load(f'{result_path}/{dataset_name}/features/edge_features/{name}.npy')
        src, dst = np.nonzero(edge_features)
        # [L, L] -> [num_edges]
        edge_features = normalize_adj(edge_features)
        edge_features = edge_features[np.nonzero(edge_features)]
        graph = dgl.graph((src, dst), num_nodes=len(sequence))
        
        if not len(sequence) == node_features.shape[0]:
            print(f"{dataset_name} {name} sequence, label, node features error!")
            assert False
        if not len(edge_features) == len(src) == len(dst):
            print(f"{dataset_name} {name} edge features error!")
            assert False
        
        # add features
        graph.ndata['x'] = torch.from_numpy(node_features).float()
        graph.edata['x'] = torch.from_numpy(edge_features).float()
        
        # add all
        names_list.append(name)
        sequences_dict[name] = sequence
        graphs_dict[name] = graph
        labels_dict[name] = label if isinstance(label, list) else [label]
    
    # save graphs
    with open(f'{result_path}/{dataset_name}/{dataset_name}_{mode}.pickle', 'wb') as fw:
        pickle.dump([names_list, sequences_dict, graphs_dict, labels_dict], fw)
        
        
######################################## main area ########################################

if __name__ == '__main__':
    # build the dgl graph cache
    data_path = './data/source'
    result_path = './data/preprocess'
    
    data = pd.read_csv(f'{data_path}/esol/esol_train.csv', sep=',')
    names = data['gene'].values.tolist()
    sequences = data['sequence'].values.tolist()
    labels = data['solubility'].values.tolist()
    build_dgl_graph(names, sequences, labels, data_path, result_path, dataset_name="esol", mode="train")
    
    data = pd.read_csv(f'{data_path}/esol/esol_test.csv',sep=',')
    names = data['gene'].values.tolist()
    sequences = data['sequence'].values.tolist()
    labels = data['solubility'].values.tolist()
    build_dgl_graph(names, sequences, labels, data_path, result_path, dataset_name="esol", mode="test")
    
    data = pd.read_csv(f'{data_path}/scerevisiae/scerevisiae_test.csv', sep=',')
    names = data['gene'].values.tolist()
    sequences = data['sequence'].values.tolist()
    labels = data['solubility'].values.tolist()
    build_dgl_graph(names, sequences, labels, data_path, result_path, dataset_name="scerevisiae", mode="test")
    