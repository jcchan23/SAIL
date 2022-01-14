#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/01/14 08:50:29
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import torch
import dgl
import pickle
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from torch.utils.data import Dataset, DataLoader

######################################## function area ########################################

def get_loader(smiles_list, mols_dict, graphs_dict, labels_dict, batch_size, shuffle, num_workers):
    dataset = MoleculeDataset(smiles_list, mols_dict, graphs_dict, labels_dict)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset.get_features_dim()


def collate_fn(samples):
    smiles, mols, graphs, labels = map(list, zip(*samples))
    return smiles, mols, dgl.batch(graphs), torch.from_numpy(np.array(labels))


def one_hot_vector(val, lst, add_unknown=True):
    vec = np.zeros(len(lst) + 1) if add_unknown else np.zeros(len(lst))
    vec[lst.index(val) if val in lst else -1] = 1    
    return vec


def get_atom_features(atom):
    # 100 + 1 = 101 dimensions
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)], add_unknown=True)
    # 6 + 1 = 7 dimensions
    v2 = one_hot_vector(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5], add_unknown=True)
    # 5 + 1 = 6 dimensions
    v3 = one_hot_vector(atom.GetFormalCharge(), [-2, -1, 0, 1, 2], add_unknown=True)
    # 4 + 1 = 5 dimensions
    v4 = one_hot_vector(int(atom.GetChiralTag()), [0, 1, 2, 3], add_unknown=True)
    # 5 + 1 = 6 dimensions
    v5 = one_hot_vector(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4], add_unknown=True)
    # 5 + 1 = 6 dimensions
    v6 = one_hot_vector(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ], add_unknown=True)
    # 1 dimension
    v7 = [1 if atom.GetIsAromatic() else 0]
    # 1 dimension
    v8 = [atom.GetMass() * 0.01]
    # 133 dimensions
    return np.concatenate([v1, v2, v3, v4, v5, v6, v7, v8], axis=0)
    

def get_bond_features(bond):
    # 4 + 1 = 5 dimensions
    v1 = one_hot_vector(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ], add_unknown=True)
    # 6 + 1 = 7 dimensions
    v2 = one_hot_vector(bond.GetStereo(), [
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOZ
        ], add_unknown=True)
    # 3 dimensions
    v3 = [int(bond.GetIsConjugated()), int(bond.GetIsAromatic()), int(bond.IsInRing())]
    # 15 dimensions
    return np.concatenate([v1, v2, v3], axis=0)


class MoleculeDataset(Dataset):
    
    def __init__(self, smiles_list, mols_dict, graphs_dict, labels_dict):
        super(MoleculeDataset, self).__init__()
        self.smiles = smiles_list
        self.mols = [mols_dict[smile] for smile in smiles_list]
        self.graphs = [graphs_dict[smile] for smile in smiles_list]
        self.labels = [labels_dict[smile] for smile in smiles_list]
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.smiles[idx], self.mols[idx], self.graphs[idx], self.labels[idx]
    
    def get_features_dim(self):
        # use an example will meet a single atom without edges
        return max([graph.ndata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs]), \
            max([graph.edata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs])


def build_dgl_graph(smiles, labels, store_path, mode="bbbp"):
    
    print(f"Building {mode} dgl graphs")
    smiles_list, mols_dict, graphs_dict, labels_dict = list(), dict(), dict(), dict()
    
    for smile, label in tqdm(zip(smiles, labels), total=len(smiles)):
        # smile to mol
        mol = Chem.MolFromSmiles(smile)
        
        # check the sanitizemol
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        
        # set stereochemistry
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)
        AllChem.ComputeGasteigerCharges(mol)
        
        # (num_nodes, num_nodes)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        src, dst = np.nonzero(adjacency_matrix)
        graph = dgl.graph((src, dst), num_nodes=adjacency_matrix.shape[0])
        
        node_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
        graph.ndata['x'] = torch.from_numpy(node_features).float()
        
        # (num_edges, edge_in_dim)                
        rows, cols, bond_features = list(), list(), list()
        for bond in mol.GetBonds():
            features = get_bond_features(bond)
            rows.append(bond.GetBeginAtom().GetIdx())
            cols.append(bond.GetEndAtom().GetIdx())
            bond_features.append(copy.deepcopy(features))
            
            rows.append(bond.GetEndAtom().GetIdx())
            cols.append(bond.GetBeginAtom().GetIdx())
            bond_features.append(copy.deepcopy(features))
        
        # sorted edge features with (src, dst) index
        edge_features = np.array([np.concatenate([bond_feature, node_features[row, :]], axis=-1) for row, col, bond_feature in sorted(zip(rows, cols, bond_features))])
        graph.edata['x'] = torch.from_numpy(edge_features).float()
        
        # add all, re-smiled for keeping the consistent type of smile string.
        smile = Chem.MolToSmiles(mol)
        smiles_list.append(smile)
        mols_dict[smile] = mol
        graphs_dict[smile] = graph
        labels_dict[smile] = label if isinstance(label, list) else [label]

    # store graph
    with open(f'{store_path}/{mode}.pickle', 'wb') as fw:
        pickle.dump([smiles_list, mols_dict, graphs_dict, labels_dict], fw)

if __name__ == "__main__":
    # Disable warning generated by rdkit c++
    RDLogger.DisableLog('rdApp.*')
    data_path = './data/source'
    result_path = './data/preprocess'
    
    # bbbp
    data = pd.read_csv(f'{data_path}/bbbp.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['p_np'].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="bbbp")
    
    # clintox
    data = pd.read_csv(f'{data_path}/clintox.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,1:].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="clintox")
    
    # esol
    data = pd.read_csv(f'{data_path}/esol.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['measured log solubility in mols per litre'].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="esol")
    
    # freesolv
    data = pd.read_csv(f'{data_path}/freesolv.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['expt'].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="freesolv")
    
    # lipophilicity
    data = pd.read_csv(f'{data_path}/lipophilicity.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['exp'].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="lipophilicity")
    
    # sider
    data = pd.read_csv(f'{data_path}/sider.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,1:].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="sider")
    
    # tox21
    data = pd.read_csv(f'{data_path}/tox21.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,0:-2].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="tox21")
    
    # toxcast
    data = pd.read_csv(f'{data_path}/toxcast.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,1:].values.tolist()
    build_dgl_graph(smiles, labels, result_path, mode="toxcast")


