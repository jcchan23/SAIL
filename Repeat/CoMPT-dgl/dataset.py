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
import dgl
import torch
import pickle
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


def numeric(val, lst):
    return lst.index(val) + 1 if val in lst else 0


def get_atom_features(atom):
    # 10 dimensions
    return np.array([
        numeric(atom.GetAtomicNum(), list(range(1, 120))),
        numeric(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]),
        numeric(atom.GetTotalDegree(), list(range(1, 9))),
        numeric(atom.GetTotalNumHs(), list(range(1, 9))),
        numeric(atom.GetFormalCharge(), list(range(-5, 6))),
        numeric(atom.GetTotalValence(), list(range(1, 9))),
        numeric(atom.GetNumRadicalElectrons(), list(range(1, 9))),
        numeric(atom.GetIsAromatic(), list(range(2))),
        numeric(atom.IsInRing(), list(range(2))),
        int(atom.GetIdx() + 1)
    ], dtype=np.int16)
    

def get_bond_features(bond):
    # 5 dimensions
    return np.array([
        numeric(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]),
        numeric(bond.GetStereo(), [
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOZ
        ]),
        numeric(bond.GetIsConjugated(), list(range(2))),
        numeric(bond.GetIsAromatic(), list(range(2))),
        numeric(bond.IsInRing(), list(range(2)))
    ], dtype=np.int16)


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
        distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
        distance_matrix = np.where(distance_matrix >= 0, distance_matrix, float('inf'))
        distance_weight = np.exp(-1.0 * distance_matrix)
        
        src, dst = np.nonzero(distance_weight)
        graph = dgl.graph((src, dst), num_nodes=distance_matrix.shape[0])
        
        # add node features
        node_features = np.array([get_atom_features(atom)[:-1] for atom in mol.GetAtoms()], dtype=np.int16)
        node_indices = np.array([get_atom_features(atom)[-1] for atom in mol.GetAtoms()], dtype=np.int16)
        graph.ndata['x'] = torch.from_numpy(node_features).long()
        graph.ndata['index'] = torch.from_numpy(node_indices).long().reshape(-1, 1)
        
        # (num_edges, edge_in_dim)
        edge_features, distances = list(), list()
        for row in range(mol.GetNumAtoms()):
            for col in range(mol.GetNumAtoms()):
                if distance_weight[row][col] == 0.0:
                    continue   
                else:
                    distances.append(distance_weight[row][col])        
                bond = mol.GetBondBetweenAtoms(row, col)
                if bond is None:
                    edge_features.append(np.array([0 for _ in range(5)], dtype=np.int16))
                else:
                    edge_features.append(get_bond_features(bond))
   
        graph.edata['x'] = torch.from_numpy(np.array(edge_features)).long()
        graph.edata['distance'] = torch.from_numpy(np.array(distances)).float().reshape(-1, 1)
        
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


