#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset_graph.py
@Time    :   2022/02/16 09:52:53
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import math
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
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


def pad_array(array, shape):
    padded_array = np.zeros(shape, dtype=np.int32)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def collate_fn(samples):
    smiles, mols, graphs, labels = map(list, zip(*samples))
    
    batch_node_features, batch_edge_features, batch_distance_matrix = list(), list(), list()
    max_length = max([distance_matrix.shape[0] for (node_features, edge_features, distance_matrix) in graphs])
    
    for (node_features, edge_features, distance_matrix) in graphs:
        batch_node_features.append(pad_array(node_features, (max_length, node_features.shape[-1])))
        batch_edge_features.append(pad_array(edge_features, (max_length, max_length, edge_features.shape[-1])))
        batch_distance_matrix.append(pad_array(distance_matrix, (max_length, max_length)))

    return smiles, mols, \
            torch.from_numpy(np.array(batch_node_features)).long(), \
            torch.from_numpy(np.array(batch_edge_features)).long(), \
            torch.from_numpy(np.array(batch_distance_matrix)).long(), \
            torch.from_numpy(np.array(labels)).float()
    

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
    ], dtype=np.int32)


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
    ], dtype=np.int32)


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


def build_graph(smiles, labels, store_path, mode="bbbp"):
    
    print(f"Building {mode} graphs")
    smiles_list, mols_dict, graphs_dict, labels_dict = list(), dict(), dict(), dict()
    
    for smile, label in tqdm(zip(smiles, labels), total=len(smiles)):
        # smile to mol
        mol = Chem.MolFromSmiles(smile)
        
        # check the sanitizemol
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        
        # Remove unbond hydrogens
        mol = Chem.RemoveHs(mol)
        
        # set stereochemistry
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)
        
        # (num_nodes, num_nodes)
        distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(np.int32)
        
        # (num_nodes, 10)
        node_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.int32)
        
        # (num_nodes, num_nodes, 5)
        bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), 5), dtype=np.int32)
        
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            bond_features[begin_idx, end_idx, :] = bond_features[end_idx, begin_idx, :] = get_bond_features(bond)
        
        # add all, re-smiled for keeping the consistent type of smile string.
        smile = Chem.MolToSmiles(mol)
        smiles_list.append(smile)
        mols_dict[smile] = mol
        graphs_dict[smile] = (node_features, bond_features, distance_matrix)
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
    build_graph(smiles, labels, result_path, mode="bbbp")
    
    # clintox
    data = pd.read_csv(f'{data_path}/clintox.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,1:].values.tolist()
    build_graph(smiles, labels, result_path, mode="clintox")
    
    # esol
    data = pd.read_csv(f'{data_path}/esol.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['measured log solubility in mols per litre'].values.tolist()
    build_graph(smiles, labels, result_path, mode="esol")
    
    # freesolv
    data = pd.read_csv(f'{data_path}/freesolv.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['expt'].values.tolist()
    build_graph(smiles, labels, result_path, mode="freesolv")
    
    # lipophilicity
    data = pd.read_csv(f'{data_path}/lipophilicity.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data['exp'].values.tolist()
    build_graph(smiles, labels, result_path, mode="lipophilicity")
    
    # sider
    data = pd.read_csv(f'{data_path}/sider.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,1:].values.tolist()
    build_graph(smiles, labels, result_path, mode="sider")
    
    # tox21
    data = pd.read_csv(f'{data_path}/tox21.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,0:-2].values.tolist()
    build_graph(smiles, labels, result_path, mode="tox21")
    
    # toxcast
    data = pd.read_csv(f'{data_path}/toxcast.csv',sep=',')
    data = data.fillna(-1.0)
    smiles = data['smiles'].values.tolist()
    labels = data.iloc[:,1:].values.tolist()
    build_graph(smiles, labels, result_path, mode="toxcast")

