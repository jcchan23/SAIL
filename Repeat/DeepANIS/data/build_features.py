#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/01/27 09:14:07
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import PDB
from Bio import SeqIO
from Bio import pairwise2
from Bio.PDB import Selection, NeighborSearch

######################################## function area ########################################

amino2id = {
    'U': 0, 'X': 21,
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
    'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 
}

id2amino = {value:key for key, value in amino2id.items()}

amino3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}


def fasta_process(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    return lines[1].strip()


def pssm_process(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    
    if os.path.exists(fname) == 0:
        print(fname)
        return np.zeros((120,20))
    
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).dropna().values[:,2:22].astype(float)
        
    if tmp_pssm.shape[0] != len(seq):
        print(tmp_pssm.shape[0], len(seq))
        raise ValueError('PSSM file is in wrong format or incorrect!')
        
    return tmp_pssm


def hmm_process(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    
    if os.path.exists(fname) == 0:
        print(fname)
        return np.zeros((120,30))
    
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        print(hhm.shape[0], len(seq))
        raise ValueError('HHM file is in wrong format or incorrect!')
        
    return hhm[:,2:-12].astype(float)


def spd3_feature_sincos(x,seq):
    ASA = x[:,0]
    rnam1_std = "ACDEFGHIKLMNPQRSTVWYX"
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                        185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    ASA_div =  np.array([dict_rnam1_ASA[i] for i in seq])
    ASA = (ASA/ASA_div)[:,None]
    angles = x[:,1:5]
    HSEa = x[:,5:7]
    HCEprob = x[:,-3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles),np.cos(angles)],1)
    return np.concatenate([ASA,angles,HSEa,HCEprob],1)


def spd33_process(fname,seq):
    if os.path.exists(fname) == 0:
        print(fname)
        return np.zeros((120,14))
    
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True).values[:,3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq)
    if tmp_spd3.shape[0] != len(seq):
        print(tmp_spd3.shape[0], len(seq))
        raise ValueError('SPD3 file is in wrong format or incorrect!')
    return tmp_spd3


def label_process(fname, seq):
    with open(fname, 'r') as f:
        lines = f.readlines()
    sequence, label = lines[1].strip(), lines[2].strip()
    if not len(seq) == len(sequence) == len(label):
        print(fname)
        print(seq)
        print(sequence)
        print(label)
        assert False
    return [int(num) for num in label]

######################################## main area ########################################

if __name__ == '__main__':
    
    # warning
    warnings.filterwarnings("ignore")
    data_path = './source'
    result_path = './preprocess'
    
    data = pd.read_csv(f'{data_path}/dataset.csv', sep=',')
    data_pdb, data_H, data_L, data_A = data['pdb'].values, data['Hchain'].values, data['Lchain'].values, data['antigen_chain'].values
    # refernce: https://github.com/eliberis/parapred
    # H1: 24-34; H2: 50-56; H3: 89-97 ; 11 + 7 + 9 + (4 * 3) = 39
    # L1: 26-32; L2: 52-56; L3: 95-102; 7 + 5 + 8 + (4 * 3) = 32
    extra_index, distance = 2, 4.5
    cdr_ranges = {'H1':[-extra_index + 24, 34 + extra_index], 'H2':[-extra_index + 50, 56 + extra_index], 'H3':[-extra_index + 89, 97 + extra_index],
                  'L1':[-extra_index + 26, 32 + extra_index], 'L2':[-extra_index + 52, 56 + extra_index], 'L3':[-extra_index + 95, 102 + extra_index]}
    
    ######################################## 1. from pdb to chain ########################################
    for pdb_id, H_id, L_id, A_id in tqdm(zip(data_pdb, data_H, data_L, data_A), total=len(data_pdb)):
        
        # prepared pdb
        model = PDB.PDBParser(PERMISSIVE=1, QUIET=1).get_structure(pdb_id, f'{data_path}/features/pdb/{pdb_id}.pdb')[0]
        
        # antigen chain maybe multichain
        A_id = [A_id] if len(A_id) == 1 else A_id.split(' | ')
        
        # save chain
        for chain_id in [H_id, L_id] + A_id:
            io = PDB.PDBIO()
            io.set_structure(model[chain_id])
            if os.path.exists(f'{data_path}/features/chain/{pdb_id}{chain_id}.pdb'):
                continue
            io.save(f'{data_path}/features/chain/{pdb_id}{chain_id}.pdb')

    ######################################## 2. from chain to fasta ########################################
    for pdb_id, H_id, L_id in tqdm(zip(data_pdb, data_H, data_L), total=len(data_pdb)):
        
        for record in SeqIO.parse(f'{data_path}/features/chain/{pdb_id}{H_id}.pdb', 'pdb-atom'):
            if f'????:{H_id}' == record.id or H_id == record.id or H_id == record.id[-1]:
                with open(f'{data_path}/features/fasta/{pdb_id}{H_id}.fasta','w') as fw:
                    fw.write(f'>{pdb_id}{H_id}\n')
                    fw.write("".join(list(record.seq)) + '\n')
        
        for record in SeqIO.parse(f'{data_path}/features/chain/{pdb_id}{L_id}.pdb', 'pdb-atom'):
            if f'????:{L_id}' == record.id or L_id == record.id or L_id == record.id[-1]:
                with open(f'{data_path}/features/fasta/{pdb_id}{L_id}.fasta','w') as fw:
                    fw.write(f'>{pdb_id}{L_id}\n')
                    fw.write("".join(list(record.seq)) + '\n')
    
    ####################################### 3. from fasta to check pssm, hmm and spd33 ########################################
    for name in tqdm(os.listdir(f'{data_path}/features/fasta')):
        name = name.split('.')[0]
        pdb_id, chain_id = name[:-1], name[-1]
        
        sequence = fasta_process(f'{data_path}/features/fasta/{name}.fasta')
        
        if not os.path.exists(f'{data_path}/features/pssm/{name}.pssm') \
            and os.path.exists(f'{data_path}/features/hmm/{name}.hhm') \
            and os.path.exists(f'{data_path}/features/spd33/{name}.spd33'):
                print(name)
                continue
            
        pssm = pssm_process(f'{data_path}/features/pssm/{name}.pssm', sequence)
        hmm = hmm_process(f'{data_path}/features/hmm/{name}.hhm', sequence)
        spd33 = spd33_process(f'{data_path}/features/spd33/{name}.spd33', sequence)
        
        if not len(sequence) == pssm.shape[0] == hmm.shape[0] == spd33.shape[0]:
            print(name)
            continue

    ######################################## 4. construct label ########################################
    for pdb_id, H_id, L_id, A_id in tqdm(zip(data_pdb, data_H, data_L, data_A), total=len(data_pdb)):
        
        # prepared pdb
        model = PDB.PDBParser(PERMISSIVE=1, QUIET=1).get_structure(pdb_id, f'{data_path}/features/pdb/{pdb_id}.pdb')[0]
        
        # get antigen chain
        if '|' in A_id:
            A_ids = A_id.split(' | ')
            A_atoms = [a for c in A_ids for a in Selection.unfold_entities(model[c], target_level='A')]
            A_chain = None
        else:
            A_atoms = Selection.unfold_entities(model[A_id], target_level='A')
            A_chain = model[A_id]
        A_search = NeighborSearch(A_atoms)
        
        # H_chain
        H_chain = model[H_id]
        # collect label
        H_temp_label = list()
        for residue in H_chain:
            if residue.get_resname() in amino3to1.keys():
                if any(len(A_search.search(a.coord, distance)) > 0 for a in residue.get_unpacked_list()):
                    H_temp_label.append("1")
                else:
                    H_temp_label.append("0")
        # alignment label
        H_label = list()
        H_Chain_fasta = fasta_process(f'{data_path}/features/fasta/{pdb_id}{H_id}.fasta')
        sequence = "".join([amino3to1[residue.get_resname()] for residue in H_chain if residue.get_resname() in amino3to1.keys()])
        if sequence == H_Chain_fasta:
            H_label = H_temp_label
        else:
            alignments = pairwise2.align.globalxx(H_Chain_fasta, sequence)
            standard_fasta = alignments[0].seqA
            align_sequence = alignments[0].seqB
            for idx, aa in enumerate(standard_fasta):
                if aa == align_sequence[idx]:
                    try:
                        cls = H_temp_label.pop(0)
                    except:
                        print(idx, aa)
                        print(standard_fasta)
                        print(align_sequence)
                        assert False
                    H_label.append(cls)
                else:
                    H_label.append("0")
        assert len(H_label) == len(H_Chain_fasta)
        # write file
        with open(f'{data_path}/features/label/{pdb_id}{H_id}.txt', 'w') as fw:
            fw.write(f'>{pdb_id}{H_id}\n')
            fw.write(H_Chain_fasta + '\n')
            fw.write("".join(H_label) + '\n')
        
        # L_chain
        L_chain = model[L_id]
        # collect label
        L_temp_label = list()
        for residue in L_chain:
            if residue.get_resname() in amino3to1.keys():
                if any(len(A_search.search(a.coord, distance)) > 0 for a in residue.get_unpacked_list()):
                    L_temp_label.append("1")
                else:
                    L_temp_label.append("0")
        # alignment label
        L_label = list()
        L_Chain_fasta = fasta_process(f'{data_path}/features/fasta/{pdb_id}{L_id}.fasta')
        sequence = "".join([amino3to1[residue.get_resname()] for residue in L_chain if residue.get_resname() in amino3to1.keys()])
        if sequence == L_Chain_fasta:
            L_label = L_temp_label
        else:
            alignments = pairwise2.align.globalxx(L_Chain_fasta, sequence)
            standard_fasta = alignments[0].seqA
            align_sequence = alignments[0].seqB
            for idx, aa in enumerate(standard_fasta):
                if aa == align_sequence[idx]:
                    try:
                        cls = L_temp_label.pop(0)
                    except:
                        print(idx, aa)
                        print(standard_fasta)
                        print(align_sequence)
                        assert False
                    L_label.append(cls)
                else:
                    L_label.append("0")
        assert len(L_label) == len(L_Chain_fasta)
        # write file
        with open(f'{data_path}/features/label/{pdb_id}{L_id}.txt', 'w') as fw:
            fw.write(f'>{pdb_id}{L_id}\n')
            fw.write(L_Chain_fasta + '\n')
            fw.write("".join(L_label) + '\n')
        
    ######################################## 5. construct concatenate cdrs ########################################
    
    with open(f'{data_path}/total_277.fasta', 'w') as fw:
        
        for pdb_id, H_id, L_id in tqdm(zip(data_pdb, data_H, data_L), total=len(data_pdb)):

            H_Chain_fasta = fasta_process(f'{data_path}/features/fasta/{pdb_id}{H_id}.fasta')
            H_Chain_pssm = pssm_process(f'{data_path}/features/pssm/{pdb_id}{H_id}.pssm', H_Chain_fasta)
            H_Chain_hmm = hmm_process(f'{data_path}/features/hmm/{pdb_id}{H_id}.hhm', H_Chain_fasta)
            H_Chain_spd33 = spd33_process(f'{data_path}/features/spd33/{pdb_id}{H_id}.spd33', H_Chain_fasta)
            H_Chain_label = label_process(f'{data_path}/features/label/{pdb_id}{H_id}.txt', H_Chain_fasta)
            
            L_Chain_fasta = fasta_process(f'{data_path}/features/fasta/{pdb_id}{L_id}.fasta')
            L_Chain_pssm = pssm_process(f'{data_path}/features/pssm/{pdb_id}{L_id}.pssm', L_Chain_fasta)
            L_Chain_hmm = hmm_process(f'{data_path}/features/hmm/{pdb_id}{L_id}.hhm', L_Chain_fasta)
            L_Chain_spd33 = spd33_process(f'{data_path}/features/spd33/{pdb_id}{L_id}.spd33', L_Chain_fasta)
            L_Chain_label = label_process(f'{data_path}/features/label/{pdb_id}{L_id}.txt', L_Chain_fasta)
            
            # construct concatenate label
            cdrs, pssms, hmms, spd33s, labels = list(), list(), list(), list(), list()
            
            for cdr in ['H1', 'H2', 'H3']:
                start, end = cdr_ranges[cdr]
                try:
                    fasta = H_Chain_fasta[start:end]
                    pssm  = H_Chain_pssm[start:end]
                    hmm   = H_Chain_hmm[start:end]
                    spd33 = H_Chain_spd33[start:end]
                    label = H_Chain_label[start:end]
                except:
                    print(f'{pdb_id}{H_id}')
                    assert False
                
                # H1 U H2 U H3 U
                cdrs.extend([amino2id[amino] for amino in fasta] + [amino2id['U']])
                pssms.extend(np.vstack([pssm, np.array([0] * pssm.shape[-1])]))
                hmms.extend(np.vstack([hmm, np.array([0] * hmm.shape[-1])]))
                spd33s.extend(np.vstack([spd33, np.array([0] * spd33.shape[-1])]))
                labels.extend(label + [0])
                
            for cdr in ['L1', 'L2', 'L3']:
                try:
                    fasta = L_Chain_fasta[start:end]
                    pssm  = L_Chain_pssm[start:end]
                    hmm   = L_Chain_hmm[start:end]
                    spd33 = L_Chain_spd33[start:end]
                    label = L_Chain_label[start:end]
                except:
                    print(f'{pdb_id}{L_id}')
                    assert False
                
                # H1 U H2 U H3 U L1 U L2 U L3
                if cdr != 'L3':
                    cdrs.extend([amino2id[amino] for amino in fasta] + [amino2id['U']])
                    pssms.extend(np.vstack([pssm, np.array([0] * pssm.shape[-1])]))
                    hmms.extend(np.vstack([hmm, np.array([0] * hmm.shape[-1])]))
                    spd33s.extend(np.vstack([spd33,  np.array([0] * spd33.shape[-1])]))
                    labels.extend(label + [0])
                else:
                    cdrs.extend([amino2id[amino] for amino in fasta])
                    pssms.extend(pssm)
                    hmms.extend(hmm)
                    spd33s.extend(spd33)
                    labels.extend(label)

            cdrs, pssms, hmms, spd33s, labels = np.array(cdrs), np.array(pssms), np.array(hmms), np.array(spd33s), np.array(labels)
            node_features = np.concatenate([pssms, hmms, spd33s], axis=-1)
            
            if not len(cdrs) == node_features.shape[0] == len(labels):
                print(len(cdrs), node_features.shape[0], len(labels))
                assert False
            
            np.save(f'{result_path}/features/embedding/{pdb_id}{H_id}{L_id}.npy', cdrs)
            np.save(f'{result_path}/features/node_features/{pdb_id}{H_id}{L_id}.npy', node_features)
            np.save(f'{result_path}/features/label/{pdb_id}{H_id}{L_id}.npy', labels)
            
            fw.write(f'>{pdb_id}{H_id}{L_id}\n')
            fw.write("".join([id2amino[num] for num in cdrs]) + '\n')
            fw.write("".join([str(int(num)) for num in labels]) + '\n')
            
    