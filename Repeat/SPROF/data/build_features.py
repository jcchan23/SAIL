#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_features.py
@Time    :   2022/01/22 15:46:04
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import csv
import warnings
import numpy as np
from tqdm import tqdm
from Bio import PDB
from Bio import SeqIO
from Bio import pairwise2
from biopandas.pdb import PandasPdb

######################################## function area ########################################

amino2id = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

amino_3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def fasta_process(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    return lines[1].strip()


def dssp_process(dssp_file, standard_seq):
    with open(dssp_file, "r") as f:
        lines = f.readlines()

    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    # dssp from # to start the table
    p = 0
    while lines[p].strip()[0] != "#":
        p += 1

    seq, result = "", list()
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*" or aa == "X":
            continue
        seq += aa

        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        # The last dim represents "Unknown" for missing residues
        SS_vec = np.zeros(9)
        SS_vec[SS_type.find(SS)] = 1
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        result.append(np.concatenate((np.array([aa_type.find(aa)]), SS_vec, np.array([ASA]))))

    if standard_seq != seq:
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_seq_align = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = []
        for idx, aa in enumerate(standard_seq):
            if aa == align_seq[idx]:
                # pop the first element in the list
                feature_row = result.pop(0)
                # double-check the map between amino to features
                if feature_row[0] != aa_type.find(aa):
                    raise "the amino map incorrectly!"
                else:
                    align_result.append(feature_row[1:])
            else:
                # 9 dim of SS_vec, 1 dim of ASA
                align_result.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1] + [1]))

        # L * 10
        return align_seq, np.array(align_result)
    else:
        # L * 11 -> L * 10, delete the amino column
        return seq, np.array(result)[:, 1:]


def ppo_process(ppo_file, standard_seq):
    with open(ppo_file, 'r') as f:
        lines = f.readlines()[2:]

    seq, result = "", list()
    for line in lines:
        line = line.strip().split(' ')
        seq += line[1]
        if len(line[2:]) != 3:
            raise "the number of angles incorrect!"
        PHI, PSI, OMEGA = [np.deg2rad(float(data)) for data in line[2:]]
        result.append([amino2id[line[1]], np.sin(PHI), np.cos(PHI), np.sin(PSI), np.cos(PSI), np.sin(OMEGA), np.cos(OMEGA)])

    if len(standard_seq) != seq:
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_align_seq = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = []
        for idx, aa in enumerate(standard_seq):
            if aa == align_seq[idx]:
                # pop the first element in the list
                feature_row = result.pop(0)
                # double-check the map between amino to features
                if feature_row[0] != amino2id[aa]:
                    raise "the amino map incorrectly!"
                else:
                    align_result.append(feature_row[1:])
            else:
                # np.sin(2π)=0.0, np.cos(2π)=1.0
                align_result.append([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        # L * 6
        return align_seq, np.array(align_result)
    else:
        # L * 7 -> L * 6, delete the amino column
        return seq, np.array(result)[:, 1:]


def theta_process(theta_file, standard_seq):
    with open(theta_file, 'r') as f:
        lines = f.readlines()[1:]

    seq, result = "", list()
    for line in lines:
        line = line.strip().split(' ')
        seq += line[1]
        if len(line[2:]) != 3:
            raise "the number of angles incorrect!"
        THETA1, THETA2 = [np.deg2rad(float(data)) for data in line[2:4]]
        result.append([amino2id[line[1]], np.sin(THETA1), np.cos(THETA1), np.sin(THETA2), np.cos(THETA2)])

    if len(standard_seq) != len(seq):
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_align_seq = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = []
        for idx, aa in enumerate(standard_seq):
            if aa == align_seq[idx]:
                # pop the first element in the list
                feature_row = result.pop(0)
                # double-check the map between amino to features
                if feature_row[0] != amino2id[aa]:
                    raise "the amino map incorrectly!"
                else:
                    align_result.append(feature_row[1:])
            else:
                # np.sin(2π)=0.0, np.cos(2π)=1.0
                align_result.append([0.0, 1.0, 0.0, 1.0])
        # L * 4
        return align_seq, np.array(align_result)
    else:
        # L * 5 -> L * 4, delete the amino column
        return seq, np.array(result)[:, 1:]


def dfire_process(dfire_file, standard_seq):
    with open(dfire_file, 'r') as f:
        lines = f.readlines()[1:]
    seq, result = "", list()
    for line in lines:
        line = line.strip().split(' ')
        seq += line[1]
        result.append([amino2id[line[1]]] + [float(data) for data in line[2:]])

    if len(standard_seq) != len(seq):
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_align_seq = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = []
        for idx, aa in enumerate(standard_seq):
            if aa == align_seq[idx]:
                # pop the first element in the list
                feature_row = result.pop(0)
                # double-check the map between amino to features
                if feature_row[0] != amino2id[aa]:
                    raise "the amino map incorrectly!"
                else:
                    align_result.append(feature_row[1:])
            else:
                align_result.append([0.0 for _ in range(112)])
        # L * 112
        return align_seq, np.array(align_result)
    else:
        # L * 113 -> L * 112, delete the amino column
        return seq, np.array(result)[:, 1:]


def fragroto_pdb_process(pdb_file):
    ppdb = PandasPdb()
    result = ppdb.read_pdb(pdb_file)
    result = result.df['ATOM']
    amino_list, amino_idxs = list(result['residue_name'].values), list(result['residue_number'].values)
    assert len(amino_list) == len(amino_idxs)

    seq_result = list()
    cur_amino, cur_idx = "", 0
    while len(amino_list) == len(amino_idxs) and len(amino_list) != 0:
        temp_amino, temp_idx = amino_list.pop(0), int(amino_idxs.pop(0))
        if temp_idx != cur_idx:
            cur_amino, cur_idx = temp_amino, temp_idx
            seq_result.append(amino_3to1[cur_amino])

    return "".join(seq_result)


def fragment_process(fragment_file, standard_seq, seq):
    with open(fragment_file, 'r') as f:
        lines = f.readlines()
    if len(lines) != len(seq):
        print(fragment_file + "fragment files not match with fragment pdb, please double check")

    if len(standard_seq) != len(seq):
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_align_seq = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = []
        for idx, aa in enumerate(standard_seq):
            if aa == align_seq[idx]:
                try:
                    features_row = lines.pop(0)
                except IndexError:
                    print(fragment_file)
                    print(idx, aa)
                    print(standard_seq)
                    print(align_seq)
                    assert False
                features_row = features_row.strip().split()
                align_result.append([float(num) for num in features_row])
            else:
                align_result.append([0.0 for _ in range(20)])
        return align_seq, np.array(align_result)
    else:
        result = []
        for line in lines:
            line = line.strip().split()
            result.append([float(num) for num in line])
        return seq, np.array(result)


def rotomer_process(rotomer_file, standard_seq, seq):
    with open(rotomer_file, 'r') as f:
        lines = f.readlines()
    if len(lines) != len(seq):
        print(rotomer_file + "rotomer files not match with fragment pdb, please double check")

    if len(standard_seq) != len(seq):
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_align_seq = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = []
        for idx, aa in enumerate(standard_seq):
            if aa == align_seq[idx]:
                try:
                    features_row = lines.pop(0)
                except IndexError:
                    print(rotomer_file)
                    print(idx, aa)
                    print(standard_seq)
                    print(align_seq)
                    assert False
                features_row = features_row.strip().split()
                align_result.append([float(num) for num in features_row])
            else:
                align_result.append([0.0 for _ in range(112)])
        return align_seq, np.array(align_result)
    else:
        result = []
        for line in lines:
            line = line.strip().split()
            result.append([float(num) for num in line])
        return seq, np.array(result)


def distance_map_process(distance_map_file, standard_seq):
    with open(distance_map_file, 'r') as f:
        lines = f.readlines()
    seq = lines[0].strip()
    if len(seq) != len(lines[1:]) + 1:
        raise "The protein sequence incorrect!"

    if len(standard_seq) != len(seq):
        alignments = pairwise2.align.globalxx(standard_seq, seq)
        standard_align_seq = alignments[0].seqA
        align_seq = alignments[0].seqB

        # use standard_seq as template
        align_result = np.ones((len(standard_seq), len(standard_seq))) * float('inf')
        features = [["0.0"]] + [line.strip().split(' ') + ["0.0"] for line in lines[1:]]
        inf_idxs = list()
        for row, aa in enumerate(standard_seq):
            if aa == align_seq[row]:
                try:
                    features_row = features.pop(0)
                except IndexError:
                    print(distance_map_file)
                    print(row)
                    print(standard_seq)
                    print(align_seq)
                    assert False
                for inf_idx in inf_idxs:
                    features_row.insert(inf_idx, 'inf')
                for column, distance in enumerate(features_row):
                    align_result[row][column] = float(distance)
                    align_result[column][row] = float(distance)
            else:
                inf_idxs.append(row)
        return align_seq, align_result
    else:
        # the file record distance only below the diagonal
        result = np.zeros((len(seq), len(seq)), dtype=np.float)
        for row, line in enumerate(lines[1:], start=1):
            line = line.strip().split(' ')
            for column, distance in enumerate(line):
                result[row][column] = float(distance)
                result[column][row] = float(distance)
        return seq, result


def target_process(seq):
    result = list()
    for s in seq:
        if s not in amino2id.keys():
            result.append(len(amino2id))
        else:
            result.append(amino2id[s])
    return np.array(result)


def cal_mean_std(names_list, data_path, length=264):
    mean_array = np.zeros(length)
    mean_square_array = np.zeros(length)
    total_length = 0

    for name in tqdm(names_list):
        matrix = np.load(f'{data_path}/{name}.npy')
        total_length += matrix.shape[0]
        mean_array += np.sum(matrix, axis=0)
        mean_square_array += np.sum(np.square(matrix), axis=0)

    # EX
    mean_array /= total_length
    # E(X^2)
    mean_square_array /= total_length
    # DX = E(X^2)-(EX)^2, std=sqrt(DX)
    std_array = np.sqrt(np.subtract(mean_square_array, np.square(mean_array)))

    print(mean_array.shape, std_array.shape)
    print((True in np.isnan(mean_array)), (True in np.isnan(std_array)))
    return mean_array, std_array


######################################## main area ########################################

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    modes_dict = {'densecpd':['train', 't500', 'ts50'], 'spin2': ['train', 'test', 'casp']}
    data_path = './source'
    result_path = './preprocess'
    
    for dataset_name in ['densecpd', 'spin2']:
        modes_list = modes_dict[dataset_name]
        
        for mode in modes_list:
            names_list = [data[0] for data in csv.reader(open(f'{data_path}/{dataset_name}/{mode}_list.txt', 'r'))]
            
            # 1. from pdb to chain
            for name in tqdm(names_list):
                pdb_id, chain_id = name[:-1], name[0][-1]
                
                # download pdb or use local pdb
                PDB.PDBList(verbose=True).retrieve_pdb_file(pdb_id, pdir=f'{data_path}/{dataset_name}/features/pdb', file_format='pdb')
                # prepared pdb
                model = PDB.PDBParser(PERMISSIVE=1, QUIET=1).get_structure(pdb_id, f'{data_path}/{dataset_name}/features/pdb/pdb{pdb_id.lower()}.ent')[0]
                
                # save chain
                io = PDB.PDBIO()
                io.set_structure(model[chain_id])
                io.save(f'{data_path}/{dataset_name}/features/chain/{pdb_id}{chain_id}.pdb')
            
            
            # 2. from chain to fasta
            for name in tqdm(names_list):
                pdb_id, chain_id = name[:-1], name[0][-1]
                for record in SeqIO.parse(f'{data_path}/{dataset_name}/features/chain/{pdb_id}.pdb', 'pdb-atom'):
                    if f'{pdb_id.upper()}:{chain_id}' == record.id or chain_id == record.id or chain_id == record.id[-1]:
                        with open(f'{result_path}/{dataset_name}/features/fasta/{pdb_id}{chain_id}.fasta', 'w') as fw:
                            fw.write(f'>{pdb_id}{chain_id}\n')
                            fw.write("".join(list(record.seq)) + '\n')
            
            
            # 3. from fasta to all features
            for name in tqdm(names_list):
                # read fasta
                seq_fasta = fasta_process(f'{result_path}/{dataset_name}/features/fasta/{name}.fasta')
                
                # L * 10 dssp features
                seq_dssp, dssp = dssp_process(f'{data_path}/{dataset_name}/features/dssp/{name}.dssp', seq_fasta)
                
                # L * 6 ppo features
                seq_ppo, ppo = ppo_process(f'{data_path}/{dataset_name}/features/ppo/{name}.ppo', seq_fasta)
                
                # L * 4 theta features
                seq_theta, theta = theta_process(f'{data_path}/{dataset_name}/features/theta/{name}.theta', seq_fasta)
                
                # L * 112 dfire features
                seq_dfire, dfire = dfire_process(f'{data_path}/{dataset_name}/features/dfire/{name}.dfire', seq_fasta)
                
                # read merge ala sequence in pdb
                seq_fragroto_pdb = fragroto_pdb_process(f'{data_path}/{dataset_name}/features/fragroto/{name}_fragroto.pdb')
                
                # L * 20 fragments features, aligned by dfire
                seq_fragment, fragment = fragment_process(f'{data_path}/{dataset_name}/features/fragments/{name}.fragments', seq_fasta, seq_fragroto_pdb)
                
                # L * 112 rotomers features, aligned by dfire
                seq_romoter, rotomer = rotomer_process(f'{data_path}/{dataset_name}/features/fragments/rotomers/{name}.rotomers', seq_fasta, seq_fragroto_pdb)
                
                # L * L
                seq_dist, edge_features = distance_map_process(f'{data_path}/{dataset_name}/features/fragments/distance/{name}.distance', seq_fasta)
                
                if not len(seq_fasta) == dssp.shape[0] == ppo.shape[0] == theta.shape[0] == dfire.shape[0] == fragment.shape[0] == rotomer.shape[0] == edge_features.shape[0]:
                    print(name, len(seq_fasta), dssp.shape[0], ppo.shape[0], theta.shape[0], dfire.shape[0], fragment.shape[0], rotomer.shape[0], edge_features.shape[0])
                    continue
                else:
                    # 10 + 6 + 4 + 112 + 20 + 112 = 264
                    node_features = np.concatenate([dssp, ppo, theta, dfire, fragment, rotomer], axis=1)
                    assert node_features.shape[1] == 264
                    # save features
                    np.save(f'{result_path}/{dataset_name}/features/node_features/{name}.npy', node_features)
                    np.save(f'{result_path}/{dataset_name}/features/edge_features/{name}.npy', edge_features)
                    
                    # L target result
                    label = target_process(seq_fasta)
                    np.save(f'{result_path}/{dataset_name}/features/label/{name}.npy', label)
    
            if mode == 'train':
                mean_array, std_array = cal_mean_std(names_list, f'{result_path}/{dataset_name}/features/node_features', length=264)
                np.save(f'{result_path}/{dataset_name}/train_mean.npy', mean_array)
                np.save(f'{result_path}/{dataset_name}/train_std.npy', std_array)


