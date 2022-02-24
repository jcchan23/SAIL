######################################## import area ########################################

# common library
import os
import numpy as np
from tqdm import tqdm
from Bio import pairwise2

######################################## function area ########################################

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


def cal_mean_std(data_path, names, length=384):
    total_length = 0
    mean = np.zeros(length)
    mean_square = np.zeros(length)
    for name in tqdm(names):
        matrix = np.load(f'{data_path}/features/af2_node_features/{name}.npy')
        total_length += matrix.shape[0]
        mean += np.sum(matrix, axis=0)
        mean_square += np.sum(np.square(matrix), axis=0)
    # EX
    mean /= total_length
    # E(X^2)
    mean_square /= total_length
    # DX = E(X^2) - (EX)^2, std = sqrt(DX)
    std = np.sqrt(mean_square - np.square(mean))
    return mean, std


def cal_min_max(data_path, names, length=384):
    min_array = np.array([float('inf') for _ in range(length)])
    max_array = np.array([float('-inf') for _ in range(length)])
    for name in tqdm(names):
        matrix = np.load(f'{data_path}/features/af2_node_features/{name}.npy')
        min_matrix = np.min(matrix, axis=0)
        max_matrix = np.max(matrix, axis=0)
        assert len(min_matrix) == len(max_matrix) == length
        for i in range(length):
            if min_matrix[i] < min_array[i]:
                min_array[i] = min_matrix[i]
            if max_matrix[i] > max_array[i]:
                max_array[i] = max_matrix[i]    
    return min_array, max_array


if __name__ == "__main__":
    
    ######################################## generate fasta file ########################################
    
    # run on the desktop
    data_path = './source'
    result_path = './preprocess'
    
    name_map = {'DNA_Test_129.fa':'test', 'DNA_Test_181.fa':'test', 'DNA_Train_573.fa':'train'}
    
    for src_name, dst_name in name_map.items():
        src_names_list, src_sequences_dict, src_labels_dict = load_dataset(f'{data_path}/{src_name}')
        dst_names_list, dst_sequences_dict, dst_labels_dict = list(), dict(), dict()
        
        for name in tqdm(src_names_list):
            if os.path.exists(f'{data_path}/features/af2_pdb/{name}.pdb') and \
                os.path.exists(f'{data_path}/features/af2_distance/{name}.distance') and \
                os.path.exists(f'{result_path}/features/af2_node_features/{name}.npy'):
                    dst_names_list.append(name)
                    dst_sequences_dict[name] = src_sequences_dict[name]
                    dst_labels_dict[name] = src_labels_dict[name]
        
        with open(f'{data_path}/{dst_name}_{len(dst_names_list)}.fasta', 'w') as fw:
            for name in dst_names_list:
                sequence, label = dst_sequences_dict[name], "".join([str(num) for num in dst_labels_dict[name]])
                fw.write(f'>{name}\n')
                fw.write(sequence + '\n')
                fw.write(label + '\n')

    ######################################## generate npy features ########################################
    
    # run on the desktop
    data_path = './source'
    result_path = './preprocess'
    
    for name in ['train_569', 'test_181', 'test_129']:
        names_list, sequences_dict, labels_dict = load_dataset(f'{data_path}/{name}.fasta')
        print(f'generate {name} edge features')
        for name in tqdm(names_list):
            sequence, label = sequences_dict[name], labels_dict[name]
            align_sequence, result = distance_map_process(f'{data_path}/features/af2_distance/{name}.distance', sequence)
            if not len(align_sequence) == result.shape[0] == result.shape[1]:
                print(name)
                assert False
            np.save(f'{result_path}/features/af2_edge_features/{name}.npy', result)
