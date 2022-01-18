######################################## import area ########################################

# common library
import dgl
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
    idx, masks = 0, list()
    for graph in graphs:
        masks.append([idx, graph.num_nodes()])
        idx += graph.num_nodes()
    return names, sequences, dgl.batch(graphs), torch.from_numpy(np.concatenate(labels, axis=-1).astype(np.int64)), torch.from_numpy(np.array(masks, dtype=np.int64))


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
    
    # construct Btest_31.fa
    names_list, sequences_dict, labels_dict = load_dataset(f'{data_path}/Test_60.fa')
    with open(f'{data_path}/bound_unbound_mapping.txt', 'r') as f, open(f'{data_path}/Btest_31.fa', 'w') as fw:
        lines = f.readlines()[1:]
        for line in lines:
            bound_ID, unbound_ID, _ = line.strip().split()
            sequence, labels = sequences_dict[bound_ID], labels_dict[bound_ID]
            fw.write(f'>{bound_ID}\n')
            fw.write(sequence + '\n')
            fw.write("".join([str(num) for num in labels]) + '\n')

    for dataset_name in ['Test_60', 'Test_315', 'Train_335', 'Btest_31', 'UBtest_31']:
        print(f'build {dataset_name} dgl graph')
        cut_off = 14
        names_list, sequences_dict, labels_dict = load_dataset(f'{data_path}/{dataset_name}.fa')

        graphs_dict = dict()

        for name in tqdm(names_list):
            sequence, label = sequences_dict[name], labels_dict[name]
            pssm_features = np.load(f'{result_path}/features/pssm/{name}.npy')
            hmm_features = np.load(f'{result_path}/features/hmm/{name}.npy')
            dssp_features = np.load(f'{result_path}/features/dssp/{name}.npy')
            # [L, 20 + 20 + 14 = 54]
            node_features = np.concatenate([pssm_features, hmm_features, dssp_features], axis=-1)

            # [L, L]
            distance_map = np.load(f'{result_path}/features/distance_map/{name}.npy')
            # [L, L]
            mask = ((distance_map >= 0) * (distance_map < cut_off))
            edge_features = normalize_adj(mask.astype(np.float))
            edge_features = edge_features[np.nonzero(edge_features)]

            # build dgl graph
            src, dst = np.nonzero(mask.astype(np.int))
            graph = dgl.graph((src, dst), num_nodes=len(sequence))
            
            if not len(sequence) == len(label) == node_features.shape[0]:
                print(f"{dataset_name} {name} sequence, label, node features error!")
                assert False
            if not len(edge_features) == len(src) == len(dst):
                print(f"{dataset_name} {name} edge features error!")
                assert False

            # only node features
            graph.ndata['x'] = torch.from_numpy(node_features).float()
            graph.edata['x'] = torch.from_numpy(edge_features).float()
            graphs_dict[name] = graph

        # save graphs
        with open(f'{result_path}/{dataset_name}.pickle', 'wb') as fw:
            pickle.dump([names_list, sequences_dict, graphs_dict, labels_dict], fw)
