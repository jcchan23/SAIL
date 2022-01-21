######################################## import area ########################################

# common library
import torch
import dgl
import csv
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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


def normalize_dis(mx):
    result = 2 / (1 + np.maximum(mx/4, 1))
    result[np.isnan(result)] = 0
    result[np.isinf(result)] = 0
    return result


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin=0, dmax=5, step=0.1, var=0.5):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distances: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


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
        return max([graph.ndata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs]), \
            max([graph.edata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs])

    
######################################## main area ########################################

if __name__ == "__main__":
    # build the dgl graph cache
    data_path = './data/source'
    result_path = './data/preprocess'
    dataset_dict = {'spin2':['train', 'test', 'casp'], 'densecpd':['train', 't500', 'ts50']}
    cutoff = 7.5
    
    for dataset_name in ['spin2', 'densecpd']:
        for mode in dataset_dict[dataset_name]:
            print(f'build {dataset_name} {mode} dgl graph')
            
            names_list = [data[0] for data in csv.reader(open(f'{data_path}/{dataset_name}/{mode}_list.txt', 'r'))]
            
            sequences_dict, graphs_dict, labels_dict = dict(), dict(), dict()
            
            mean_array = np.load(f'{result_path}/{dataset_name}/train_mean.npy')
            std_array = np.load(f'{result_path}/{dataset_name}/train_std.npy')
            
            for name in tqdm(names_list):
                # sequence
                with open(f'{result_path}/{dataset_name}/features/fasta/{name}.fasta', 'r') as f:
                    lines = f.readlines()
                sequences_dict[name] = lines[1].strip()
                
                # label
                labels_dict[name] = np.load(f'{result_path}/{dataset_name}/features/label/{name}.npy')
                
                # [L, 264]
                node_features = np.load(f'{result_path}/{dataset_name}/features/node_features/{name}.npy')
                node_features = (node_features - mean_array) / std_array
                
                # [L, L]   
                distance_map = np.load(f'{result_path}/{dataset_name}/features/edge_features/{name}.npy')
                 # find 0 <= ditance <= cutoff rows and columns
                distance_map = np.where((distance_map >= 0) & (distance_map <= cutoff), distance_map, float('inf'))
                distance_weight = normalize_dis(distance_map)
                distance_weight = distance_weight / (np.sum(distance_weight, axis=-1, keepdims=True) + 1e-5)
                
                # [L, L] -> [num_edges, 1]
                edge_features = distance_weight[np.nonzero(distance_weight)].reshape(-1, 1)
                
                # build dgl graph
                src, dst = np.nonzero(distance_weight)
                graph = dgl.graph((src, dst), num_nodes=node_features.shape[0])
                
                if not len(sequences_dict[name]) == len(labels_dict[name]) == node_features.shape[0] == distance_weight.shape[0] == distance_weight.shape[1]:
                    print(f"{dataset_name} {name} sequence, label, node features, distance map error!")
                    assert False
                if not len(edge_features) == len(src) == len(dst):
                    print(f"{dataset_name} {name} edge features error!")
                    assert False
                
                # add features
                graph.ndata['x'] = torch.from_numpy(node_features).float()
                graph.edata['x'] = torch.from_numpy(edge_features).float()
                graphs_dict[name] = graph
            
            # save graphs
            with open(f'{result_path}/{dataset_name}/{mode}.pickle', 'wb') as pw:
                pickle.dump([names_list, sequences_dict, graphs_dict, labels_dict], pw)
    