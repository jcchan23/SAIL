######################################## import area ########################################

# common library
import os
import dgl
import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.io.vasp import Poscar
from torch.utils.data import Dataset, DataLoader

######################################## function area ########################################

def get_loader(names_list, crystals_dict, graphs_dict, labels_dict, task_index, batch_size, shuffle, num_workers):
    dataset = CrystalDataset(names_list, crystals_dict, graphs_dict, labels_dict, task_index)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn), dataset.get_features_dim()


def collate_fn(samples):
    names, crystals, graphs, labels = map(list, zip(*samples))
    return names, crystals, dgl.batch(graphs), torch.from_numpy(np.array(labels))


def get_radius_dict(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(' ', '').strip('\n') for line in lines][1:-1]
    return {item.split(':')[0]: float(item.split(':')[1]) for item in lines}


class CrystalDataset(Dataset):
    
    def __init__(self, names_list, crystals_dict, graphs_dict, labels_dict, task_index):
        super(CrystalDataset, self).__init__()
        self.names = names_list
        self.crystals = [crystals_dict[name] for name in names_list]
        self.graphs = [graphs_dict[name] for name in names_list]
        self.labels = [[labels_dict[name][task_index]] for name in names_list]
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        return self.names[idx], self.crystals[idx], self.graphs[idx], self.labels[idx]
    
    def get_features_dim(self):
        # use an example will meet a single atom without edges
        return max([graph.ndata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs]), \
            max([graph.edata['x'].shape[1] if len(graph.edata['x'].shape) > 1 else 0 for graph in self.graphs])


class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_features(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        # 92 dimensions
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]
    

class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
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


def build_dgl_graph(names, labels, data_path, result_path, mode="mp"):
    
    print(f"Building {mode} dgl graphs")
    # parameters
    dmin, dmax, step, var=0, 5, 0.1, 0.5
    radius, max_neighbors = 5, 8
    
    names_list, graphs_dict, labels_dict = list(), dict(), dict()
    # pymatgen object dict, please cache pymatgen object first, it will cost lots of time!
    if not os.path.exists(f'{data_path}/{mode}/pymatgen_dict.pickle'):
        crystals_dict = {crystal_name: Poscar.from_file(f'{data_path}/{mode}/structures/{crystal_name}').as_dict() 
                         for crystal_name in tqdm(os.listdir(f'{data_path}/{mode}/structures/'))}
        with open(f'{data_path}/{mode}/pymatgen_dict.pickle', 'wb') as fw:
            pickle.dump(crystals_dict, fw)
    else:
        with open(f'{data_path}/{mode}/pymatgen_dict.pickle','rb') as f:
            crystals_dict = pickle.load(f)
    
    ari = AtomCustomJSONInitializer(f'{data_path}/atom_init.json')
    gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)
    rad = get_radius_dict(f'{data_path}/hubbard_u.yaml')
    
    for name, label in tqdm(zip(names, labels), total=len(names)):
        # pymatgen object
        crystal = Poscar.from_dict(crystals_dict[name]).structure
        
        # node features
        node_features = np.vstack([ari.get_atom_features(crystal[i].specie.number) for i in range(len(crystal))])
        
        # edge features
        all_neighbors = crystal.get_all_neighbors(r=radius, include_index=True)
        # judge the empty list
        if not any(all_neighbors):
            print(f'{name} no edges!')
            continue
        all_neighbors = [sorted(neighbors, key=lambda x:x[1]) for neighbors in all_neighbors]
        
        src, dst, distances, offset_vectors = list(), list(), list(), list()
        for src_idx, neighbors in enumerate(all_neighbors):
            if len(neighbors) > max_neighbors:
                # src
                src.extend([src_idx for _ in range(max_neighbors)])
                dst.extend(list(map(lambda x:x[2], neighbors[:max_neighbors])))
                distances.extend(list(map(lambda x:x[1], neighbors[:max_neighbors])))
                offset_vectors.extend(list(map(lambda x:x[3], neighbors[:max_neighbors])))
            else:
                src.extend([src_idx for _ in range(len(neighbors))])
                dst.extend(list(map(lambda x:x[2], neighbors)))
                distances.extend(list(map(lambda x:x[1], neighbors)))
                offset_vectors.extend(list(map(lambda x:x[3], neighbors)))
        
        distances = gdf.expand(np.array(distances))
        distances = distances.reshape((-1, distances.shape[-1]))
        
        offset_vectors = np.array(offset_vectors)
        offset_vectors = offset_vectors.reshape((-1, offset_vectors.shape[-1]))
        
        edge_features = np.concatenate([offset_vectors, distances], axis=-1)
        
        if not len(src) == len(dst) == distances.shape[0] == offset_vectors.shape[0] == edge_features.shape[0]:
            print(name + 'error!')
            print(crystal)
            print(len(src), len(dst), distances.shape[0], offset_vectors.shape[0], edge_features.shape[0])
            assert False

        # construct graph
        graph = dgl.graph((src, dst), num_nodes=len(crystal))
        graph.ndata['x'] = torch.from_numpy(node_features).float()
        graph.edata['x'] = torch.from_numpy(edge_features).float()
        
        # add all
        names_list.append(name)
        graphs_dict[name] = graph
        labels_dict[name] = label if isinstance(label, list) else [label]
        
    # store graph
    with open(f'{result_path}/{mode}.pickle', 'wb') as fw:
        pickle.dump([names_list, crystals_dict, graphs_dict, labels_dict], fw)
        print(f'dump {len(names_list)} graphs!')

if __name__ == "__main__":
    data_path = './data/source'
    result_path = './data/preprocess'
    
    # build demo dataset
    for dataset_name in ['demo_band_gap', 'demo_energy', 'demo_fermi', 'demo_mag', 'demo_volume']:
        data = pd.read_csv(f'{data_path}/{dataset_name}/property.csv', sep=',')
        names = data['name'].values.tolist()
        labels = data.iloc[:, 1:].values.tolist()
        build_dgl_graph(names, labels, data_path, result_path, mode=dataset_name)
    
    # # material project
    # data = pd.read_csv(f'{data_path}/property.csv', sep=',')
    # names = data['name'].values.tolist()
    # # band_gap, formation_energy
    # labels = data.iloc[:,1:].values.tolist()
    # build_dgl_graph(names, labels, data_path, result_path, mode="mp")

    