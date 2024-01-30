import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os
from torch_geometric.data import Data, InMemoryDataset

# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs
FILE_PATH = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/TAPE_chen/dataset'

def get_cora_casestudy(SEED=0) -> InMemoryDataset:
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./dataset', data_name,
                        transform=T.NormalizeFeatures())

    data = dataset[0]
    # check is data has changed and try to return dataset
    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).long()
    y = torch.tensor(data_Y).long()
    num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    test_id = np.sort(node_id[int(data.num_nodes * 0.8):])
    
    train_mask = torch.tensor(
        [x in train_id for x in range(data.num_nodes)])
    val_mask = torch.tensor(
        [x in val_id for x in range(data.num_nodes)])
    test_mask = torch.tensor(
        [x in test_id for x in range(data.num_nodes)])

    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        
    dataset._data = data
    
    return dataset, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora():
    path = FILE_PATH + '/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open(FILE_PATH + '/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn
    
    path = FILE_PATH + '/cora_orig/mccallum/cora/extractions/'
    
    # for debug
    # save file list
    with open('extractions.txt', 'w') as txt_file:
        # Write each file name to the text file
        for file_name in os.listdir(path):
            txt_file.write(file_name + '\n')
            
    text = []
    not_loaded = []
    i = 0
    for pid in data_citeid:
        fn = pid_filename[pid]
        try:
            if os.path.exists(path+fn): 
                pathfn = path+fn
            elif os.path.exists(path+fn.replace(":", "_")):
                pathfn = path+fn.replace(":", "_")
            elif os.path.exists(path+fn.replace("_", ":")):
                pathfn = path+fn.replace("_", ":")
                
            with open(pathfn) as f:
                lines = f.read().splitlines()
                    
            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
            text.append(ti+'\n'+ab)
        except:
            not_loaded.append(pathfn)
            i += 1

        print(f"not loaded {i} papers.")
        print(f"not loaded papers: {not_loaded}")
    return data, text
