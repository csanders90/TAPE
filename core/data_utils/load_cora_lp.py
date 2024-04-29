        
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

from typing import Dict
import numpy as np
import scipy.sparse as ssp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from utils import get_git_repo_root_path, config_device

FILE_PATH = get_git_repo_root_path() + '/'

def parse_cora():
    # load original data from cora orig without text features
    
    path = FILE_PATH + 'core/dataset/cora_orig/cora'
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


def get_cora_casestudy(args) -> InMemoryDataset:
    undirected = args.data.undirected
    include_negatives = args.data.include_negatives
    val_pct = args.data.val_pct
    test_pct = args.data.test_pct
    split_labels = args.data.split_labels
    
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    device = config_device(args)

    transform = T.Compose([
        T.NormalizeFeatures(),  
        T.ToDevice(device),    
        T.RandomLinkSplit(num_val=val_pct, num_test=test_pct, is_undirected=undirected,  # 这一步很关键，是在构造链接预测的数据集
                        split_labels=split_labels, add_negative_train_samples=False),])

    # load data
    dataset = Planetoid('./generated_dataset', 'cora',
                        transform=transform)

    data = dataset[0]
    # check is data has changed and try to return dataset
    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).long()
    y = torch.tensor(data_Y).long()
    num_nodes = len(data_Y)

    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        
    dataset._data = data

    undirected = data.is_undirected()

    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}

    return dataset, data_citeid, splits

