import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
from torch_geometric.transforms import RandomLinkSplit
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from lpda.adjacency import plot_coo_matrix
from data_utils.load_pubmed import get_raw_text_pubmed, get_pubmed_casestudy, parse_pubmed
import matplotlib.pyplot as plt
from lpda.adjacency import construct_sparse_adj
import scipy.sparse as ssp

from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR


def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    x = torch.tensor(data_X)
    edge_index = torch.tensor(data_edges)
    y = torch.tensor(data_Y)
    num_nodes = data.num_nodes
    
    # split data
    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        train_id = [i for i in train_id if not is_mistake[i]]
        val_id = [i for i in val_id if not is_mistake[i]]
        test_id = [i for i in test_id if not is_mistake[i]]


    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        

    undirected = data.is_undirected()
    undirected = True
    include_negatives = True
    val_pct = 0.15
    test_pct = 0.05
    
    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    
    dataset._data = data
    
    return dataset, data_pubid, splits

if __name__ == "__main__":
        
    dataset, data_pubid, splits = get_pubmed_casestudy(corrected=False, SEED=0)
    print(dataset)

    test_split = splits['test']
    labels = test_split.edge_label
    test_index = test_split.edge_label_index
    
    edge_index = splits['train'].edge_index
    edge_weight = torch.ones(edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
        pos_test_pred, edge_index = eval(use_lsf)(A, test_index)
        
        plt.figure()
        plt.plot(pos_test_pred)
        plt.plot(labels)
        plt.savefig(f'{use_lsf}.png')
        
        acc = torch.sum(pos_test_pred == labels)/pos_test_pred.shape[0]
        print(f" {use_lsf}: accuracy: {acc}")
        
    m = construct_sparse_adj(edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
            
    # 'shortest_path', 'katz_apro', 'katz_close', 'Ben_PPR'
    for use_gsf in ['Ben_PPR', 'SymPPR']:
        scores, edge_reindex, labels = eval(use_gsf)(A, test_index, labels)
        
        # print(scores)
        # print(f" {use_heuristic}: accuracy: {scores}")
        pred = torch.zeros(scores.shape)
        cutoff = 0.05
        thres = scores.max()*cutoff 
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")
    
    
    for use_gsf in ['shortest_path', 'katz_apro', 'katz_close']:
        scores = eval(use_gsf)(A, test_index)
        
        pred = torch.zeros(scores.shape)
        thres = scores.min()*10
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")