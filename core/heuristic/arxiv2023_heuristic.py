
import torch
import pandas as pd
import numpy as np
import torch
import random
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data.dataset import Dataset

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils.dataset import CustomPygDataset, CustomLinkDataset
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR
from data_utils.load_pubmed import get_raw_text_pubmed, get_pubmed_casestudy, parse_pubmed
import matplotlib.pyplot as plt
from lpda.adjacency import construct_sparse_adj
import scipy.sparse as ssp
from lpda.adjacency import plot_coo_matrix

FILE = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/TAPE_chen/'

from torch_geometric.data import Dataset
import torch



def get_raw_text_arxiv_2023(use_text=False, seed=0):

    data = torch.load(FILE + 'dataset/arxiv_2023/graph.pt')
    
    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        text = None

    df = pd.read_csv(FILE + 'dataset/arxiv_2023_orig/paper_info.csv')
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
        
    dataset = CustomLinkDataset('./dataset', 'arxiv_2023', transform=T.NormalizeFeatures())
    dataset._data = data
    
    undirected = data.is_undirected()
    undirected = True
    include_negatives = True
    val_pct = 0.15
    test_pct = 0.05
    
    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    

    return dataset, text, splits



if __name__ == "__main__":
        
    dataset, text, splits = get_raw_text_arxiv_2023(use_text=False, seed=0)
    print(dataset._data)
    
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