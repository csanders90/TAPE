import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from typing import Dict
import numpy as np
import scipy.sparse as ssp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from utils import get_git_repo_root_path, config_device, set_cfg, init_cfg_test


FILE_PATH = get_git_repo_root_path() + '/'


def get_raw_text_arxiv(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(root='./generated_dataset',
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    
    train_mask = train_mask
    val_mask = val_mask
    test_mask = test_mask

    if data.adj_t.is_symmetric():
        is_symmetric = True
    else:
        edge_index = data.adj_t.to_symmetric()
    
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        'generated_dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    tsv_path = FILE_PATH + 'core/dataset/ogbn_arixv_orig/titleabs.tsv'
    raw_text = pd.read_csv(tsv_path,
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    raw_text['paper id'] = pd.to_numeric(raw_text['paper id'], errors='coerce')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    
    # recreate InMemoryDataset
    num_nodes = data.num_nodes
    x = data.x
    y = data.y
    
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
    
    return dataset, text


def get_raw_text_ogbn_arxiv_lp(args, use_text=False, seed=0)-> InMemoryDataset:

    undirected = args.data.undirected
    include_negatives = args.data.include_negatives
    val_pct = args.data.val_pct
    test_pct = args.data.test_pct
    split_labels = args.data.split_labels
    
    device = config_device(args)

    transform = T.Compose([
        T.NormalizeFeatures(),  
        T.ToDevice(device),    
        T.RandomLinkSplit(num_val=val_pct, num_test=test_pct, is_undirected=undirected,  # 这一步很关键，是在构造链接预测的数据集
                        split_labels=split_labels, add_negative_train_samples=False),])

    # load data
    dataset = PygNodePropPredDataset(root='./generated_dataset',
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    if data.adj_t.is_symmetric():
        is_symmetric = True
    else:
        edge_index = data.adj_t.to_symmetric()
        
    # check is data has changed and try to return dataset
    x = torch.tensor(data.x).float()
    edge_index = torch.LongTensor(edge_index).long()
    y = torch.tensor(data.y).long()
    num_nodes = len(data.y)

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

    return dataset, splits


# TEST CODE
if __name__ == '__main__':
    data, text = get_raw_text_arxiv(use_text=True)
    print(data)
    print(len(text))
    
    args = init_cfg_test()
    get_raw_text_ogbn_arxiv_lp(args, use_text=False, seed=0)