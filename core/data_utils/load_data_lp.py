import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from utils import get_git_repo_root_path
from typing import Dict
import numpy as np
import scipy.sparse as ssp
import json 
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.transforms import RandomLinkSplit
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import normalize

from utils import get_git_repo_root_path, config_device, init_cfg_test
from utils import time_logger
from data_utils.dataset import CustomLinkDataset
from data_utils.load_pubmed import parse_pubmed

FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'
FILE_PATH = get_git_repo_root_path() + '/'

# arxiv_2023
def get_raw_text_arxiv_2023_lp(args)-> CustomLinkDataset:
    """
    Retrieves raw text data related to ArXiv 2023.

    Args:
        args: A namespace containing data-related arguments.

    Returns:
        dataset: CustomLinkDataset object with ArXiv 2023 data.
        text: List of strings containing titles and abstracts of ArXiv 2023 papers.
        splits: Dictionary containing train, validation, and test data splits.
    
    Refs:
        refer to load_arxiv_2023.py, load_ogbn_arxiv.py
    """

    data = torch.load(FILE_PATH + 'dataset/arxiv_2023/graph.pt')

    # data.edge_index = data.adj_t.to_symmetric()

    df = pd.read_csv(FILE_PATH + 'dataset/arxiv_2023_orig/paper_info.csv')
    text = [
        f'Title: {ti}\nAbstract: {ab}'
        for ti, ab in zip(df['title'], df['abstract'])
    ]
    dataset = CustomLinkDataset('./generated_dataset', 'arxiv_2023', transform=T.NormalizeFeatures())
    dataset._data = data

    splits = get_split(dataset, 
                       args.val_pct, 
                       args.data.test_pct,
                       args.data.include_negatives,
                       args.split_labels
                       )
    return dataset, text, splits


# cora
def parse_cora():
    # load original data from cora orig without text features

    path = f'{FILE_PATH}core/dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_cora_lp(args) -> InMemoryDataset:
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


# ogbn_arxiv
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


def get_split(dataset: Dataset,
              val_pct: float,
              test_pct: float,
              include_negatives: bool,
              split_labels: bool):
    undirected = dataset._data.is_undirected()

    transform = RandomLinkSplit(is_undirected=undirected, 
                                num_val=val_pct,
                                num_test=test_pct,
                                add_negative_train_samples=include_negatives, 
                                split_labels=split_labels)

    train_data, val_data, test_data = transform(dataset._data)
    return {'train': train_data, 'valid': val_data, 'test': test_data}


def get_raw_text_arxiv_2023_lp(args):
    """
    Retrieves raw text data related to ArXiv 2023.

    Args:
        args: A namespace containing data-related arguments.

    Returns:
        dataset: CustomLinkDataset object with ArXiv 2023 data.
        text: List of strings containing titles and abstracts of ArXiv 2023 papers.
        splits: Dictionary containing train, validation, and test data splits.
    
    Refs:
        refer to load_arxiv_2023.py, load_ogbn_arxiv.py
    """
    
    undirected = args.data.undirected
    include_negatives = args.data.include_negatives
    val_pct = args.data.val_pct
    test_pct = args.data.test_pct
    split_labels = args.data.split_labels

    data = torch.load(FILE_PATH + 'dataset/arxiv_2023/graph.pt')

    # data.edge_index = data.adj_t.to_symmetric()
    df = pd.read_csv(FILE_PATH + 'dataset/arxiv_2023_orig/paper_info.csv')
    text = [
        f'Title: {ti}\nAbstract: {ab}'
        for ti, ab in zip(df['title'], df['abstract'])
    ]
    dataset = CustomLinkDataset('./generated_dataset', 'arxiv_2023', transform=T.NormalizeFeatures())
    dataset._data = data

    undirected = data.is_undirected()

    transform = RandomLinkSplit(is_undirected=undirected, 
                                num_val=val_pct,
                                num_test=test_pct,
                                add_negative_train_samples=include_negatives, 
                                split_labels=split_labels)

    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}


    return dataset, text, splits


def get_raw_text_products_lp(use_text=False, seed=0):
    data = torch.load(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    return (data, text) if use_text else (data, None)


@time_logger
def _process():
    """Process raw text data and convert it into a DataFrame for ogbn-products dataset.
        Download dataset from website http://manikvarma.org/downloads/XC/XMLRepository.html, 
        we utilize https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing
    Args:
        None

    Returns:
        None
    """
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")

    data = []
    files = [FILE_PATH + 'core/dataset/ogbn_products/Amazon-3M.raw/trn.json',
             FILE_PATH + 'core/dataset/ogbn_products/Amazon-3M.raw/tst.json']

    for f in files:
        # Read each line from the input file and parse JSON
        with open(f, "r") as input_file:
            for line in input_file:
                json_object = json.loads(line)
                data.append(json_object)
        

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    dataset = PygNodePropPredDataset(root='./generated_dataset',
        name='ogbn-products', transform=T.ToSparseTensor())
    
    nodeidx2asin = pd.read_csv(
        'generated_dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir(FILE_PATH + 'core/dataset/ogbn_products_orig'):
        os.mkdir(FILE_PATH + 'core/dataset/ogbn_products_orig')

    pd.DataFrame.to_csv(graph_df, FILE_PATH + FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])

    return graph_df

def get_pubmed_lp(args):
    corrected = False
    undirected = args.data.undirected
    include_negatives = args.data.include_negatives
    val_pct = args.data.val_pct
    test_pct = args.data.test_pct
    split_labels = args.data.split_labels
    
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    # load data
    dataset = Planetoid('./generated_dataset', 'PubMed', transform=T.NormalizeFeatures())
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

    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    
    dataset._data = data
    
    return dataset, data_pubid, splits


def get_pubmed_lp(args):
    corrected = False
    undirected = args.data.undirected
    include_negatives = args.data.include_negatives
    val_pct = args.data.val_pct
    test_pct = args.data.test_pct
    split_labels = args.data.split_labels
    
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    # load data
    dataset = Planetoid('./generated_dataset', 'PubMed', transform=T.NormalizeFeatures())
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

    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    
    dataset._data = data
    
    return dataset, data_pubid, splits
    
# TEST CODE
if __name__ == '__main__':
    args = init_cfg_test()
    get_raw_text_arxiv_2023_lp(args)
    
    dataset, data_citedid, splits = get_cora_lp(args)
    
    data, text = get_raw_text_ogbn_arxiv_lp(use_text=True)
    print(data)
    print(len(text))
    
    dataset, splits = get_raw_text_ogbn_arxiv_lp(args, use_text=False, seed=0)
    
    print(dataset)
    print(splits['test'])
    
    
    data, text = get_raw_text_products_lp(True)
    print(data)
    print(text[0])
    _process()

    dataset, data_pubid, splits = get_pubmed_lp(args)