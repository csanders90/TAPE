import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
from torch_geometric.data import Data
from utils import get_git_repo_root_path

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
        'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

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

# TEST CODE
if __name__ == '__main__':
    data, text = get_raw_text_arxiv(use_text=True)
    print(data)
    print(len(text))