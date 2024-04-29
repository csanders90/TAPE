from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_git_repo_root_path
from utils import time_logger


FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'


FILE_PATH = get_git_repo_root_path() + '/'

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


def get_raw_text_products(use_text=False, seed=0):
    data = torch.load(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()
    
    if not use_text:
        return data, None

    return data, text


def get_raw_text_products_lp(use_text=False, seed=0):
    data = torch.load(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()
    
    if not use_text:
        return data, None

    return data, text

if __name__ == '__main__':
    data, text = get_raw_text_products(True)
    print(data)
    print(text[0])
    _process()