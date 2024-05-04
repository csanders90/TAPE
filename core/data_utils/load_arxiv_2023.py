import torch
import pandas as pd
import numpy as np
import torch
import random
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Dataset
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_git_repo_root_path

FILE_PATH = get_git_repo_root_path() + '/'

def get_raw_text_arxiv_2023(use_text=False, seed=0):
    """
    Load and process the arxiv_2023 dataset for node classification task. 
    # TODO add data resource
    Args:
        use_text (bool, False): If True, the raw text of the dataset will be used. 
        seed (int, 0): The seed RNG for reproducibility of the dataset split. 
    
    Default Vars:
    graph path: dataset/arxiv_2023/graph.pt
    text path:'dataset/arxiv_2023_orig/paper_info.csv'

    Returns:
        data (Data): A PyTorch Geometric Data object containing the processed dataset. 
        The dataset is split into training, validation, and test sets. 
        The split is determined by the provided seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load(FILE_PATH + 'core/dataset/arxiv_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    # data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None
    else:
        df = pd.read_csv(FILE_PATH + 'core/dataset/arxiv_2023_orig/paper_info.csv')
        text = []
        for ti, ab in zip(df['title'], df['abstract']):
            text.append(f'Title: {ti}\nAbstract: {ab}')
            # text.append((ti, ab))
    return data, text

# TEST CODE
if __name__ == '__main__':
    data, text = get_raw_text_arxiv_2023(use_text=True)
    print(data)
    print(text[0])