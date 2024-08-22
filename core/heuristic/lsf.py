"""
A selection of heuristic methods (Personalized PageRank, Adamic Adar and Common Neighbours) for link prediction
"""
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from typing import Tuple
import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from tqdm import tqdm
from torch_geometric.utils import from_scipy_sparse_matrix
np.seterr(divide = 'ignore') 


def CN(A: csr_matrix, edge_index: torch.Tensor, batch_size: int = 100000) -> Tuple[FloatTensor, torch.Tensor]:
    """
    Common neighbours
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index (torch.Tensor of shape [2, num_edges])
    :param batch_size: int
    :return: Tuple containing a FloatTensor of scores and the pyg edge_index
    """
    edge_index = edge_index.t()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def InverseRA(A: csr_matrix, edge_index: torch.Tensor, batch_size: int = 100000) -> Tuple[FloatTensor, torch.Tensor]:
    """
    Inverse Adamic Adar
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index (torch.Tensor of shape [2, num_edges])
    :param batch_size: int
    :return: Tuple containing a FloatTensor of scores and the pyg edge_index
    """
    edge_index = edge_index.t()
    multiplier = np.exp(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated InverseRA for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def AA(A: csr_matrix, edge_index: torch.Tensor, batch_size: int = 100000) -> Tuple[FloatTensor, torch.Tensor]:
    """
    Adamic Adar
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index (torch.Tensor of shape [2, num_edges])
    :param batch_size: int
    :return: Tuple containing a FloatTensor of scores and the pyg edge_index
    """
    edge_index = edge_index.t()
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Adamic Adar for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def RA(A: csr_matrix, edge_index: torch.Tensor, batch_size: int = 100000) -> Tuple[FloatTensor, torch.Tensor]:
    """
    Resource Allocation https://arxiv.org/pdf/0901.0553.pdf
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index (torch.Tensor of shape [2, num_edges])
    :param batch_size: int
    :return: Tuple containing a FloatTensor of scores and the pyg edge_index
    """
    edge_index = edge_index.t()
    multiplier = 1 / A.sum(axis=0)
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Resource Allocation for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index