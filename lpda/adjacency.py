import networkx as nx
from matplotlib import pyplot, patches
import numpy as np 
import torch
import random
import os 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_torch_coo_tensor
from ogb.nodeproppred import NodePropPredDataset
from scipy.sparse import csc_array
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_utils.load import load_data

import matspy as spy # https://github.com/alugowski/matspy

def plot_adjacency_matrix(G: nx.graph, name: str) -> None:
    """
    Plot the adjacency matrix of a networkx graph.

    Parameters:
    - G: nx.Graph, input graph
    - name: str, output file name
    
    adopted from  https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern

    """
    adjacency_matrix = nx.to_numpy_array(G)
    # , dtype=np.bool, nodelist=node_order
    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none"
                  )
    pyplot.savefig(f'{name}')


def draw_adjacency_matrix(adj: np.array, name: str) -> None:
    """
    Plot the adjacency matrix of a numpy array.

    Parameters:
    - adj: np.array, adjacency matrix
    - name: str, output file name
    """
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adj,
                  cmap="Greys",
                  interpolation="none")
    pyplot.savefig(f'{name}')
    

import os.path as osp 

def compare_adj(data_name, data_edges):
      """_summary_

      Args:
            data_name (_type_): _description_
            data_edges (_type_): _description_
      """
      
      path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
      dataset = Planetoid(path, data_name,
                        transform=T.NormalizeFeatures())
      data = dataset[0]

      # check if loaded dataset is the same as Planetoid
      sorted_data_edges = np.sort(data_edges, axis=0)
      sorted_data_index = np.sort(data.edge_index.numpy(), axis=0)
      are_datasets_equal = np.array_equal(sorted_data_edges, sorted_data_index)

      print(sorted_data_edges)
      print(sorted_data_index)
      print(f'Are datasets equal? {are_datasets_equal}')
      # original Planetoid dataset
      adj = to_torch_coo_tensor(data.edge_index)
      adj = adj.to_dense().numpy()
      G = nx.from_numpy_array(adj, create_using=nx.Graph)

      plot_adjacency_matrix(G, f'plots/{data_name}_data.edge_index.png')

      # TAG 
      adj = to_torch_coo_tensor(torch.tensor(data_edges))
      adj = adj.to_dense().numpy()
      plot_adjacency_matrix(G, f'plots/{data_name}_data_edges.png')
    
def plot_adj_sparse():
      """plot the adjacency matrix of a sparse matrix"""

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
      
def plot_coo_matrix(m: coo_matrix, name: str):
    """
    Plot the COO matrix.

    Parameters:
    - m: coo_matrix, input COO matrix
    - name: str, output file name
    """
    
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, 's', color='black', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(name)
    return ax

def construct_sparse_adj(edge_index) -> coo_matrix:
      """
      Construct a sparse adjacency matrix from an edge index.

      Parameters:
      - edge_index: np.array or tuple, edge index
      """
      # Resource: https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern

      if type(edge_index) == tuple:
            edge_index = np.concatenate([[edge_index[0].numpy()], 
                                         [edge_index[1].numpy()]], axis=0)
      elif type(edge_index) != np.ndarray:
            edge_index.numpy()
            
      rows, cols = edge_index[0, :], edge_index[1, :]
      vals = np.ones_like(rows)
      shape = (edge_index.max()+1, edge_index.max()+1)
      m = coo_matrix((vals, (rows, cols)), shape=shape)
      return m
      
import argparse 

if __name__ == '__main__':
      
      parser = argparse.ArgumentParser()
      parser.add_argument('--dataset', type=str, default='cora',
                          help='Dataset name.')
      args = parser.parse_args()
      
      scale = 100000
      name = args.dataset
      
      if name == 'ogbn-products':
            dataset = NodePropPredDataset(name)
            edge_index = dataset[0][0]['edge_index']

            # edge index to sparse matrix
            edge_index = edge_index[:, ::scale]
            m = construct_sparse_adj(edge_index)
            plot_coo_matrix(m, f'plots/{name}_data_edges.png')
            
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}_data_edges_spy.png", bbox_inches='tight')

            
            data, num_class, text = load_data(name)
            m = construct_sparse_adj(data.edge_index.coo())
            plot_coo_matrix(m, f'plots/{name}_data_index.png')
            
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}_data_index_spy.png", bbox_inches='tight')


      if name == 'ogbn-arxiv':
            dataset = NodePropPredDataset(name)
            edge_index = dataset[0][0]['edge_index']

            m = construct_sparse_adj(edge_index[:, ::2])
            plot_coo_matrix(m, f'plots/{name}_data_edges.png')
            
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}_data_edges_spy.png", bbox_inches='tight')

            
      if name == 'arxiv_2023':
            data, num_class, text = load_data(name)
            m = construct_sparse_adj(data.edge_index.numpy())
            plot_coo_matrix(m, f'plots/{name}_data_index.png')
            
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}_data_index_spy.png", bbox_inches='tight')


      for name in ['cora', 'pubmed']:
            data, num_class, text = load_data(name)
            
            compare_adj(name, data.edge_index.numpy())
            m = construct_sparse_adj(data.edge_index.numpy())
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}_data_index_spy.png", bbox_inches='tight')
            
