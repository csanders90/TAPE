import networkx as nx
from matplotlib import pyplot, patches
import numpy as np 
import numpy as np
import torch
import random
import os 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_torch_coo_tensor
import networkx as nx
import matspy as spy

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_utils.load import load_data

def plot_adjacency_matrix(G: nx.graph, name: str) -> None:
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_array(G)
    # , dtype=np.bool, nodelist=node_order
    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    pyplot.savefig(f'{name}')


def draw_adjacency_matrix(adj: np.array, name: str) -> None:
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    # , dtype=np.bool, nodelist=node_order
    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adj,
                  cmap="Greys",
                  interpolation="none")
    pyplot.savefig(f'{name}')
    
def compare_adj(data_name, data_edges):
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
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
    
    draw_adjacency_matrix(G, f'plots/{data_name}_data.edge_index.png')

    # TAG 
    adj = to_torch_coo_tensor(torch.tensor(data_edges))
    adj = adj.to_dense().numpy()
    draw_adjacency_matrix(G, f'plots/{data_name}_data_edges.png')
    
def plot_adj_sparse():
      """plot the adjacency matrix of a sparse matrix"""

import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
      
def plot_coo_matrix(m: coo_matrix, name: str):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
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
      """plot the adjacency matrix of a sparse matrix
      shape = (100000, 100000)
      rows = np.int_(np.round_(shape[0]*np.random.random(1000)))
      cols = np.int_(np.round_(shape[1]*np.random.random(1000)))
      vals = np.ones_like(rows)
      coo_matrix((vals, (rows, cols)), shape=shape)
      """
      rows, cols = edge_index[0, :], edge_index[1, :]
      vals = np.ones_like(rows)
      shape = (edge_index.max()+1, edge_index.max()+1)
      m = coo_matrix((vals, (rows, cols)), shape=shape)
      return m
      
      
if __name__ == '__main__':
          
      name = 'ogbn-products'
      if name == 'ogbn-products':
            from ogb.nodeproppred import NodePropPredDataset
            dataset = NodePropPredDataset('ogbn-products')
            edge_index = dataset[0][0]['edge_index']
            # TAG 
            # edge index to sparse matrix
            edge_index = edge_index[:, ::100000]
            m = construct_sparse_adj(edge_index)
            plot_coo_matrix(m, f'plots/{name}_data_edges.png')
            
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig("plots/spy.png", bbox_inches='tight')

            spy(m)
            
            data, num_class, text = load_data(name)

            # sorted_data_edges = np.sort(edge_index, axis=0)
            
            # sorted_data_index = np.sort(data.edge_index.numpy(), axis=0)
            # are_datasets_equal = np.array_equal(sorted_data_edges, sorted_data_index)

            # print(sorted_data_edges)
            # print(sorted_data_index)
            # print(f'Are datasets equal? {are_datasets_equal}')
            # original Planetoid dataset
            data.edge_index.coo()[0]
            
            adj = data.edge_index.to_torch_sparse_coo_tensor()[:, ::10000]
            adj = adj.to_dense().numpy()  
            plot_adjacency_matrix(adj, f'plots/{name}_data.edge_index.png')


            
      for name in ['cora', 'pubmed']:
            data, text = load_data(name)
            compare_adj(name, data.edge_index.numpy())