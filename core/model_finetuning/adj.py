import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import networkx as nx
from matplotlib import pyplot
import numpy as np
import torch
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_torch_coo_tensor
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import matspy as spy  # https://github.com/alugowski/matspy
import math
import argparse
from graphgps.utility.utils import get_git_repo_root_path, config_device, init_cfg_test
import os.path as osp
import numpy as np

from data_utils.load import load_data_lp
from tqdm import tqdm 
import timeit 
import time 

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.5f} seconds")
        return result
    return wrapper

@time_function
def calculate_heterogeneity(graph):
    degrees = [degree for _, degree in graph.degree()]
    average_degree = sum(degrees) / len(degrees)
    variance_degree = sum((degree - average_degree) ** 2 for degree in degrees) / len(degrees)
    return math.log10(math.sqrt(variance_degree) / average_degree)


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
    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
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
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adj,
                  cmap="Greys",
                  interpolation="none")
    pyplot.savefig(f'{name}')



def compare_adj(data_name, data_edges):
    """_summary_

    Args:
          data_name (_type_): _description_
          data_edges (_type_): _description_
    """

    path = osp.join(osp.dirname(osp.realpath(__file__)), './dataset')
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

    plot_adjacency_matrix(G, f'{data_name}_data.edge_index.png')

    # TAG
    adj = to_torch_coo_tensor(torch.tensor(data_edges))
    adj = adj.to_dense().numpy()
    plot_adjacency_matrix(G, f'{data_name}_data_edges.png')


def plot_adj_sparse():
    """plot the adjacency matrix of a sparse matrix"""
    raise NotImplementedError

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

def coo_tensor_to_coo_matrix(coo_tensor: SparseTensor):
    coo = coo_tensor.coo()
    row_indices = coo[0].numpy()
    col_indices = coo[1].numpy()
    values = coo[2].numpy()
    shape = coo_tensor.sizes()

    # Create a scipy coo_matrix
    return coo_matrix((values, (row_indices, col_indices)), shape=shape)


def plot_pos_neg_adj(m_pos: coo_matrix, m_neg: coo_matrix, name: str):
    """
    Plot the COO matrix.

    Parameters:
    - m: coo_matrix, input COO matrix
    - name: str, output file name
    """

    if not isinstance(m_pos, coo_matrix):
        m_pos = coo_matrix(m_pos)
    if not isinstance(m_neg, coo_matrix):
        m_neg = coo_matrix(m_neg)

    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m_neg.col, m_neg.row, 's', color='black', ms=1)
    ax.plot(m_pos.col, m_pos.row, 's', color='blue', ms=1)
    #     ax.set_xlim(0, m_neg.shape[1])
    #     ax.set_ylim(0, m_neg.shape[0])
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

    if edge_index.shape[0] > edge_index.shape[1]:
        edge_index = edge_index.T

    rows, cols = edge_index[0, :], edge_index[1, :]
    vals = np.ones_like(rows)
    shape = (edge_index.max() + 1, edge_index.max() + 1)
    m = coo_matrix((vals, (rows, cols)), shape=shape)
    return m


def avg_degree2(G, avg_degree_dict={}):
    avg_degree_list = []
    for index in range(max(G.nodes) + 1):
        degree = 0
        adj_list = list(G.neighbors(index))
        if adj_list != []:
            for neighbors in adj_list:
                try:
                    degree += avg_degree_dict[neighbors]
                except:
                    degree += 0
            avg_degree_list.append(degree / len(adj_list))

    return np.array(avg_degree_list).mean()


def avg_degree(G):
    avg_deg = nx.average_neighbor_degree(G)
    # Calculate the sum of all values
    total_sum = sum(avg_deg.values())

    # Calculate the total number of values
    num_values = len(avg_deg)

    # Calculate the average value
    average_value = total_sum / num_values
    return average_value, avg_deg

import networkx as nx
from itertools import combinations

def calc_avg_cluster(G, name):
    # name pwc_large scale 100
    
    if name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed']:
        avg_cluster = nx.average_clustering(G)
    else:
        avg_cluster =  []
        for i, n in tqdm(enumerate(G.nodes())):
            if i % int(scale/10) == 0:
                avg_cluster.append(nx.clustering(G, n))
                
        # Optionally, to compute the average clustering coefficient of the entire graph:
        avg_cluster = sum(avg_cluster) / len(avg_cluster) if avg_cluster else 0
    return avg_cluster

def calc_avg_shortest_path(G, data, name):
    all_avg_shortest_paths = []
    if name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed']:
        all_avg_shortest_paths = nx.average_shortest_path_length(G)
    else:
        for i, j in G.edges():
            if i % scale == 0:
                try:
                    all_avg_shortest_paths.append(nx.shortest_path_length(G, source=i, target=j)) 
                except:
                    all_avg_shortest_paths.append(0)
                    
        all_avg_shortest_paths = sum(all_avg_shortest_paths) / len(all_avg_shortest_paths) if all_avg_shortest_paths else 0
    return all_avg_shortest_paths


def calc_diameters(G, name):
    avg_all_diameters = []
    if name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed']:
        avg_diameters = nx.diameter(G)
    else:
        for i in G.nodes():
            if i % int(scale/10) == 0:
                avg_all_diameters.append(nx.eccentricity(G, v=i))
    avg_diameters = max(avg_all_diameters) if avg_all_diameters else 0
    return avg_diameters 
    
def print_data_stats(data, name, scale):

    m = construct_sparse_adj(data.edge_index.numpy())
    G = nx.from_scipy_sparse_array(m)

    heterogeneity = calculate_heterogeneity(G)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    avg_degree_arithmetic = int(num_edges / num_nodes)
    avg_degree_G, avg_degree_dict = avg_degree(G)
    avg_degree_G2 = avg_degree2(G, avg_degree_dict)

    print(f"------ Dataset {name}------ : Whole")
    print("Num nodes: ", data.num_nodes)
    print("Num edges: ", data.edge_index.shape[1])
    print(f"heterogeneity: {heterogeneity}")
    print(f"avg degree arithmetic {avg_degree_arithmetic}")
    print(f"avg degree G {avg_degree_G}, avg degree G2 {avg_degree_G2}.")


    all_diameters, avg_cluster, all_avg_shortest_paths= None, None, None
    if not nx.is_connected(G):
        # who?
        print("Graph is not connected. Cannot calculate average shortest path length.")
        all_avg_shortest_paths = 0  # Or handle it in a way that suits your needs
        all_diameters = 0
    elif name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed', 'arxiv_2023', 'pwc_large', 'pwc_medium', 'citationv8']:
        all_diameters = calc_diameters(G, name)
        all_avg_shortest_paths = calc_avg_shortest_path(G, data, name)
    avg_cluster = calc_avg_cluster(G, name)
            
    
    for dataset_name, dataset_data in splits.items():
        print()
        print(f"------ Dataset {name}------ : {dataset_name}")
        # print("Node attributes: ", dataset_data.node_attrs())
        print("Positive edge labels: ", dataset_data.pos_edge_label.shape[0])
        print("Negative edge labels: ", dataset_data.neg_edge_label.shape[0])
        

    print(f"clustering {avg_cluster}.")    
    print("avg. of avg. shortest paths: ", np.mean(all_avg_shortest_paths))
    print("std. of avg. shortest paths: ", np.std(all_avg_shortest_paths))
    print("avg. diameter: ", np.mean(all_diameters))
    print("std. diameter: ", np.std(all_diameters))
    

if __name__ == '__main__':

    cfg = init_cfg_test()
    cfg = config_device(cfg)

    name_list = ['pwc_small', 'cora', 'pubmed', 'arxiv_2023', 'pwc_medium', 'ogbn-arxiv', 'citationv8', 'pwc_large']

    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--data', dest='data', type=str, required=False,
                        help='data name', default='ogbn-arxiv')
    parser.add_argument('--scale', dest='scale', type=int, required=False,
                        help='data name')
    args = parser.parse_args()
    
    scale = 1000
    name = args.data
    
    if name == 'pwc_small':
        splits, text, data = load_data_lp[name](cfg.data)

        print_data_stats(data, name, scale)

    if name == 'pwc_medium':
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(data, name, scale)
        
    if name == 'pwc_large':
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(data, name, scale)
    
    if name == 'ogbn-arxiv':        
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(data, name)

    if name == 'citationv8':
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(data, name, scale)
        
    exit(-1)


    # name = 'cora'
    if name == 'cora':
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(data)

    # name = 'arxiv_2023'
    if name == 'arxiv_2023':
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(G)

    # name = 'pubmed'
    if name == 'pubmed':
        splits, text, data = load_data_lp[name](cfg.data)
        print_data_stats(data)


