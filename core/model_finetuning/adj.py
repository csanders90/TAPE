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
from ogb.linkproppred import PygLinkPropPredDataset
import random 
from data_utils.load import load_data_lp
from tqdm import tqdm 
import timeit 
import time 
import pandas as pd 
from lpda.lcc_3 import use_lcc, get_largest_connected_component
import networkx as nx
from torch_geometric.utils import to_undirected 
from torch_geometric.data import Data
from typing import Dict, Tuple, List, Union
from yacs.config import CfgNode as CN
from data_utils.load_data_lp import get_edge_split, load_text_citationv8
from data_utils.load_data_nc import load_embedded_citationv8


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

@time_function
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

@time_function
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

@time_function
def avg_degree(G):
    avg_deg = nx.average_neighbor_degree(G)
    # Calculate the sum of all values
    total_sum = sum(avg_deg.values())

    # Calculate the total number of values
    num_values = len(avg_deg)

    # Calculate the average value
    average_value = total_sum / num_values
    return average_value, avg_deg

@time_function
def calc_avg_cluster(G, name):
    # name pwc_large scale 100
    
    if name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed']:
        avg_cluster = nx.average_clustering(G)
    else:
        avg_cluster =  []
        for i, n in tqdm(enumerate(G.nodes())):
            if i % scale == 0:
                avg_cluster.append(nx.clustering(G, n))
                
        # Optionally, to compute the average clustering coefficient of the entire graph:
        # avg_cluster = sum(avg_cluster) / len(avg_cluster) if avg_cluster else 0
    return avg_cluster 

@time_function
def calc_avg_shortest_path(G, data, name):
    all_avg_shortest_paths = []
    if name in ['cora', 'pwc_small', 'arxiv_2023']:
        all_avg_shortest_paths = nx.average_shortest_path_length(G)
    else:
        # for index, (i, j) in tqdm(enumerate(G.nodes())):
        for _ in tqdm(range(scale)):
            n1, n2 = random.choices(list(G.nodes()), k=2)
            length = nx.shortest_path_length(G, source=n1, target=n2)
            all_avg_shortest_paths.append(length)
                    
        # all_avg_shortest_paths = sum(all_avg_shortest_paths) / len(all_avg_shortest_paths) 
    return all_avg_shortest_paths

@time_function
def calc_diameters(G, name):
    avg_diameters = []
    if name in ['cora', 'pwc_small', 'arxiv_2023']:
        avg_diameters = nx.diameter(G)
    else:
        for i in tqdm(G.nodes()):
            if i % int(scale) == 0:
                avg_diameters.append(nx.eccentricity(G, v=i))
    # avg_diameters = max(avg_all_diameters) 
    return avg_diameters


def print_data_stats(data, name, scale):

    m = construct_sparse_adj(data.edge_index.numpy())
    start = time.time()
    G = nx.from_scipy_sparse_array(m)
    print('create graph:', time.time() - start)
    
    heterogeneity = calculate_heterogeneity(G)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    avg_degree_arithmetic = int(num_edges / num_nodes)
    avg_degree_G, avg_degree_dict = avg_degree(G)
    avg_degree_G2 = avg_degree2(G, avg_degree_dict)

    all_diameters, avg_cluster, all_avg_shortest_paths= None, None, None
    if not nx.is_connected(G):
        print("Graph is not connected. Cannot calculate average shortest path length.")
        if name == 'cora':
            data = use_lcc(data)[0]
            m = construct_sparse_adj(data.edge_index.numpy())
            G = nx.from_scipy_sparse_array(m)

        if name == 'arxiv_2023':
            print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)][:4])
            data = use_lcc(data)[0]
            m = construct_sparse_adj(data.edge_index.numpy())
            G = nx.from_scipy_sparse_array(m)
        
        if name == 'pwc_large':
            print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)][:4])
            data = use_lcc(data)[0]
            m = construct_sparse_adj(data.edge_index.numpy())
            G = nx.from_scipy_sparse_array(m)
            
        if name == 'citationv8':
            print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)][:4])
            data, _, G = use_lcc(data)
            
    if name in ['cora', 'pwc_small', 'arxiv_2023', 'pubmed', 'pwc_medium', 'pwc_large', 'citationv8', 'ogbn_arxiv']:
        # all_diameters = calc_diameters(G, name)
        
        all_avg_shortest_paths = calc_avg_shortest_path(G, data, name)
        try:
            print(sorted(all_avg_shortest_paths))
        except:
            print(all_avg_shortest_paths)
        # all_diameters = max(all_avg_shortest_paths)
    avg_cluster = calc_avg_cluster(G, name)    

    print(f"------ Dataset {name}------ : Whole, "
            f"Num nodes: {data.num_nodes}, " 
            f"Num edges: {data.edge_index.shape[1]}, "
            f"heterogeneity: {heterogeneity}, "
            f"avg degree arithmetic: {avg_degree_arithmetic}, "
            f"avg degree G: {avg_degree_G}, avg degree G2: {avg_degree_G2}, "
            f"clustering: mean {np.mean(np.array(avg_cluster))}, std {np.std(np.array(avg_cluster))},"
            f"avg. of avg. shortest paths: {np.mean(all_avg_shortest_paths)}, "
            f"std. of avg. shortest paths: {np.std(all_avg_shortest_paths)}, "
            # f"avg. diameter: {np.mean(all_diameters)}, "
            # f"std. diameter: {np.std(all_diameters)}"
            )
    train_edges = splits['train'].pos_edge_label.shape[0]*2
    valid_edges = splits['valid'].pos_edge_label.shape[0]*2
    test_edges = splits['test'].pos_edge_label.shape[0]*2
        
    new_df = pd.DataFrame({
        f"Dataset {name}": [
            num_nodes,
            num_edges,
            heterogeneity,
            avg_degree_arithmetic,
            avg_degree_G,
            avg_degree_G2,
            np.mean(avg_cluster),
            np.std(avg_cluster),
            np.mean(all_avg_shortest_paths),
            np.std(all_avg_shortest_paths),
            # np.max(all_diameters),
            # np.min(all_diameters),
            int(train_edges), 
            int(valid_edges),
            int(test_edges)
        ]
    }, index=[
        "Num nodes",
        "Num edges",
        "Heterogeneity",
        "Avg degree (arithmetic)",
        "Avg degree (G)",
        "Avg degree (G2)",
        "Mean clustering coefficient",
        "Std. deviation of clustering coefficient",
        "Mean of avg. shortest paths",
        "Std. deviation of avg. shortest paths",
        # "Max diameter",
        # "Min diameter",
        "Train edges",
        "Valid edges",
        "Test edges"
    ])

    csv_filename = "data_stats.csv"
    
    try:
        existing_df = pd.read_csv(csv_filename, index_col=0)
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        
    new_df = new_df.T
    updated_df = pd.concat([existing_df, new_df])

    updated_df.to_csv(csv_filename)
    new_df.to_csv(f"{name}_data_stats.csv") 


def load_taglp_citationv8(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument
    
    data = load_embedded_citationv8(cfg.method)
    text = load_text_citationv8()
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    
    return splits, text, data

# small_data = ['pwc_small', 'cora', 'arxiv_2023']
# medium_data = ['pwc_medium', 'pubmed']
# large_data = ['citationv8', 'pwc_large']

if __name__ == '__main__':

    cfg = init_cfg_test()
    cfg = config_device(cfg)

    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--data', dest='data', type=str, required=False,
                        help='data name', default='pwc_medium')
    parser.add_argument('--scale', dest='scale', type=int, required=False,
                        help='data name')
    args = parser.parse_args()
    # scale = 1000
    name = args.data
    scale = args.scale 
    
    if name == 'citationv8':
        start = time.time()
        splits, text, data = load_taglp_citationv8(cfg.data)
        print(f"Time taken to load data: {time.time() - start} s")

    elif name.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=name)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
    else:
        splits, text, data = load_data_lp[name](cfg.data)
    print_data_stats(data, name, scale)

    exit(-1)





