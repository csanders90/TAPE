import os
import sys
sys.path.insert(0, '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core')

import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from data_utils.load_data_nc import (
    load_graph_cora, 
    load_graph_pubmed, 
    load_tag_arxiv23, 
    load_graph_ogbn_arxiv
)
from graphgps.encoder.seal import do_edge_split, do_ogb_edge_split
import copy as cp
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch 

color_list = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
data_list = ['custom-cora', 'custom-arxiv_2023', 'custom-pubmed', 'custom-ogbn_arxiv']

# Example function placeholders (replace these with actual implementations)
# List of colors and datasets
color_list = ['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
data_list = ['custom-cora', 'custom-arxiv_2023', 
             'custom-pubmed', 'custom-ogbn_arxiv', 
             'obgl-ppa', 'ogbl-collab', 'ogbl-ddi',
             'ogbl-citation2', 'ogbl-wikikg2', 'ogbl-vessel', 'ogbl-biokg']

plt.figure(figsize=(8, 6))

# Plotting each dataset
for dataset, color in zip(data_list, color_list):
    fast_split = True

    if dataset == 'custom-pubmed':
        data = load_graph_pubmed(False)
    elif dataset == 'custom-cora':
        data, _ = load_graph_cora(False)
    elif dataset == 'custom-arxiv_2023':
        data, _ = load_tag_arxiv23()
    elif dataset == 'custom-ogbn_arxiv':
        data = load_graph_ogbn_arxiv(False)
    elif dataset.startswith('ogbl'):
        data = PygLinkPropPredDataset(name=dataset)
        split_edge = data.get_edge_split()
        data = data[0]
        if dataset.startswith('ogbl-vessel'):
            # normalize node features
            data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
            data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
            data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)   
    else:
        continue
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=1, shuffle=False)
    for data in loader:
        print(data)
    print(f"Dataset: {dataset}")
    adj = to_scipy_sparse_matrix(data.edge_index)

    G = nx.from_scipy_sparse_array(adj)
    # Get the degree of each node
    degrees = [G.degree(n) for n in G.nodes()]

    # Compute degree distribution
    degree_counts = np.bincount(degrees)
    degree_distribution = degree_counts / len(G.nodes())

    # Compute cumulative degree distribution
    cumulative_distribution = np.cumsum(degree_distribution[::-1])[::-1]

    plt.loglog(range(len(cumulative_distribution)), cumulative_distribution, marker='o', linestyle='-', color=color, label=dataset)

# Add legend outside the plot
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.xlabel('degree d')
plt.ylabel('fraction of vertices with degree ≥ d')
plt.title('Degree Distribution for Various Datasets')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig(f'{dataset}_degree_distribution.png')
