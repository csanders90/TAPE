"""
utils for getting the largest connected component of a graph
"""
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from adjacency import load_data
import torch 
import matspy as spy
from adjacency import construct_sparse_adj


from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.data import Data, InMemoryDataset
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
import torch
from typing import List 
from torch_geometric.utils import is_undirected, to_undirected, contains_self_loops, contains_isolated_nodes, subgraph, get_laplacian
from torch_geometric.utils import spmm, one_hot, normalized_cut, to_scipy_sparse_matrix, erdos_renyi_graph, remove_self_loops, add_self_loops, to_networkx, from_networkx

from lcc_3 import use_lcc



if __name__ == '__main__':

    # params
    path = '.'
    # 'cora', 'pubmed', 'ogbn-arxiv', 'ogbn-products', 'arxiv_2023'
    for name in [ 'arxiv_2023']:
        
        if name in ['cora', 'pubmed', 'citeseer', 'ogbn-arxiv', 'ogbn-products', 'arxiv_2023']:
            use_lcc_flag = True
            
            # planetoid_data = Planetoid(path, name) 
            data, num_class, text = load_data(name, use_dgl=False, use_text=False, use_gpt=False, seed=0)
            
            print(f" original num of nodes: {data.num_nodes}")
            
            if name.startswith('ogb'):
                edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
            else:
                edge_index = data.edge_index.numpy()
                
            m = construct_sparse_adj(edge_index)
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}/{name}_ori_data_index_spy.png", bbox_inches='tight')
            
            
            if use_lcc_flag:
                data_lcc = use_lcc(data)
            print(data_lcc.num_nodes)


            m = construct_sparse_adj(data_lcc.edge_index.numpy())
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}/{name}_lcc_data_index_spy.png", bbox_inches='tight')
        