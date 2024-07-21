import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data

from data_utils.load import load_data_lp




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--name', type=str, required=False, default='cora')
    parser.add_argument('--undirected', type=bool, required=False, default=True)
    parser.add_argument('--include_negatives', type=bool, required=False, default=True)
    parser.add_argument('--val_pct', type=float, required=False, default=0.15)
    parser.add_argument('--test_pct', type=float, required=False, default=0.05)
    parser.add_argument('--split_labels', type=bool, required=False, default=True)
    parser.add_argument('--device', type=str, required=False, default='cpu')
    return parser.parse_args()

def remap_node_indices(data, splits):
    edge_index = data.edge_index
    degrees = torch.bincount(edge_index.view(-1))
    sorted_indices = torch.argsort(degrees, descending=True)
    index_map = torch.empty_like(sorted_indices)
    index_map[sorted_indices] = torch.arange(len(sorted_indices))
    new_edge_index = index_map[edge_index]
    new_x = data.x[sorted_indices]
    data.edge_index = new_edge_index
    data.x = new_x
    for set in ['train', 'valid', 'test']:
        splits[set].x = new_x
        splits[set].edge_index = index_map[splits[set].edge_index]
        splits[set].pos_edge_label_index = index_map[splits[set].pos_edge_label_index]
        splits[set].neg_edge_label_index = index_map[splits[set].neg_edge_label_index]
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        num_nodes=data.num_nodes
    )
    return new_data, splits


def visualize_adjacency_matrix(name, num_nodes, pos_edge_index=None, neg_edge_index=None):
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix = adj_matrix.numpy()

    fig, ax = plt.subplots(figsize=(20, 20))

    if pos_edge_index is not None:
        pos_edges = pos_edge_index.t().numpy()
        ax.scatter(pos_edges[:, 1], pos_edges[:, 0], color='green', label='Positive Edges', s=10)

    if neg_edge_index is not None:
        neg_edges = neg_edge_index.t().numpy()
        ax.scatter(neg_edges[:, 1], neg_edges[:, 0], color='red', label='Negative Edges', s=10)

    plt.title(f'Adjacency Matrix - {name}')
    plt.legend()
    plt.savefig('./visualize/'+ f'Adjacency Matrix - {name}')
    plt.close()




if __name__ == "__main__":
    args = parse_args()
    args.split_index = [0.8, 0.15, 0.05]
    if not os.path.exists('./visualize'):
        os.makedirs('./visualize')
    for dataset in ['pwc_small', 'cora', 'pubmed', 'arxiv_2023', 'ogbn-arxiv', 'citationv8', 'pwc_large', 'pwc_medium']:
        args.name = dataset
        splits, text, data = load_data_lp[dataset](args)
        data, splits = remap_node_indices(data, splits)
        visualize_adjacency_matrix(f'{dataset}_all', data.num_nodes, data.edge_index)
        visualize_adjacency_matrix(f'{dataset}_train', splits['train'].num_nodes,
                                   splits['train'].pos_edge_label_index, splits['train'].neg_edge_label_index)
        visualize_adjacency_matrix(f'{dataset}_valid', splits['valid'].num_nodes,
                                   splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index)
        visualize_adjacency_matrix(f'{dataset}_test',splits['test'].num_nodes,
                                   splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index)
        print(f"Data remapping completed for {dataset}.")

