import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from itertools import product
from graphgps.network.gsaint import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from graphgps.train.opt_train import Trainer, Trainer_Saint
from torch_geometric.graphgym.cmd_args import parse_args
from data_utils.load import load_data_lp
from torch_geometric import seed_everything
from torch_geometric.data.makedirs import makedirs
from torch_geometric.graphgym.utils.device import auto_select_device
from custom_main import run_loop_settings, custom_set_run_dir, set_printing
from graphgps.network.custom_gnn import create_model
from utils import set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir \
    , custom_set_run_dir, set_printing, run_loop_settings, create_optimizer, config_device, \
        init_model_from_pretrained, create_logger

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph
import matplotlib.patches as mpatches

# def to_network(data):
#     G = nx.Graph()
#     row, col = data.edge_index.numpy()
#     G.add_edges_from(zip(row, col), color='black')

#     node_color = {node: 'lightgray' for node in range(data.num_nodes)}

#     pos_row, pos_col = data.pos_edge_label_index.numpy()
#     for u, v in zip(pos_row, pos_col):
#         if G.has_edge(u, v):
#             G[u][v]['color'] = 'green'
#             node_color[u] = 'green'
#             node_color[v] = 'green'

#     neg_row, neg_col = data.neg_edge_label_index.numpy()
#     for u, v in zip(neg_row, neg_col):
#         if not G.has_edge(u, v):
#             node_color[u] = 'red' if node_color[u] != 'green' else node_color[u]
#             node_color[v] = 'red' if node_color[v] != 'green' else node_color[v]

#     for node in G.nodes():
#         G.nodes[node]['color'] = node_color[node]

#     return G

def to_network(subgraph_nodes, test_data):
    # Extract the subgraph using the indices from subgraph_nodes
    sub_edge_index, mapping = subgraph(subgraph_nodes, test_data.edge_index, relabel_nodes=True)
    
    # Create a graph based on the subgraph
    G = nx.Graph()

    # Prepare node colors
    node_color = {}
    pos_nodes = test_data.pos_edge_label_index.tolist()[0]
    neg_nodes = test_data.neg_edge_label_index.tolist()[0]
    
    # Отображаем индексы из исходного графа в индексы подграфа и устанавливаем цвета
    for idx, original_idx in enumerate(subgraph_nodes.tolist()):
        if original_idx in pos_nodes:
            node_color[idx] = 'green'
        elif original_idx in neg_nodes:
            node_color[idx] = 'red'
        else:
            node_color[idx] = 'lightgray'
    
    isolated_nodes = list()
    # Добавляем ребра, только если оба узла зеленые
    for u, v in sub_edge_index.t().tolist():
        if node_color[u] == 'green' and node_color[v] == 'green':
            G.add_edge(u, v)
        else:
            if node_color[u] == 'red' or node_color[u] == 'lightgray':
                isolated_nodes.append(u)
            if node_color[v] == 'red' or node_color[v] == 'lightgray':
                isolated_nodes.append(v)
    
    # Add isolated nodes to the graph
    
    for node in isolated_nodes:
        G.add_node(node)

    return G, node_color


def plot_graph(G, node_color, idx):
    # Set up the plot
    plt.figure(figsize=(10, 10))

    # Determine positions for all nodes using the Spring layout
    pos = nx.spring_layout(G, seed=42)  # Using a seed for reproducibility

    # Extract node colors from the node_color dictionary
    colors = [node_color[node] for node in G.nodes()]

    # Draw the nodes with colors based on the node_color mapping
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=10)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edge_color='gray')  # Assuming a generic color for all edges

    # Create a legend with custom handles
    legend_handles = [
        mpatches.Patch(color='red', label='Negative Relationship'),
        mpatches.Patch(color='green', label='Positive Relationship'),
        mpatches.Patch(color='lightgray', label='Potential Candidates')
    ]
    plt.legend(handles=legend_handles)

    # Save the figure, incorporating the index to distinguish files
    plt.savefig(f'sampled_subgraph_{idx}.png')
    plt.close()  # Close the figure to free up memory

def get_loader(data, batch_size, walk_length, num_steps, sample_coverage):
    return GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=walk_length, num_steps=num_steps, sample_coverage=sample_coverage)

if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    torch.set_num_threads(cfg.num_threads)
    # Best params: {'batch_size': 64, 'walk_length': 10, 'num_steps': 30, 'sample_coverage': 100, 'accuracy': 0.82129}
    batch_sizes = [64]#[8, 16, 32, 64]
    walk_lengths = [10]#[10, 15, 20]
    num_steps = [30]#[10, 20, 30]
    sample_coverages = [100]#[50, 100, 150]

    loggers = create_logger(args.repeat)
    
    for batch_size, walk_length, num_steps, sample_coverage in product(batch_sizes, walk_lengths, num_steps, sample_coverages):
        for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)): # In run_loop_settings we should send 2 parameeters

            # Set configurations for each run
            custom_set_run_dir(cfg, run_id)
            set_printing(cfg) # We should send cfg else we get error Attribute error: run_dir
            cfg.seed = seed
            cfg.run_id = run_id
            seed_everything(cfg.seed)
            auto_select_device()
            splits, text = load_data_lp[cfg.data.name](cfg.data)

            lst_args = cfg.model.type.split('_')
            if lst_args[0] == 'gsaint':
                sampler = get_loader(splits['test'], 
                                batch_size=64, 
                                walk_length=10, 
                                num_steps=30, 
                                sample_coverage=100)

                cfg.model.type = lst_args[1].upper() # Convert to upper case
            else:
                sampler = None 
            print(splits['test'])
            for idx in range(len(sampler)-25):
                G, node_color = to_network(sampler[idx][0], splits['test'])
                plot_graph(G, node_color, idx)