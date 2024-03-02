# Import the W&B Python Library and log into W&B
import wandb

# Import objective relevant libraries
import os
import sys
from typing import Dict
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization
import numpy as np
import scipy.sparse as ssp
import torch

import matplotlib.pyplot as plt
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from heuristic.eval import (
    evaluate_auc,
    evaluate_hits,
    evaluate_mrr,
    get_metric_score,
    get_prediction
)

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import random 
from numba.typed import List
from torch_geometric.utils import to_scipy_sparse_matrix    
from Embedding.node2vec_tagplus import node2vec, set_cfg, data_loader
from yacs.config import CfgNode as CN

wandb.login()

# 1: Define objective/training function
def objective(config):
    with wandb.init(config=config):
        config = wandb.config
        
        cfg_file = "configs/pubmed/node2vec.yaml"
        # # Load args file
        with open(cfg_file, "r") as f:
            args = CN.load_cfg(f)
        

        # Set Pytorch environment
        torch.set_num_threads(args.num_threads)
        
        dataset, data_cited, splits = data_loader[args.data.name](args)
        
        # ust test edge_index as full_A
        full_edge_index = splits['test'].edge_index
        full_edge_weight = torch.ones(full_edge_index.size(1))
        num_nodes = dataset._data.num_nodes
        
        m = construct_sparse_adj(full_edge_index)
        plot_coo_matrix(m, f'test_edge_index.png')
        
        full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

        # Access individual parameters
        walk_length = config.walk_length
        # `num_walks = args.model.node2vec.num_walks` is extracting the number of walks to be performed
        # for each node in the Node2Vec algorithm from the argsuration settings. This parameter controls
        # the number of random walks to be generated starting from each node in the graph during the node
        # embedding generation process.
        num_walks = config.num_walks
        p = config.p
        q = config.q
        
        embed_size = config.embed_size
        ws = config.window_size
        iter = config.iter
    
        # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
        adj = to_scipy_sparse_matrix(full_edge_index)

        embed = node2vec(adj, embedding_dim=embed_size, walk_length=walk_length, walks_per_node=num_walks,
                  workers=8, window_size=ws, num_neg_samples=1, p=p, q=q)
    
        # TODO different methods to generate node embeddings
        # embedding method 
        X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
        # dot product
        X_train = embed[X_train_index]
        X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
        X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
        # dot product 
        X_test = embed[X_test_index]
        X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
        
        
        clf = LogisticRegression(solver='lbfgs',max_iter=iter, multi_class='auto')
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: Define the search space
sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "walk_length": {"max": 30, "min": 5, 'distribution': 'int_uniform'},
        "num_walks": {"values": [40, 60, 80]},
        "embed_size": {"max": 128, "min": 32, 'distribution': 'int_uniform'},
        "window_size": {"max": 10, "min": 2, 'distribution': 'int_uniform'},
        "p": {"max": 1, "min": 0.1, 'distribution': 'uniform'},
        "q": {"max": 1, "min": 0.1, 'distribution': 'uniform'},
        "ws": {"values": [3, 7, 9, 11]},
        "iter": {"values": [1, 3, 7]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="embedding-sweep")
import pprint

pprint.pprint(sweep_config)
wandb.agent(sweep_id, function=main, count=20)