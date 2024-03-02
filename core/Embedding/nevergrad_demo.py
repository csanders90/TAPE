import numpy as np
import nevergrad as ng
# Import the W&B Python Library and log into W&B
import wandb

import sys
from typing import Dict
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization
import numpy as np
import torch

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
from IPython import embed

# Let us define a function.
def myfunction(arg1, arg2, arg3, value=3):
    return np.abs(value) + (1 if arg1 != "a" else 0) + (1 if arg2 != "e" else 0)

def objective(ws, 
            walk_length,
            p, 
            q, 
            num_walks, 
            embed_size, 
            iter, 
            num_neg_samples):
    
    cfg_file = "core/configs/pubmed/node2vec.yaml"
    # # Load args file
    with open(cfg_file, "r") as f:
        args = CN.load_cfg(f)
    
    # Set Pytorch environment
    torch.set_num_threads(args.num_threads)
    
    _, _, splits = data_loader[args.data.name](args)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    
    # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
    adj = to_scipy_sparse_matrix(full_edge_index)

    embed = node2vec(adj, 
                        embedding_dim=embed_size,
                        walk_length=walk_length, 
                        walks_per_node=num_walks,
                        workers=8, 
                        window_size=ws, 
                        num_neg_samples=num_neg_samples,
                        p=p,
                        q=q)
    print(embed.shape)
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
    print("acc", score)
    return 1-score


window_size = ng.p.Choice([3, 5, 7, 9])
walk_length = ng.p.Choice([20, 40, 60, 80])
p = ng.p.Choice([0.2, 0.5, 0.7, 0.9])
q = ng.p.Choice([0.2, 0.5, 0.7, 0.9])
num_walks = ng.p.Choice( [40, 60, 80])
embed_size = ng.p.Choice([32, 64, 128, 256])
ws = ng.p.Choice([3, 7, 9, 11])
iter = ng.p.Choice([100, 200, 300, 400])
num_neg_samples = ng.p.Choice([1, 3, 5])

# create the parametrization
# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
instru = ng.p.Instrumentation(ws, 
                            walk_length,
                            p, 
                            q, 
                            num_walks, 
                            embed_size, 
                            iter, 
                            num_neg_samples)

print(instru.dimension)  # 5 dimensional space
print(instru.args, instru.kwargs)
objective(*instru.args)

budget = 4# How many episode we will do before concluding.
for name in ["RandomSearch"]:
    optim = ng.optimizers.registry[name](parametrization=instru, budget=budget)
    for u in range(budget//2):
        x1 = optim.ask()
        x2 = optim.ask()
        y1 = objective(*x1.args)
        y2 = objective(*x2.args)
        optim.tell(x1, y1)
        optim.tell(x2, y2)
    recommendation = optim.recommend()
    print("* ", name, " provides a vector of parameters with test error ",
            myfunction(*recommendation.args, **recommendation.kwargs))
    print("with params ", recommendation.kwargs, recommendation.args)