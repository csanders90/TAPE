# Import the W&B Python Library and log into W&B
# https://github.com/wandb/wandb/issues/5214
import wandb
import os
import sys
from typing import Dict
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization
import numpy as np
import scipy.sparse as ssp
import torch
from utils import (
    get_git_repo_root_path
)
from ogb.linkproppred import Evaluator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import random 
from numba.typed import List
from torch_geometric.utils import to_scipy_sparse_matrix    
from Embedding.node2vec_tagplus import node2vec, data_loader
from yacs.config import CfgNode as CN
from heuristic.eval import (
    get_metric_score,
)
from utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)

wandb.login()

FILE_PATH = get_git_repo_root_path() + '/'

cfg_file = FILE_PATH + "core/configs/pubmed/node2vec.yaml"
# # Load args file
with open(cfg_file, "r") as f:
    args = CN.load_cfg(f)

# Set Pytorch environment
torch.set_num_threads(args.num_threads)

_, _, splits = data_loader[args.data.name](args)
        
# embedding method 
X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
        
print("X_train_index range", X_train_index.max(), X_train_index.min())

def objective(config=None):
    with wandb.init(config=config, settings=wandb.Settings(_service_wait=300)):
        config = wandb.config

        # ust test edge_index as full_A
        full_edge_index = splits['test'].edge_index
        print("full_edge_index", full_edge_index.shape)
        
        # Access individual parameters
        walk_length = 13
        num_walks = 15
        p = config.p
        q = config.q
        
        embed_size = 64
        
        iter = 100
        num_neg_samples = 1
        workers = 10
        epoch = config.epoch
        sg = config.sg 
        hs = config.hs
             
        # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
        adj = to_scipy_sparse_matrix(full_edge_index)
        print(f"adj shape", adj.shape)
        
        embed = node2vec(workers,
                         adj, 
                         embedding_dim=embed_size,
                         walk_length=walk_length, 
                         walks_per_node=num_walks,
                         num_neg_samples=num_neg_samples,
                         p=p,
                         q=q,
                         epoch=epoch,
                         hs=hs,
                         sg=sg)
        
        print(f"embed.shape: {embed.shape}")

        if X_train_index.max() < embed.shape[0]:

            # dot product
            X_train = embed[X_train_index]
            X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
            X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
            # dot product 
            X_test = embed[X_test_index]
            X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
            
            
            clf = LogisticRegression(solver='lbfgs',max_iter=iter, multi_class='auto')
            clf.fit(X_train, y_train)

            acc = clf.score(X_test, y_test)
            print("acc", acc)
            
            y_pred = clf.predict_proba(X_test)
            results_acc = {'node2vec_acc': acc}
            pos_test_pred = torch.tensor(y_pred[y_test == 1])
            neg_test_pred = torch.tensor(y_pred[y_test == 0])
            
            evaluator_hit = Evaluator(name='ogbl-collab')
            evaluator_mrr = Evaluator(name='ogbl-citation2')
            pos_pred = pos_test_pred[:, 1]
            neg_pred = neg_test_pred[:, 1]
            result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
            results_mrr = {'node2vec_mrr': result_mrr}
            print(results_acc, results_mrr)
        

            root = FILE_PATH + 'results'
            acc_file = root + f'/{args.data.name}_acc.csv'
            mrr_file = root +  f'/{args.data.name}_mrr.csv'
            if not os.path.exists(root):
                os.makedirs(root, exist_ok=True)
            append_acc_to_excel(results_acc, acc_file, args.data.name)
            append_mrr_to_excel(results_mrr, mrr_file)

            print(results_acc, '\n', results_mrr)
            wandb.log({"score": acc})
        else:
            wandb.log({"score": 0})
        del result_mrr, results_acc, acc
    return 


# 2: Define the search space
sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        # "walk_length": {"max": 18, "min": 17, 'distribution': 'int_uniform'},
        # "embed_size": {"max": 128, "min": 32, 'distribution': 'int_uniform'},
        "p": {"max": 5, "min": 0.0, 'distribution': 'uniform'},
        "q": {"max": 2, "min": 0.0, 'distribution': 'uniform'},
        # "ws": {"values": [3, 5, 7]},
        # "iter": {"values": [1, 3, 7]},
        # "num_neg_samples": {"values": [1, 3, 5]},
        "epoch": {"max": 20, "min": 5, 'distribution': 'uniform'},
        "sg": {"values": [0, 1]},
        "hs": {"values": [0, 1]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="embedding-sweep")
import pprint

pprint.pprint(sweep_config)
wandb.agent(sweep_id, objective, count=40)

