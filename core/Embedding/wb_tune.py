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
from torch_geometric.graphgym.config import (cfg)
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
import argparse 

FILE_PATH = get_git_repo_root_path() + '/'


def set_cfg(FILE_PATH, args):
    with open(FILE_PATH + args.cfg_file, "r") as f:
        cfg = CN.load_cfg(f)
    return cfg

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default = "core/configs/cora/node2vec.yaml",
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default = "core/configs/cora/sweep.yaml",
                        help='The configuration file path.')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

args = parse_args()
# Load args file

print(args)
cfg = set_cfg(FILE_PATH, args)
cfg.merge_from_list(args.opts)


# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)

_, _, splits = data_loader[cfg.data.name](cfg)
        
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
        walk_length = config.wl
        num_walks = config.num_walks
        p = config.p
        q = config.q
        
        embed_size =  cfg.model.node2vec.embed_size # config.emb_size
        max_iter = cfg.model.node2vec.max_iter
        num_neg_samples = cfg.model.node2vec.num_neg_samples
        workers = cfg.model.node2vec.workers

        if config.p > config.q:
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
                            q=q
                            )
            
            print(f"embed.shape: {embed.shape}")

            if X_train_index.max() < embed.shape[0]:

                # dot product
                X_train = embed[X_train_index]
                X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
                X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
                # dot product 
                X_test = embed[X_test_index]
                X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
                
                
                clf = LogisticRegression(solver='lbfgs',max_iter=max_iter, multi_class='auto')
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
                acc_file = root + f'/{cfg.data.name}_acc.csv'
                mrr_file = root +  f'/{cfg.data.name}_mrr.csv'
                if not os.path.exists(root):
                    os.makedirs(root, exist_ok=True)
                append_acc_to_excel(results_acc, acc_file, cfg.data.name)
                append_mrr_to_excel(results_mrr, mrr_file)

                print(results_acc, '\n', results_mrr)
                wandb.log({"score": acc})
                # save selected files
                wandb.log_code(include_fn=wandb_record_files)
            else:
                wandb.log({"score": 0})
        else:
            wandb.log({"score": 0})
        
    print("acc", acc)
    return 

def wandb_record_files(path):
    record_or_not = False
    record_lst = [args.sweep_file, 
                  args.cfg_file, 
                  __file__,
                  ]
    
    for recorded in record_lst:
        if recorded in path:
            record_or_not = True
            break
    return record_or_not




import yaml

with open(FILE_PATH + args.sweep_file, "r") as yaml_file:
    # Load the YAML content into a Python dictionary
    sweep_config = yaml.safe_load(yaml_file)
    
# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=f"embedding-sweep-{cfg.data.name}")
import pprint

pprint.pprint(sweep_config)
wandb.agent(sweep_id, objective, count=60)
