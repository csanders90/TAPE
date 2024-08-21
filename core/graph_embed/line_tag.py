
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ge.classify import read_node_label, Classifier
from ge.models.line_tf import LINE
import argparse
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

# Third-party library imports
import numpy as np
from sklearn.linear_model import LogisticRegression


# External module imports
import torch
import matplotlib.pyplot as plt
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
import itertools 
import scipy.sparse as ssp

from heuristic.eval import get_metric_score
from data_utils.load import load_graph_lp as data_loader
from core.data_utils.graph_stats import plot_coo_matrix, construct_sparse_adj
from data_utils.load_data_lp import get_edge_split
from core.graph_embed.tune_utils import (
    get_git_repo_root_path,
    param_tune_acc_mrr
)
from graphgps.utility.utils import (
    set_cfg,
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)
import wandb 


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')

    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
   
    return parser.parse_args()

if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'
    args = parse_args()
    
    cfg = set_cfg(FILE_PATH, args.cfg_file)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    if torch.cuda.is_available():
        # Get the number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()

        if num_cuda_devices > 0:
            # Set the first CUDA device as the active device
            torch.cuda.set_device(0)
            device = 'cuda'
    else:
        device = 'cpu'
        
    max_iter = cfg.model.line.max_iter
    dataset, _ = data_loader[cfg.data.name](cfg)
    undirected = dataset.is_undirected()
    splits = get_edge_split(dataset,
                            undirected,
                            cfg.data.device,
                            cfg.data.split_index[1],
                            cfg.data.split_index[2],
                            cfg.data.include_negatives,
                            cfg.data.split_labels
                            )
    
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 
    
    result_dict = {}
    # Access individual parameters

    adj = to_scipy_sparse_matrix(full_edge_index)

    G = nx.from_scipy_sparse_array(adj)
    model = LINE(G, embedding_size=96, order='all', lr=0.01) 
    model.train(batch_size=2048, epochs=8, verbose=2)

    embed = model.get_embeddings()
    print(embed.shape)
    
    print(f"embedding size {embed.shape}")

    # embedding method 
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    

    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)

    acc = clf.score(X_test, y_test)

    method = cfg.model.type
    
    plt.figure()
    plt.plot(y_pred, label='pred')
    plt.plot(y_test, label='test')
    plt.savefig(f'{method}_pred.png')
        
    results_acc = {f'{method}_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    pos_pred = pos_test_pred[:, 1]
    neg_pred = neg_test_pred[:, 1]
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred)
    result_mrr['ACC'] = acc
    results_mrr = {f'{method}_mrr': result_mrr}

    print(results_acc)

    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root +  f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    id = wandb.util.generate_id()
    append_acc_to_excel(id, results_acc, acc_file, cfg.data.name, method)
    append_mrr_to_excel(id, results_mrr, mrr_file, cfg.data.name, method)
    


