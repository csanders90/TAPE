
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import pprint
import numpy as np
import scipy.sparse as ssp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from numba.typed import List
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN
from heuristic.eval import get_metric_score
from ogb.linkproppred import Evaluator
import argparse
from utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel,
)

# Constants
FILE_PATH = get_git_repo_root_path() + '/'

def set_cfg(file_path, args):
    with open(file_path + args.cfg_file, "r") as f:
        return CN.load_cfg(f)

def load_sweep_config(file_path):
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)

def print_args(args):
    print(args)

def initialize_config(args):
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)
    torch.set_num_threads(cfg.num_threads)
    
    return cfg

# TODO how to save wandb files 

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default = "core/configs/cora/node2vec.yaml",
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default = "core/configs/cora/sweep2.yaml",
                        help='The configuration file path.')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()

def process_edge_index(full_edge_index):
    print("full_edge_index", full_edge_index.shape)
    return to_scipy_sparse_matrix(full_edge_index)

from IPython import embed
def save_struc2vec_acc_excel(uuid_val, metrics_acc, root, name, method):
    # if not exists save the first row
    
    csv_columns = ['Metric'] + list(k for k in metrics_acc) 

    # load old csv
    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)
    
    # create new line 
    acc_lst = []
    
    for k, v in metrics_acc.items():
        acc_lst.append(process_value(v))
        
    # merge with old lines, 
    v_lst = [f'{name}_{uuid_val}_{method}'] + acc_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    # best value
    highest_values = new_Data.apply(lambda column: max(column, default=None))

    # concat and save
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)

    return upt_Data


def save_struc2vec_mrr_excel(uuid_val, metrics_mrr, root, method):
 
    csv_columns, csv_numbers = [], []
    for i, (k, v) in enumerate(metrics_mrr.items()): 
        if i == 0:
            csv_columns = ['Metric'] + list(v.keys())
        csv_numbers.append([f'{k}_{uuid_val}_{method}'] + list(v.values()))
    
    print(csv_numbers)

    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    
    new_df = pd.DataFrame(csv_numbers, columns = csv_columns)
    new_Data = pd.concat([Data, new_df])
    
    highest_values = new_Data.apply(lambda column: max(column, default=None))
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=csv_columns)
    upt_Data = pd.concat([new_Data, Best_df])
    
    upt_Data.to_csv(root, index=False)

    
    return upt_Data

def train_and_evaluate_logistic_regression(id, X_train, y_train, X_test, y_test, max_iter, method):
    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='auto')
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

    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root + f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    append_acc_to_excel(id, results_acc, acc_file, cfg.data.name, method)
    append_mrr_to_excel(id, results_mrr, mrr_file, method)

    print(results_acc, '\n', results_mrr)
    return acc