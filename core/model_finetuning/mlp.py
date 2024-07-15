import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
# Assuming other necessary imports from your script
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
from graph_embed.tune_utils import mvari_str2csv
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from ogb.linkproppred import Evaluator
from heuristic.eval import get_metric_score
from sklearn.neural_network import MLPClassifier
import argparse
import wandb
from torch_geometric import seed_everything
from pdb import set_trace as st 
import time 
from create_dataset import create_tfidf
from yacs.config import CfgNode as CN
from graphgps.utility.utils import get_git_repo_root_path

FILE_PATH = f'{get_git_repo_root_path()}/'

yaml_file = {   
             'tfidf': 'core/yamls/cora/lms/tfidf.yaml',
             'w2v': 'core/yamls/cora/lms/tfidf.yaml',
            }

split_index_data = {
    'pwc_small': 1,
    'cora': 1,
    'pubmed': 1,
    'arxiv_2023': 1,
    'pwc_medium': 0.5,
    'citationv8': 0.5,
    'ogbn_arxiv': 0.5,
    'pwc_large': 0.5
}

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='cora',
                        help='data name')
    parser.add_argument('--device', dest='device', required=False, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=400,
                        help='data name')
    parser.add_argument('--embedder', dest='embedder', type=str, required=False,
                        default='w2v',
                        help='word embedding method')
    parser.add_argument('--decoder', dest='decoder', type=str, required=False,
                        default='MLP',
                        help='word embedding method')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--max_iter', dest='max_iter', type=int, required=False, default=1000,
                        help='decoder name')

    parser.add_argument('--repeat', type=int, default=5,

                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def get_metrics(clf, dataset, labels, evaluator_hit, evaluator_mrr):
    # Predict and calculate accuracy
    pred = clf.predict(dataset)
    labels = np.asarray(labels)
    acc = np.mean( labels== pred)
    
    # Calculate positive and negative predictions
    y_pos_pred = torch.tensor(pred[labels == 1])
    y_neg_pred = torch.tensor(pred[labels == 0])
    
    # Get additional metrics
    metrics = get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred)
    metrics.update({'ACC': round(acc, 4)})
    
    return metrics


def project_main(): 
    # process params
    args = parse_args()
    args.cfg_file = yaml_file[args.embedder]
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data

    cfg.data.device = args.device
    cfg.decoder.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    cfg.embedder.type = args.embedder
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    cfg.data.scale = split_index_data[args.data]

    loggers = create_logger(args.repeat)
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        
        print(f'run id : {run_id}, seed: {seed}, split_index: {split_index}')
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels = create_tfidf(cfg, seed)
            
        print(f"loaded dataset")
        clf = MLPClassifier(random_state=run_id, max_iter=args.max_iter)
        print(f"created model")
        
        clf.fit(train_dataset, train_labels)  

        # Calculate and print metrics for test set
        test_metrics = get_metrics(clf, test_dataset, test_labels, evaluator_hit, evaluator_mrr)
        print(test_metrics)
        # Calculate and print metrics for train set
        train_metrics = get_metrics(clf, train_dataset, train_labels, evaluator_hit, evaluator_mrr)
        print(train_metrics)
        # Calculate and print metrics for validation set
        val_metrics = get_metrics(clf, val_dataset, val_labels, evaluator_hit, evaluator_mrr)
        print(val_metrics)

        results_rank = {
            key: (train_metrics[key], val_metrics[key], test_metrics[key])
            for key in test_metrics.keys()
        }

        for key, result in results_rank.items():
            loggers[key].add_result(run_id, result)

        for key in results_rank:
            print(key, loggers[key].results)

    for key in results_rank.keys():
        print(loggers[key].calc_all_stats())

    cfg.out_dir = 'results/tfidf'
    root = os.path.join(FILE_PATH, cfg.out_dir)
    acc_file = os.path.join(root, f'{cfg.data.name}_lm_mrr.csv')

    run_result = {}
    for key in loggers.keys():
        print(key)
        _, _, _, test_bvalid, _, _ = loggers[key].calc_all_stats(True)
        run_result[key] = test_bvalid
    
    os.makedirs(root, exist_ok=True)
    name_tag = cfg.wandb.name_tag = f'{cfg.data.name}_run{run_id}_{args.embedder}{args.max_iter}_{cfg.data.scale}'
    mvari_str2csv(name_tag, run_result, acc_file)
    
    
if __name__ == '__main__':
    project_main()
