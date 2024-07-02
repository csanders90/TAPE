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
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
# Assuming other necessary imports from your script
from graphgps.utility.utils import (
    set_cfg, parse_args, get_git_repo_root_path, Logger, custom_set_out_dir,
    custom_set_run_dir, set_printing, run_loop_settings, create_optimizer,
    config_device, init_model_from_pretrained, create_logger, use_pretrained_llm_embeddings
)
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from heuristic.eval import get_metric_score
from graphgps.lm_trainer.tfidf_trainer import Trainer_TFIDF
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
import argparse
import wandb
from pdb import set_trace as st 

FILE_PATH = f'{get_git_repo_root_path()}/'

yaml_file = {   
             'tfidf': 'core/yamls/cora/lms/tfidf.yaml',
             'word2vec': 'core/yamls/cora/lms/word2vec.yaml',
             'bert': 'core/yamls/cora/gcns/heart_gnn_models.yaml',
            }

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='pubmed',
                        help='data name')
    parser.add_argument('--device', dest='device', required=False, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=False,
                        default=400,
                        help='data name')
    parser.add_argument('--embedder', dest='embedder_type', type=str, required=False,
                        default='tfidf',
                        help='word embedding method')
    parser.add_argument('--score', dest='score', type=str, required=False, default='mlp_score',
                        help='decoder name')
    parser.add_argument('--repeat', type=int, default=3,
                        help='The number of repeated jobs.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


def project_main(): 
    # process params
    args = parse_args()
    args.cfg_file = yaml_file[args.embedder_type]
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data
    
    cfg.data.device = args.device
    cfg.decoder.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch
    cfg.embedder.type = args.embedder_type
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    custom_set_out_dir(cfg, args.cfg_file, cfg.wandb.name_tag)
    # torch.set_num_threads(20)
    loggers = create_logger(args.repeat)
    for run_id, seed, split_index in zip(*run_loop_settings(cfg, args)):
        print(f'run id : {run_id}')
        # Set configurations for each run TODO clean code here 
        train_dataset = torch.load(f'./generated_dataset/{cfg.data.name}/{cfg.embedder.type}_{seed}_train_dataset.pt')
        train_labels = torch.load(f'./generated_dataset/{cfg.data.name}/{cfg.embedder.type}_{seed}_train_labels.pt')
        val_dataset = torch.load(f'./generated_dataset/{cfg.data.name}/{cfg.embedder.type}_{seed}_val_dataset.pt')
        val_labels = torch.load(f'./generated_dataset/{cfg.data.name}/{cfg.embedder.type}_{seed}_val_labels.pt')
        test_dataset = torch.load(f'./generated_dataset/{cfg.data.name}/{cfg.embedder.type}_{seed}_test_dataset.pt')
        test_labels = torch.load(f'./generated_dataset/{cfg.data.name}/{cfg.embedder.type}_{seed}_test_labels.pt')

        clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
        clf.fit(train_dataset, train_labels)
        
        test_pred = clf.predict(test_dataset)
        test_acc = sum(np.asarray(test_labels) == test_pred ) / len(test_labels)    
        y_pos_pred, y_neg_pred = torch.tensor(test_pred[test_labels == 1]), torch.tensor(test_pred[test_labels == 0])
        test_metrics = get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred)
        test_metrics.update({'ACC': round(test_acc, 4)})
        
        train_pred = clf.predict(train_dataset)
        train_acc = sum(np.asarray(train_labels) == train_pred ) / len(train_labels)    
        y_pos_pred, y_neg_pred = torch.tensor(train_pred[train_labels == 1]), torch.tensor(train_pred[train_labels == 0])
        train_metrics = get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred)
        train_metrics.update({'ACC': round(train_acc, 4)})

        val_pred = clf.predict(val_dataset)
        val_acc = sum(np.asarray(val_labels) == val_pred ) / len(val_labels)    
        y_pos_pred, y_neg_pred = torch.tensor(val_pred[val_labels == 1]), torch.tensor(val_pred[val_labels == 0])
        val_metrics = get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred)
        val_metrics.update({'ACC': round(val_acc, 4)})
        
        print(f'Accuracy: {test_acc:.4f}')
        print(f'metrics : {test_metrics}')

        results_rank = {
            key: (test_metrics[key], train_metrics[key], val_metrics[key])
            for key in test_metrics.keys()
        }
        
        for key, result in results_rank.items():
            loggers[key].add_result(run_id, result)
        st()

        
        for key in results_rank.keys():
           print(loggers[key].calc_run_stats(run_id))
        
    for key in results_rank.keys():
        print(loggers[key].calc_all_stats())
        
    # clf = MLPClassifier(random_state=1, max_iter=300).fit(train_dataset, train_labels)
        # test_proba = clf.predict_proba(test_dataset)
        # test_pred = clf.predict(test_dataset)
        # acc = clf.score(test_dataset, test_labels)
if __name__ == '__main__':
    project_main()
