import os
import sys

# Add parent directory to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
This script implements the compatibility matrix method with and without negative links,
as described in the paper "NETINFOF FRAMEWORK: MEASURING AND EXPLOITING NETWORK USABLE INFORMATION".
"""
import os
import sys
import torch
import numpy as np
import scipy.sparse as ssp
import matplotlib.pyplot as plt
from pathlib import Path
from IPython import embed
import wandb
from scipy.spatial import distance
from sklearn.neural_network import MLPClassifier
from yacs.config import CfgNode as CN
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)
from data_utils.load_cora_lp import get_cora_casestudy
from data_utils.load_arxiv2023_lp import get_raw_text_arxiv_2023
from data_utils.load_pubmed_lp import get_pubmed_casestudy
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj
from heuristic.eval import get_metric_score
from embedding.tune_utils import parse_args, param_tune_acc_mrr
import torch
from time import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm 
from textfeat.mlp_dot_product import distance_metric, data_loader, FILE_PATH, set_cfg
import torch.nn.functional as F
method = 'compat'
LOG_FREQ = 10 

norm = lambda x: (x-x.mean())/ x.std()

def create_node_feat(edge_label_index, train_node_feat):
    node_feat = torch.zeros(edge_label_index.shape[0], train_node_feat.shape[1]*2)
    for i, (srt, trg) in enumerate(edge_label_index):
        node_feat[i, :] = torch.cat((train_node_feat[srt], train_node_feat[trg]), 0)
    
    return node_feat


class CompatTrainer():

    def __init__(self, 
                 cfg, 
                 ):
        # self.seed = cfg.seed
        # self.device = cfg.device
        self.dataset_name = cfg.data.name
        self.model_name = 'comp'
        
        self.lr = 0.001
        self.epochs = 200
        self.batch_size = 256
        self.device = 'cpu'

        # preprocess data
        dataset, data_cited, splits = data_loader[cfg.data.name](cfg)
        
        # use test edge_index as full_A
        full_edge_index = splits['test'].edge_index
        full_edge_weight = torch.ones(full_edge_index.size(1))
        num_nodes = dataset._data.num_nodes
        
        m = construct_sparse_adj(full_edge_index)
        plot_coo_matrix(m, f'test_edge_index.png')


        self.features = splits['train'].x
        
        self.train_links = splits['train'].edge_label_index
        self.train_labels = splits['train'].edge_label
        self.valid_links = splits['valid'].edge_label_index 
        self.valid_labels = splits['valid'].edge_label
        self.test_links = splits['test'].edge_label_index
        self.test_labels = splits['test'].edge_label
        # self.
        self.test_loader = DataLoader(range(self.test_links.shape[1]), self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(range(self.valid_links.shape[1]), self.batch_size, shuffle=False)
        self.train_loader = DataLoader(range(self.train_links.shape[1]), self.batch_size, shuffle=False)
        
        num_feat = self.features.shape[1]
        # models
        self.model = model(num_feat, num_feat).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.loss_func = torch.nn.MSELoss()
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')


    def _forward(self, x1,):
        logits = self.model(x1)  # small-graph
        return logits


    def _train(self):
        # ! Shared
        self.model.train()
        for batch_count, indices in enumerate(tqdm(self.train_loader)):
            # do node level things
            emb = (self.features.to(self.device))
        
            self.optimizer.zero_grad()
            # ! Specific
            links_indices = self.train_links[:, indices]
            emb_right = self._forward(emb[links_indices[0]])
            
            loss = self.loss_func(
                emb_right, emb[links_indices[1]])

            loss.backward()
            self.optimizer.step()
            del emb, batch_count, indices 
        return loss.item()


    def _link_predictor(self, emb1, emb2):
        """ calc the similarity between two node embeddings 
        # previous version is
        # def _link_predictor(self, emb1, emb2):
        #   return F.cosine_similarity(norm(emb1), norm(emb2), dim=0)
        """
        link_pred = np.zeros(emb1.shape[0])
        for index in range(emb1.shape[0]):
            link_pred[index] = distance_metric['dot'](emb1[index], emb2[index])
        return torch.tensor(link_pred)
    
    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        emb = (self.features.to(self.device))
        train_pred_list = []
        for _, indices in enumerate(tqdm(self.train_loader)):
            links_indices = self.train_links[:, indices]
            
            emb_right = self._forward(emb[links_indices[0]])
            train_pred_list.append(self._link_predictor(emb_right, emb[links_indices[1]]))
            
        predictions = torch.cat(train_pred_list)
        # evaluation refer to paper section
        y_pos_pred = predictions[self.train_labels == 1]
        y_neg_pred = predictions[self.train_labels == 0]
        train_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, y_pos_pred, y_neg_pred)

        return train_mrr

    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss = self._train()
            
            if epoch % LOG_FREQ == 0:
                train_mrr = self._evaluate()
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Train_acc: {0:.4f}, Train_mrr: {train_mrr}, ValAcc: {0:.4f}, ES: {0}')

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, val_mrr = self._evaluate()
        print(
            f'[{self.model_name}] ValAcc: {val_acc:.4f}, \n')
        return 


import torch

class model(torch.nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim):
        super(model, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.softmax = torch.nn.Softmax()

    def forward(self, x1):
        x = self.linear1(x1)
        x = self.softmax(x)
        return x

      
        
def MLP_as_compat(cfg, method='node_sim', predict='soft', data=None):
    # train loop
    
    raise NotImplementedError 
    if predict == 'hard':
        y_pos_pred = torch.tensor(clf.predict(X_pos_test_feat))
        y_neg_pred = torch.tensor(clf.predict(X_neg_test_feat))
    elif predict == 'soft':
        y_pos_pred = torch.tensor(clf.predict_proba(X_pos_test_feat)[:, 1])
        y_neg_pred = torch.tensor(clf.predict_proba(X_neg_test_feat)[:, 1])
    
    test_pos_label = splits['test'].edge_label[splits['test'].edge_label == 1]
    test_neg_label = splits['test'].edge_label[splits['test'].edge_label == 0]
    
    acc_pos = clf.score(X_pos_test_feat, test_pos_label)
    acc_neg = clf.score(X_neg_test_feat, test_neg_label)
    results_acc = {f'{method}_acc': (acc_pos+ acc_neg)/2}
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    weight_matrices = clf.coefs_
    
    plt.figure()
    plt.imshow(weight_matrices[0].T)
    id = wandb.util.generate_id()
    plt.savefig(f'MLP_{cfg.data.name}visual_{id}.png')

    results_acc.update(get_metric_score(evaluator_hit, evaluator_mrr, y_pos_pred, y_neg_pred))
    return results_acc


def eval_node_sim_acc(cfg) -> None:
    """load text attribute graph in link predicton setting
    """
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    result_dict = {}

    input_dim = 1000 
    hidden_dim = 256
    clf = CompatTrainer(cfg)
    # train loop 
    clf.train()
    # evaluate loop 
    exit(-1)
    

    root = FILE_PATH + 'results/node_sim/'
    acc_file = root + f'{cfg.data.name}_acc_{method}.csv'

    if not os.path.exists(root):
        Path(root).mkdir(parents=True, exist_ok=False)
    
    id = wandb.util.generate_id()
    param_tune_acc_mrr(id, result_dict, acc_file, cfg.data.name, method)
    
    return result_dict


if __name__ == "__main__":
    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    result = eval_node_sim_acc(cfg)
    print(result)
