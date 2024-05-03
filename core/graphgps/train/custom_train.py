import os
import sys
# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import time
import logging
import wandb

from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.register import register_train
from sklearn.metrics import roc_auc_score, \
        average_precision_score, \
        roc_curve, auc


from heuristic.eval import get_metric_score
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
from graphgps.loss.custom_loss import RecLoss
from embedding.tune_utils import param_tune_acc_mrr
from utils import config_device

class Trainer():
    def __init__(self, 
                 FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits,
                 ):
        
        self.device = config_device(cfg)
        self.model = model.to(self.device)
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        
        self.train_func = {
        'gae': self._train_gae,
        'vgae': self._train_vgae, 
        'GAT': self._train_gae,
        'GraphSage': self._train_gae,
        'GNNStack': self._train_gae
        }
        
        self.test_func = {
            'GAT': self._test,
            'gae': self._test,
            'vgae': self._test,
            'GraphSage': self._test,
            'GNNStack': self._test
        }
        
        self.evaluate_func = {
            'GAT': self._evaluate,
            'gae': self._evaluate,
            'vgae': self._evaluate,
            'GraphSage': self._evaluate,
            'GNNStack': self._evaluate
        }
        
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

    
    def _train_gae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        # loss = self.model.recon_loss(pred, self.train_data.edge_label)
        loss_func = RecLoss()
        loss = loss_func(z, self.train_data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_vgae(self):
        """Training the VGAE model, the loss function consists of reconstruction loss and kl loss"""
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss() # add kl loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        best_hits, best_auc = 0, 0
        for epoch in range(1, self.epochs + 1):

            loss = self.train_func[self.model_name]()
            if epoch % 100 == 0:
                auc, ap, acc = self.test_func[self.model_name]()
                result_mrr = self.evaluate_func[self.model_name]()
                print('Epoch: {:03d}, Loss_train: {:.4f}, AUC: {:.4f}, \
                      AP: {:.4f}, Hits@100: {:.4f}, MRR{:.4f}'.format(epoch, loss, auc, ap, acc, result_mrr['Hits@100']))
                if auc > best_auc:
                    best_auc = auc 
                elif result_mrr['Hits@100'] > best_hits:
                    best_hits = result_mrr['Hits@100']
        return best_auc, best_hits, result_mrr

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encoder(self.test_data.x, self.test_data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1)) # positive samples
        neg_y = z.new_zeros(neg_edge_index.size(1)) # negative samples
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr

        
    @torch.no_grad()
    def _test(self):
        """test"""
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
        self.model.eval()

        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encoder(self.test_data.x, self.test_data.edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1)) # positive samples
        neg_y = z.new_zeros(neg_edge_index.size(1)) # negative samples
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        auc = auc(fpr, tpr)
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc
    
        

    def save_result(self, results_dict):

        root = self.FILE_PATH + cfg.out_dir
        acc_file = root + f'/{self.model_name}_acc_mrr.csv'

        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        id = wandb.util.generate_id()
        param_tune_acc_mrr(id, results_dict, acc_file, self.data_name, self.model_name)
   
# TODO integrate my trainer to train module wandb? docu?

