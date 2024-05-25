import os
import sys
import os
import sys
from os.path import abspath, dirname, join

from torch.nn import BCEWithLogitsLoss

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from embedding.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score
from core.graphgps.utiliity.utils import config_device
from typing import Dict, Tuple
from core.graphgps.utiliity.utils import Logger

class Trainer_SEAL(Trainer):
    def __init__(self,
                 FILE_PATH,
                 cfg,
                 model,
                 optimizer,
                 splits,
                 run,
                 repeat,
                 loggers,
                 batch_size=None,):
        self.device = config_device(cfg)

        self.model = model.to(self.device)

        self.model_name = cfg.model.type
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH
        self.epochs = cfg.train.epochs
        self.run = run
        self.repeat = repeat
        self.loggers = loggers
        self.batch_size = batch_size

        self.test_data = splits['test']
        self.train_data = splits['train']
        self.valid_data = splits['valid']
        self.optimizer = optimizer
        self.train_func = self._train_seal
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage', 'GNNStack', 'SEAL']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

    def _train_seal(self):
        self.model.train()
        total_loss = 0
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.train_data)
    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_seal()

            if epoch % 1 == 0:
                results_rank = self.merge_result_rank()
                print(results_rank)

                for key, result in results_rank.items():
                    print(key, result)
                    self.loggers[key].add_result(self.run, result)
                    print(self.run)
                    print(result)

        return best_auc, best_hits

    @torch.no_grad()
    def _test(self, data: Data):
        self.model.eval()
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        y_pred, y_true = [], []
        for data in test_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred), auc(fpr, tpr)

    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()
        data_loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)
        y_pred, y_true = [], []
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = data.edge_weight
            node_id = data.node_id
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        y_pred, y_true = y_pred.cpu(), y_true.cpu()

        hard_thres = (y_pred.max() + y_pred.min()) / 2

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        acc = torch.sum(y_true == y_pred) / len(y_true)


        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})

        return result_mrr
