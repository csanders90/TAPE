import os
import sys
import time
from os.path import abspath, dirname, join
from torch.nn import BCEWithLogitsLoss

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from embedding.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score
from graphgps.utility.utils import config_device, Logger
from typing import Dict, Tuple
from graphgps.train.opt_train import (Trainer)
from graphgps.utility.ncn import PermIterator
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class Trainer_embedding_LLM(Trainer):
    def __init__(self,
                 FILE_PATH: str,
                 cfg: CN,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 embedding: torch.Tensor,
                 splits: Dict[str, Data],
                 run: int,
                 repeat: int,
                 loggers: Dict[str, Logger],
                 print_logger: None,
                 batch_size=None,):
        self.device = config_device(cfg).device
        self.model = model.to(self.device)
        self.model_name = cfg.model.type
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH
        self.epochs = cfg.train.epochs
        self.run = run
        self.repeat = repeat
        self.loggers = loggers
        self.print_logger = print_logger
        self.batch_size = batch_size
        self.embedding = embedding.to(self.device)

        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        model_types = ['MLP-minilm','MLP-bert','MLP-llama','MLP-e5-large']
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        self.run = run
        self.repeat = repeat
        self.results_rank = {}

        self.name_tag = cfg.wandb.name_tag
        self.run_result = {}

        self.tensorboard_writer = writer
        self.out_dir = cfg.out_dir
        self.run_dir = cfg.run_dir

        self.report_step = 1

    def _train_mlp(self):
        self.model.train()
        total_loss = 0
        pos_train_edge = self.train_data['pos_edge_label_index'].T.to(self.device)
        neg_train_edge = self.train_data['neg_edge_label_index'].T.to(self.device)

        for perm in PermIterator(self.device, pos_train_edge.shape[0], self.batch_size):
            self.optimizer.zero_grad()
            pos_edge = pos_train_edge[perm].t()
            pos_out = self.model(self.embedding[pos_edge][0], self.embedding[pos_edge][1])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            neg_edge = neg_train_edge[perm].t()
            neg_out = self.model(self.embedding[neg_edge][0], self.embedding[neg_edge][1])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * self.batch_size

        return total_loss / len(self.train_data)

    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_mlp()
            if epoch % int(self.report_step) == 0:
                self.results_rank = self.merge_result_rank()
                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    self.tensorboard_writer.add_scalar(f"Metrics/Train/{key}", result[0], epoch)
                    self.tensorboard_writer.add_scalar(f"Metrics/Valid/{key}", result[1], epoch)
                    self.tensorboard_writer.add_scalar(f"Metrics/Test/{key}", result[2], epoch)

                    train_hits, valid_hits, test_hits = result
                    self.print_logger.info(
                        f'Run: {self.run + 1:02d}, Key: {key}, '
                        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')

                self.print_logger.info('---')

        return best_auc, best_hits

    @torch.no_grad()
    def _evaluate(self, eval_data: Data):

        self.model.eval()
        pos_train_edge = eval_data['pos_edge_label_index'].T.to(self.device)
        neg_train_edge = eval_data['neg_edge_label_index'].T.to(self.device)

        pos_pred = self.model(self.embedding[pos_train_edge.t()][0], self.embedding[pos_train_edge.t()][1])
        neg_pred = self.model(self.embedding[neg_train_edge.t()][0], self.embedding[neg_train_edge.t()][1])
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        y_pred = y_pred.view(-1).cpu()
        hard_thres = (y_pred.max() + y_pred.min()) / 2
        pos_y = torch.ones(pos_train_edge.size(0))
        neg_y = torch.zeros(neg_train_edge.size(0))
        y_true = torch.cat([pos_y, neg_y], dim=0)

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach().cpu()
        y_pred = y_pred.clone().detach().cpu()

        acc = torch.sum(y_true == y_pred) / len(y_true)

        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'ACC': round(acc.tolist(), 5)})

        return result_mrr

    def finalize(self):
        import time
        for _ in range(1):
            start_train = time.time()
            self._evaluate(self.test_data)
            self.run_result['eval_time'] = time.time() - start_train

    def save_pred(self, pred, true, data):
        root = os.path.join(self.FILE_PATH, cfg.out_dir, 'pred_record')
        os.makedirs(root, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        file_path = os.path.join(root, f'{cfg.dataset.name}_{timestamp}.txt')

        with open(file_path, 'w') as f:
            for idx, subgraph in enumerate(data):
                indices = torch.where(subgraph['z'] == 1)[0]
                if len(indices) < 2:
                    continue
                corresponding_node_ids = subgraph['node_id'][indices]
                pred_value = pred[idx]
                true_value = true[idx]
                f.write(f"{corresponding_node_ids[0].item()} {corresponding_node_ids[1].item()} {pred_value} {true_value}\n")
