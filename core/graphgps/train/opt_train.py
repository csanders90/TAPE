import os
import sys
from os.path import abspath, dirname, join

from torch.nn import BCEWithLogitsLoss

sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))

import torch
import wandb
import time   
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN
from tqdm import tqdm 
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling

from embedding.tune_utils import param_tune_acc_mrr, mvari_str2csv, save_parmet_tune
from torch_geometric.loader import DataLoader
from embedding.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score
from utils import config_device
from typing import Dict, Tuple
from utils import Logger


report_step = {
    'cora': 100,
    'pubmed': 1000,
    'arxiv_2023': 100,
    'ogbn-arxiv': 1,
    'ogbn-products': 1,
}


class Trainer():
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 emb: torch.nn.Module,
                 data: Data,
                 optimizer: torch.optim.Optimizer, 
                 splits: Dict[str, Data], 
                 run: int, 
                 repeat: int,
                 loggers: Logger, 
                 print_logger: None, 
                 device: int):
        
        self.device = device
        self.model = model.to(self.device)
        self.emb = emb
        
        # params
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH 
        self.name_tag = cfg.wandb.name_tag
        self.epochs = cfg.train.epochs
        self.batch_size = cfg.train.batch_size
        
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.data = data
        self.optimizer = optimizer
        self.loggers = loggers
        self.print_logger = print_logger
        self.report_step = report_step[cfg.data.name]
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage']
        self.train_func = {model_type: self._train_gae if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._train_vgae for model_type in model_types}
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._evaluate_vgae for model_type in model_types}
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.run = run
        self.repeat = repeat
        self.results_rank = {}
        self.run_result = {}
        
    def _train_gae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def _train_vgae(self):
        self.model.train()
        self.optimizer.zero_grad()
        # encoder is VAE, forward is embedding
        z = self.model(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    @torch.no_grad()
    def _test(self, data: Data):
        """test"""
        self.model.eval()

        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encoder(self.test_data.x, self.test_data.edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1)) 
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(y, pred, pos_label=1)
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc(fpr, tpr)


    @torch.no_grad()
    def _evaluate(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model.encoder(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        # add a visualization of the threshold
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    @torch.no_grad()
    def _evaluate_vgae(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    def merge_result_rank(self):
        result_test = self.evaluate_func[self.model_name](self.test_data)
        result_valid = self.evaluate_func[self.model_name](self.valid_data)
        result_train = self.evaluate_func[self.model_name](self.train_data)

        return {
            key: (result_train[key], result_valid[key], result_test[key])
            for key in result_test.keys()
        }
    
    
    def train(self):  
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            
            if epoch % int(self.epochs/100) == 0:
                self.results_rank = self.merge_result_rank()
                
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {self.results_rank["AUC"][0]:.4f}, AP: {self.results_rank["AP"][0]:.4f}, MRR: {self.results_rank["MRR"][0]:.4f}, Hit@10 {self.results_rank["Hits@10"][0]:.4f}')
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_valid: {loss:.4f}, AUC: {self.results_rank["AUC"][1]:.4f}, AP: {self.results_rank["AP"][1]:.4f}, MRR: {self.results_rank["MRR"][1]:.4f}, Hit@10 {self.results_rank["Hits@10"][1]:.4f}')               
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_test: {loss:.4f}, AUC: {self.results_rank["AUC"][2]:.4f}, AP: {self.results_rank["AP"][2]:.4f}, MRR: {self.results_rank["MRR"][2]:.4f}, Hit@10 {self.results_rank["Hits@10"][2]:.4f}')               
                    
                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    if epoch % 500 == 0:
                        for key, result in self.results_rank.items():
                            train_hits, valid_hits, test_hits = result
                            self.print_logger.info(
                                f'Run: {self.run + 1:02d}, Key: {key}, '
                                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_hits:.2f}, Valid: {100 * valid_hits:.2f}, Test: {100 * test_hits:.2f}%')
                        self.print_logger.info('---')

        for _ in range(1):
            start_train = time.time() 
            self.train_func[self.model_name]()
            self.run_result['train_time'] = time.time() - start_train
            self.evaluate_func[self.model_name](self.test_data)
            self.run_result['eval_time'] = time.time() - start_train


    def result_statistic(self):
        result_all_run = {}
        for key in self.loggers:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list = self.loggers[key].calc_all_stats()
            
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

            elif key == 'Hits@100':
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]

        print(f'{best_metric_valid_str} {best_auc_valid_str}')
        print(best_metric_valid_str)
        best_auc_metric = best_valid_mean_metric
        return best_valid_mean_metric, best_auc_metric, result_all_run
    

    def save_result(self, results_dict: Dict[str, float]):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, f'{self.data_name}_wb_acc_mrr.csv')
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        mvari_str2csv(self.name_tag, results_dict, acc_file)


    def save_tune(self, results_dict: Dict[str, float], to_file):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, to_file)
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        save_parmet_tune(self.name_tag, results_dict, acc_file)    


class Trainer_Saint(Trainer):
    def __init__(self, 
                 FILE_PATH,
                 cfg,
                 model, 
                 emb,
                 data,
                 optimizer,
                 splits,
                 run, 
                 repeat, 
                 loggers,
                 print_logger,
                 device,
                 sampler=None):
        super().__init__(FILE_PATH,
                 cfg,
                 model, 
                 emb,
                 data,
                 optimizer,
                 splits,
                 run, 
                 repeat, 
                 loggers,
                 print_logger,
                 device)
        
        self.sampler = sampler
        batch_size = cfg.sampler.batch_size 
        walk_length = cfg.sampler.walk_length
        num_steps = cfg.sampler.num_steps
        sample_coverage = cfg.sampler.sample_coverage
        
        # Added GSAINT normalization
        if self.sampler is not None:
            self.test_data = self.sampler(splits['test'], batch_size, walk_length, num_steps, sample_coverage)
            self.train_data = self.sampler(splits['train'], batch_size, walk_length, num_steps, sample_coverage)
            self.valid_data = self.sampler(splits['valid'], batch_size, walk_length, num_steps, sample_coverage)
        else:
            self.test_data = splits['test']
            self.train_data = splits['train']
            self.valid_data = splits['valid']
        
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        

class Trainer_Heart(Trainer):
    def __init__(self, 
                FILE_PATH,
                cfg,
                model, 
                emb,
                data,
                optimizer,
                splits,
                run, 
                repeat, 
                loggers,
                print_logger,
                device):
        super().__init__(FILE_PATH,
                    cfg,
                    model, 
                    emb,
                    data,
                    optimizer,
                    splits,
                    run, 
                    repeat, 
                    loggers,
                    print_logger,
                    device)
        
        self.batch_size = cfg.train.batch_size
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage']

        self.train_func = {model_type: self._train_heart for model_type in model_types}
        self.test_func = {model_type: self._eval_heart  for model_type in model_types}
        self.evaluate_func = {model_type: self._eval_heart  for model_type in model_types}


        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
    def _train_heart(self):

        edge_index = self.train_data.edge_index
        pos_train_weight = None
        
        if self.emb is None: 
            x = self.train_data.x
            emb_update = 0
        else: 
            x = self.emb.weight
            emb_update = 1
      
        for perm in DataLoader(range(edge_index.size(1)),
                          batch_size=self.batch_size,
                          shuffle=True,  # Adjust the number of workers based on your system configuration
                          pin_memory=True,  # Enable pinning memory for faster data transfer
                          drop_last=True):  # Drop the last incomplete batch if dataset size is not divisible by batch size
            
            self.optimizer.zero_grad()
            num_nodes = x.size(0)

            ######################### remove loss edges from the aggregation
            mask = torch.ones(edge_index.size(1), dtype=torch.bool).to(edge_index.device)
            mask[perm] = 0
            train_edge_mask = edge_index[:, mask]
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)

            # visualize
            if pos_train_weight != None:
                pos_train_weight = pos_train_weight.to(mask.device)
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(edge_index.device)

            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(edge_index.device)

            row, col, _ = adj.coo()
            batch_edge_index = torch.stack([col, row], dim=0)
            
            
            x = x.to(self.device)
            pos_edge =  edge_index[:, perm].to(self.device)
            if self.model_name == 'VGAE':
                h = self.model(x, batch_edge_index)
                loss = self.model.recon_loss(h, pos_edge)
                loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
            elif self.model_name in ['GAE', 'GAT', 'GraphSage']:
                h = self.model.encoder(x, batch_edge_index)
                loss = self.model.recon_loss(h, pos_edge)

            loss.backward()

            if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()           

        return loss.item() 


    @torch.no_grad()
    def test_edge(self, h, edge_index):
        preds = []
        edge_index = edge_index.t()

        for perm  in DataLoader(range(edge_index.size(0)), self.batch_size):
            edge = edge_index[perm].t()

            preds += [self.model.decoder(h, edge).cpu()]

        return torch.cat(preds, dim=0)


    @torch.no_grad()
    def _eval_heart(self, data: Data):
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        if self.model_name == 'VGAE':
            z = self.model(data.x, data.edge_index)
        elif self.model_name in ['GAE', 'GAT', 'GraphSage']:
            z = self.model.encoder(data.x, data.edge_index)
        
        pos_pred = self.test_edge(z, pos_edge_index)
        neg_pred = self.test_edge(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        y = y.to(self.device)
        y_pred = y_pred.to(self.device)
        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    
    @torch.no_grad()
    def _test_edge(self, 
                  input_data, 
                  h):
        preds = []
        # sourcery skip: no-loop-in-tests
        for perm  in DataLoader(range(input_data.size(0)), self.batch_size):
            edge = input_data[perm].t()
            preds += [self.model.decoder(h[edge[0]], h[edge[1]]).cpu()]

        return torch.cat(preds, dim=0)
    

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
                 print_logger = None,
                 batch_size=None):
        self.name_tag = cfg.wandb.name_tag
        self.print_logger = print_logger
        self.report_step = report_step[cfg.data.name]

        self.device = config_device(cfg).device

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
        self.print_logger = print_logger
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
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            x = data.x
            edge_weight = None
            node_id = None
            logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs

        return total_loss / len(self.train_data)
    def train(self):
        best_auc, best_hits, best_hit100 = 0, 0, 0
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch: {epoch}','start training...')
            loss = self._train_seal()
            print(f'Loss: {loss}')
            if epoch % 10 == 0:
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

        hard_thres = (y_pred.max() + y_pred.min())/2
        '''plot_acc(y_true, y_pred, hard_thres)'''

        pos_pred, neg_pred = y_pred[y_true == 1].cpu(), y_pred[y_true == 0].cpu()
        y_pred = torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))

        y_true = y_true.clone().detach()
        y_pred = y_pred.clone().detach()

        acc = torch.sum(y_true == y_pred) / len(y_true)


        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})

        return result_mrr


import matplotlib.pyplot as plt
import seaborn as sns

counter = 0
import numpy as np

def plot_acc(y_true, y_pred, hard_thres):
    file_name = 'data.npz'
    np.savez(file_name, pos_pred=y_pred[y_true == 1], neg_pred=y_pred[y_true == 0])
    global counter
    counter += 1
    file_name = f'plot_{counter}_acc_{torch.sum(y_true == torch.where(y_pred >= hard_thres, torch.tensor(1), torch.tensor(0))) / len(y_true)}.png'
    save_dir = './plots'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    plt.figure(figsize=(8, 6))

    plt.axvline(x=hard_thres, color='red', linestyle='--', label=f'Hard Threshold: {hard_thres.item()}')

    sns.kdeplot(x=y_pred[y_true == 1], cmap="Blues", fill=True, bw_adjust=.5, label='y_true = 1')
    sns.kdeplot(x=y_pred[y_true == 0], cmap="Reds", fill=True, bw_adjust=.5, label='y_true = 0')

    plt.xlabel('distribution')
    plt.xlabel('y_pred')
    plt.title('Distribution Plot of Predictions vs Hard Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
