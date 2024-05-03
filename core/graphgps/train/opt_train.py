import os
import torch
import wandb
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from graphgps.loss.custom_loss import RecLoss

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc

from utils import config_device
from embedding.tune_utils import param_tune_acc_mrr
from heuristic.eval import get_metric_score

class Trainer():
    def __init__(self, FILE_PATH, cfg, model, optimizer, splits):
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

        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage', 'GNNStack']
        self.train_func = {model_type: self._train_gae if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._train_vgae for model_type in model_types}
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate for model_type in model_types}

        self.evaluator_hit = Evaluator(name='ogbl-collab'),
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

    def _train_gae(self):
        z = self._extracted_from__train_vgae_2()
        loss_func = RecLoss()
        loss = loss_func(z, self.train_data.pos_edge_label_index)
        return self._extracted_from__train_vgae_7(loss)

    def _train_vgae(self):
        z = self._extracted_from__train_vgae_2()
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        return self._extracted_from__train_vgae_7(loss)

    # TODO Rename this here and in `_train_gae` and `_train_vgae`
    def _extracted_from__train_vgae_7(self, loss):
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # TODO Rename this here and in `_train_gae` and `_train_vgae`
    def _extracted_from__train_vgae_2(self):
        self.model.train()
        self.optimizer.zero_grad()
        return self.model.encoder(self.train_data.x, self.train_data.edge_index)


    @torch.no_grad()
    def _test(self):
        """test"""
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
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc(fpr, tpr)

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
    
    def train(self):
        best_hits, best_auc = 0, 0
        for epoch in range(1, self.epochs + 1):

            loss = self.train_func[self.model_name]()
            if epoch % 100 == 0:
                auc, ap, acc = self.test_func[self.model_name]()
                result_mrr = self.evaluate_func[self.model_name]()
                print('Epoch: {:03d}, Loss_train: {:.4f}, AUC: {:.4f}, \
                      AP: {:.4f}, ACC: {:.4f}, MRR'.format(epoch, loss, auc, ap, acc, result_mrr['Hits@100']))
                if auc > best_auc:
                    best_auc = auc 
                elif result_mrr['Hits@100'] > best_hits:
                    best_hits = result_mrr['Hits@100']
        return best_auc, best_hits, result_mrr


    def save_result(self, results_dict):
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, f'{self.model_name}_acc_mrr.csv')
        os.makedirs(root, exist_ok=True)
        id = wandb.util.generate_id()
        param_tune_acc_mrr(id, results_dict, acc_file, self.data_name, self.model_name)
