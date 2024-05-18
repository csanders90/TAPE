import os
import sys
# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# External module imports
from yacs.config import CfgNode as CN

import torch
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from torch_geometric.nn import GCNConv
import wandb 
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from sklearn.metrics import *
import torch.optim as optim
import torch.nn.functional as F
import torch_scatter 
import torch_geometric 

from heuristic.eval import get_metric_score
from data_utils.load import data_loader
from textfeat.mlp_dot_product import FILE_PATH, set_cfg
from embedding.tune_utils import (parse_args, 
                                get_git_repo_root_path, 
                                param_tune_acc_mrr)
from utils import config_device



class GraphSage(MessagePassing):
    
    def __init__(self, cfg, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = cfg.data.num_features
        self.out_channels = cfg.model.out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)
        self.lin_r = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        """"""
        out = self.propagate(edge_index=edge_index, x=(x, x), size=size)
        out = self.lin_l(x) + self.lin_r(out)
        if self.normalize:
            out = F.normalize(out, p=2)
        return out

    def message(self, x_j):
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size = None):
        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index=index, dim=node_dim, dim_size=dim_size, reduce='mean')
        return out


class GAT(MessagePassing):
                        
    def __init__(self, cfg, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = cfg.data.num_features
        self.out_channels = cfg.model.out_channels
        self.heads = cfg.model.heads
        self.negative_slope = cfg.model.negative_slope
        self.dropout = cfg.model.dropout
        
        self.lin_l = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(data=torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(data=torch.zeros(self.heads, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels
        # reshape to [num_nodes, num_heads, hidden_size]
        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        # alpha vectors
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        out = self.propagate(edge_index=edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size).reshape(-1, H * C)
        return out


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        if ptr:
            att_weight = softmax(alpha, ptr)
        else:
            att_weight = softmax(alpha, index)
        att_weight = F.dropout(att_weight, p=self.dropout)
        
        out = att_weight * x_j
        return out


    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index=index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out
    

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


class LinkPredModel(torch.nn.Module):
    def __init__(self, encode):
        super(LinkPredModel, self).__init__()

        self.encode = encode
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.decoder = InnerProductDecoder()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        out = self.encode(x, edge_index)
        src_emb, dest_emb = out[edge_index[0]], out[edge_index[1]]
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746/11
        pred = (src_emb * dest_emb).sum(1)

        return pred
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """In this script we use the binary cross entropy loss function.

        params
        ----
        z: output of encoder
        pos_edge_index: positive edge index
        neg_edge_index: negative edge index
        """
        EPS = 1e-15 # prevent log(0)
        # positive loss log()
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean() # loss for positive samples

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0)) # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean() # loss for negative samples

        return pos_loss + neg_loss


class GCNEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        """GCN encoder with 2 layers
        in_channels: number of input features of node (dimension of node feature)
        hidden_channels: number of hidden features of node (dimension of hidden layer)
        out_channels: number of output features of node (dimension of node embedding)
        """
        in_channels = cfg.model.in_channels
        hidden_channels = cfg.model.hidden_channels
        out_channels = cfg.model.out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        """
        params:
        z: [num_nodes, num_features] node embedding
        edge_index: [2, num_edges] index of node pairs
        """
        # Compute the inner product between node embedding vectors
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        # Returns the probability of edge existence, using the sigmoid function to map the value to the range [0, 1]
        return torch.sigmoid(value) if sigmoid else value


class GAE(torch.nn.Module):
    """graph auto encoder。
    """
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder()

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        """In this script we use the binary cross entropy loss function.

        params
        ----
        z: output of encoder
        pos_edge_index: positive edge index
        neg_edge_index: negative edge index
        """
        EPS = 1e-15 # prevent log(0)
        # positive loss log()
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean() # loss for positive samples

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0)) # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean() # loss for negative samples

        return pos_loss + neg_loss


MAX_LOGSTD = 10  # Sets an upper limit for the logarithmic standard deviation
        
class VGAE(GAE):
    """inhert GAE class, since we need to use encode, decode and loss
    """

    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """encoder"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs) # mu stad stands for distribution for mean and std
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD) # upper bound of logstd
        z = self.reparametrize(self.__mu__, self.__logstd__) # reparameterization trick
        return z

    def kl_loss(self, mu=None, logstd=None):
        """We add a prior of (0, I) Gaussian variables to the distribution of the hidden variables,
        i.e., we want the distribution of the hidden variables to obey a (0, I) Gaussian distribution
        The difference between these two distributions is measured by the KL loss."""
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)) # KL loss between gaussian distribution and hidden variables


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
            
        in_channels = cfg.data.num_features
        out_channels = cfg.model.out_channels
        
        self.conv1 = GCNConv(in_channels, 2 * out_channels)  # 2*out_channels because we want to output both mu and logstd
        self.conv_mu = GCNConv(2 * out_channels, out_channels)  # We use 2*out_channels for the input because we want to concatenate mu and logstd
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)  # We use 2*out_channels for the input because we want to concatenate mu and logstd

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd
    

class Trainer():
    def __init__(self, 
                 FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits,
                 gsaint=None,
                 batch_size=None, 
                 walk_length=None, 
                 num_steps=None, 
                 sample_coverage=None):
        
        self.device = config_device(cfg)
            
        self.model = model.to(self.device)
        
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name

        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        
        # Added GSAINT normalization
        if gsaint is not None:
            self.test_data = gsaint(splits['test'], batch_size, walk_length, num_steps, sample_coverage)
            self.train_data = gsaint(splits['train'], batch_size, walk_length, num_steps, sample_coverage)
            self.valid_data = gsaint(splits['valid'], batch_size, walk_length, num_steps, sample_coverage)
        else:
            self.test_data = splits['test']
            self.train_data = splits['train']
            self.valid_data = splits['valid']
        
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.optimizer = optimizer
        
        self.train_func = {
        'gae': self._train_gae,
        'vgae': self._train_vgae, 
        'GAT': self._train_gae,
        'GraphSage': self._train_gae,
        'GNNStack': self._train_gnnstack,
        'GCNEncode': self._train_gnnstack,
        }
        
        self.test_func = {
            'GAT': self._test,
            'gae': self._test,
            'vgae': self._test,
            'GraphSage': self._test,
            'GNNStack': self._test,
            'GCNEncode': self._test
        }
        
        self.evaluate_func = {
            'GAT': self._evaluate_gat_true,
            'gae': self._evaluate_gat_true,
            'vgae': self._evaluate_gat_true,
            'GraphSage': self._evaluate_gat_true,
            'GNNStack': self._evaluate_gat_true,
            'GCNEncode': self._evaluate_gat_true
        }
        
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')

        
    def _train_gnnstack(self):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(pred, self.train_data.edge_label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _train_gae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_vgae(self):
        """Training the VGAE model, the loss function consists of reconstruction loss and kl loss"""
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss() # add kl loss # Forgot self
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test_gat(self):
        """test"""
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
        self.model.eval()
        edge_index = self.test_data.edge_label_index
        pred = self.model(self.test_data.x, self.test_data.edge_label_index)
        y = self.test_data.edge_label 
        
        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        auc = auc(fpr, tpr)
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc

    @torch.no_grad()
    def _evaluate_gat_true(self):
        self.model.eval()
        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encode(self.test_data.x, self.test_data.edge_index)
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
                # Forgot print MRR and I delete a big space between AUC and AP.
                print('Epoch: {:03d}, Loss_train: {:.4f}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, MRR: {:.4f}'.format(epoch, loss, auc, ap, acc, result_mrr['Hits@100']))
                if auc > best_auc:
                    best_auc = auc 
                elif result_mrr['Hits@100'] > best_hits:
                    best_hits = result_mrr['Hits@100']
        return best_auc, best_hits, result_mrr


    def evaluate(self):
        return self.evaluate_func[self.model_name]()
        
        
    @torch.no_grad()
    def _evaluate_gat_false(self):

        self.model.eval()
        edge_index = self.test_data.edge_label_index
        y = self.test_data.edge_label
        
        y_pred = self.model(self.test_data.x, edge_index)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_pred = y_pred[y== 1]
        neg_pred = y_pred[y == 0]
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.detach().cpu(), neg_pred.detach().cpu()
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

        z = self.model.encode(self.test_data.x, self.test_data.edge_index)
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
    
    
    @torch.no_grad()
    def _evaluate(self):

        self.model.eval()
        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encode(self.test_data.x, self.test_data.edge_index)
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
        

        pos_y = z.new_ones(pos_edge_index.size(1)) # positive samples
        neg_y = z.new_zeros(neg_edge_index.size(1)) # negative samples
        y = torch.cat([pos_y, neg_y], dim=0)
        
        return result_mrr
        

    def save_result(self, results_dict):

        root = self.FILE_PATH + cfg.out_dir
        acc_file = root + f'/{self.model_name}_acc_mrr.csv'
        print(acc_file)
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        id = wandb.util.generate_id()
        param_tune_acc_mrr(id, results_dict, acc_file, self.data_name, self.model_name)



if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    dataset, data_cited, splits = data_loader[cfg.data.name](cfg)   
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']


    
    if cfg.model.type == 'gae':
        model = GAE(GCNEncoder(cfg))
    elif cfg.model.type == 'vgae':
        model = VGAE(VariationalGCNEncoder(cfg))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits)
    
    trainer.train()
    results_dict = trainer.evaluate()
    
    trainer.save_result(results_dict)
    

