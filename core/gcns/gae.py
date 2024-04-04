
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# External module imports
import torch
import matplotlib.pyplot as plt
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
import itertools 
import scipy.sparse as ssp

from heuristic.eval import get_metric_score
from data_utils.load_cora_lp import get_cora_casestudy 
from data_utils.load_pubmed_lp import get_pubmed_casestudy
from data_utils.load_arxiv2023_lp import get_raw_text_arxiv_2023
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj


from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
import torch_geometric
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from textfeat.mlp_dot_product import distance_metric, data_loader, FILE_PATH, set_cfg
from embedding.tune_utils import (
    parse_args, 
    get_git_repo_root_path,
    param_tune_acc_mrr
)
import wandb 
from utils import config_device
from IPython import embed 


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        """GCN encoder with 2 layers
        in_channels: number of input features of node (dimension of node feature)
        hidden_channels: number of hidden features of node (dimension of hidden layer)
        out_channels: number of output features of node (dimension of node embedding)
        """
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
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
        
        self.train_func = {
        'gae': self._train_gae,
        'vgae': self._train_vgae
        }
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        
    def _train_gae(self):
        self.model.train()
        optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _train_vgae(self):
        """Training the VGAE model, the loss function consists of reconstruction loss and kl loss"""
        self.model.train()
        optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * model.kl_loss() # add kl loss
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self):
        
        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()
            if epoch % 100 == 0:
                auc, ap, acc = self._test()
                print('Epoch: {:03d}, Loss_train: {:.4f}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}'.format(epoch, loss, auc, ap, acc))

    @torch.no_grad()
    def _test(self):
        """test"""
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, auc
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
    def evaluate(self):

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

        root = self.FILE_PATH + f'results/gcns/{self.data_name}/'
        acc_file = root + f'{self.model_name}_acc_mrr.csv'

        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        id = wandb.util.generate_id()
        param_tune_acc_mrr(id, results_dict, acc_file, self.data_name, self.model_name)
    

data_loader = {
    'cora': get_cora_casestudy,
    'pubmed': get_pubmed_casestudy,
    'arxiv_2023': get_raw_text_arxiv_2023
}



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

    in_channels = cfg.data.num_features
    out_channels = cfg.model.out_channels
    hidden_channels = cfg.model.hidden_channels # Assume that the dimension of the hidden layer feature is 8
    
    if cfg.model.type == 'gae':
        model = GAE(GCNEncoder(in_channels, hidden_channels, out_channels))
    elif cfg.model.type == 'vgae':
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    
    trainer = Trainer(FILE_PATH,
                 cfg,
                 model, 
                 optimizer,
                 splits)
    
    trainer.train()
    results_dict = trainer.evaluate()
    
    trainer.save_result(results_dict)
    

