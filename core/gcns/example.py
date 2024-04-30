import os
import sys
# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Standard library imports
import torch.optim as optim
from collections import Counter

# Third-party imports
import torch
import torch_scatter
import torch_geometric
from ogb.linkproppred import Evaluator
from sklearn.metrics import *
import wandb
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.graphgym.config import cfg

# Local application/library specific imports

from heuristic.eval import get_metric_score
from textfeat.mlp_dot_product import data_loader, FILE_PATH, set_cfg
from utils import parse_args, get_git_repo_root_path
from embedding.tune_utils import param_tune_acc_mrr
from data_utils.load import data_loader
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
    

