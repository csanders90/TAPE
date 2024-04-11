import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv

from torch.nn import Embedding
from utils import init_seed,  Logger, save_emb
from torch.nn.init import xavier_normal_
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)
from torch_geometric.nn import global_sort_pool
import math
from utils import *
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling
import os

from torch_geometric.graphgym.config import cfg
import torch
from embedding.tune_utils import (
    parse_args, 
    get_git_repo_root_path
)
from sklearn.metrics import *
from gnn_models import set_cfg, data_loader, Trainer 
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import argparse
from trainer import train, test, test_edge

def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")

dir_path = get_root_dir()
log_print		= get_logger('testrun', 'log', get_config_dir())

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/configs/cora/gcns/heart_gnn_models.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/configs/cora/gat_sp1.yaml',
                        help='The configuration file path.')
    
    parser.add_argument('--repeat', type=int, default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, data_name=None):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels,normalize=False ))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        
        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if num_layers == 1:
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(in_channels, out_channels, heads=head))

        elif num_layers > 1:
            hidden_channels= int(self.hidden_channels/head)
            self.convs.append(GATConv(in_channels, hidden_channels, heads=head))
            
            for _ in range(num_layers - 2):
                hidden_channels =  int(self.hidden_channels/head)
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels, heads=head))
            
            out_channels = int(self.out_channels/head)
            self.convs.append(GATConv(hidden_channels, out_channels, heads=head))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gat: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))

        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in sage: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class mlp_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(mlp_model, self).__init__()

        self.lins = torch.nn.ModuleList()

        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t=None):
        if self.invest == 1:
            print('layers in mlp: ', len(self.lins))
            self.invest = 0
       
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None,  cat_node_feat_mf=False,  data_name=None):
        super(GIN, self).__init__()

         # self.mlp1= mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
        # self.mlp2 = mlp_model( hidden_channels, hidden_channels, out_channels, gin_mlp_layer, dropout)

        self.convs = torch.nn.ModuleList()
        gin_mlp_layer = mlp_layer
        
        if num_layers == 1:
            self.mlp= mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
            self.convs.append(GINConv(self.mlp))

        else:
            # self.mlp_layers = torch.nn.ModuleList()
            self.mlp1 = mlp_model( in_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
            
            self.convs.append(GINConv(self.mlp1))
            for _ in range(num_layers - 2):
                self.mlp = mlp_model( hidden_channels, hidden_channels, hidden_channels, gin_mlp_layer, dropout)
                self.convs.append(GINConv(self.mlp))

            self.mlp2 = mlp_model( hidden_channels, hidden_channels, out_channels, gin_mlp_layer, dropout)
            self.convs.append(GINConv(self.mlp2))

        self.dropout = dropout
        self.invest = 1
          
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # self.mlp1.reset_parameters()
        # self.mlp2.reset_parameters()



    def forward(self, x, adj_t):
        if self.invest == 1:
            print('layers in gin: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class MF(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  mlp_layer=None,  head=None, node_num=None, cat_node_feat_mf=False,  data_name=None):
        super(MF, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.data = data_name
        if num_layers == 0:
            out_mf = out_channels
            if self.data=='ogbl-citation2':
                out_mf = 96

            self.emb =  torch.nn.Embedding(node_num, out_mf)
        else:
            self.emb =  torch.nn.Embedding(node_num, in_channels)

        if cat_node_feat_mf:
            in_channels = in_channels*2
    

        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers
        self.cat_node_feat_mf = cat_node_feat_mf

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            
        if self.data == 'ogbl-citation2':
            print('!!!! citaion2 !!!!!')
            torch.nn.init.normal_(self.emb.weight, std = 0.2)

        else: 
            self.emb.reset_parameters()



    def forward(self, x=None, adj_t=None):
        if self.invest == 1:
            print('layers in mlp: ', len(self.lins))
            self.invest = 0
        if self.cat_node_feat_mf and x != None:
            # print('xxxxxxx')
            x = torch.cat((x, self.emb.weight), dim=-1)

        else:
            x =  self.emb.weight


        if self.num_layers == 0:
            return self.emb.weight
        
        else:
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.lins[-1](x)

            return x


class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 dynamic_train=False, GNN=GCNConv, use_feature=False, 
                 node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        emb = x

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def data_preprocess(cfg):

    device = cfg.train.device
    dataset, data_cited, splits = data_loader[cfg.data.name](cfg) 
    data = dataset._data

    edge_index = data.edge_index
    emb = None # here is your embedding
    node_num = data.num_nodes

    if hasattr(data, 'x'):
        if data.x != None:
            x = data.x
            cfg.model.input_channels = x.size(1)
        else:
            emb = torch.nn.Embedding(node_num, args.hidden_channels)
            cfg.model.input_channels = args.hidden_channels

    else:
        emb = torch.nn.Embedding(node_num, args.hidden_channels)
        cfg.model.input_channels = args.hidden_channels
    
    if not hasattr(data, 'edge_weight'): 
        train_edge_weight = torch.ones(splits['train'].edge_index.shape[1])
        train_edge_weight = train_edge_weight.to(torch.float)


    data = T.ToSparseTensor()(data)

    if cfg.train.use_valedges_as_input:
        # in the previous setting we share the same train and valid 
        val_edge_index = splits['valid'].edge_index.t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        edge_weight = torch.ones(full_edge_index.shape[1])
        train_edge_weight = torch.ones(splits['train'].edge_index.shape[1])
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    else:
        data.full_adj_t = data.adj_t


    if emb != None:
        torch.nn.init.xavier_uniform_(emb.weight)
    return dataset, splits, emb, cfg, train_edge_weight


if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    device = config_device(cfg)

    cfg.train.device = device
    print(f"device {device}")
    
    dataset, splits, emb, cfg, train_edge_weight = data_preprocess(cfg)

    model = eval(cfg.model.type)(cfg.model.input_channels, cfg.model.hidden_channels,
                                 cfg.model.hidden_channels, cfg.model.num_layers, 
                                 cfg.model.dropout).to(device)
    
    score_func = eval(cfg.score_model.name)(cfg.score_model.hidden_channels, 
                                            cfg.score_model.hidden_channels,
                                            1, 
                                            cfg.score_model.num_layers_predictor, 
                                            cfg.score_model.dropout).to(device)

    # train_pos = data['train_pos'].to(x.device)

    # eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    # config reset parameters 
    model.reset_parameters()
    score_func.reset_parameters()
    
    if cfg.model.emb is True:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=cfg.train.lr, weight_decay=cfg.train.l2)

    if cfg.data.name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif cfg.data.name =='ogbl-ddi':
        eval_metric = 'Hits@20'

    elif cfg.data.name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    
    elif cfg.data.name =='ogbl-citation2':
        eval_metric = 'MRR'
        
    elif cfg.data.name in ['cora', 'pubmed', 'arxiv_2023']:
        eval_metric = 'Hits@100'
        
    if cfg.data.name != 'ogbl-citation2':
        pos_train_edge = splits['train'].edge_index

        pos_valid_edge = splits['valid'].pos_edge_label_index
        neg_valid_edge = splits['valid'].neg_edge_label_index
        pos_test_edge = splits['test'].pos_edge_label_index
        neg_test_edge = splits['test'].neg_edge_label_index
    
    else:
        source_edge, target_edge = splits['train']['source_node'], splits['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

        # idx = torch.randperm(split_edge['train']['source_node'].numel())[:split_edge['valid']['source_node'].size(0)]
        # source, target = split_edge['train']['source_node'][idx], split_edge['train']['target_node'][idx]
        # train_val_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)

        source, target = splits['valid']['source_node'],  splits['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        neg_valid_edge = splits['valid']['target_node_neg'] 

        source, target = splits['test']['source_node'],  splits['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        neg_test_edge = splits['test']['target_node_neg']

    loggers = {
        'Hits@20': Logger(cfg.train.runs),
        'Hits@50': Logger(cfg.train.runs),
        'Hits@100': Logger(cfg.train.runs),
        'MRR': Logger(cfg.train.runs),
        'AUC':Logger(cfg.train.runs),
        'AP':Logger(cfg.train.runs),
        'mrr_hit20':  Logger(cfg.train.runs),
        'mrr_hit50':  Logger(cfg.train.runs),
        'mrr_hit100':  Logger(cfg.train.runs),
    }
            
    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]

    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    
    for run in range(cfg.train.runs):

        print('#################################          ', run, '          #################################')
        if cfg.train.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        save_path = cfg.save.output_dir+'/lr'+str(cfg.train.lr) \
            + '_drop' + str(cfg.model.dropout) + '_l2'+ \
                str(cfg.train.l2) + '_numlayer' + str(cfg.model.num_layers)+ \
                    '_numPredlay' + str(cfg.score_model.num_layers_predictor) +\
                        '_numGinMlplayer' + str(cfg.score_model.gin_mlp_layer)+ \
                            '_dim'+str(cfg.model.hidden_channels) + '_'+ 'best_run_'+str(seed)
        
        # save_valid = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'valid_output'+str(run)
        # save_valid = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'valid_output'+str(run)
        

        if emb != None:
            torch.nn.init.xavier_uniform_(emb.weight)

        model.reset_parameters()
        score_func.reset_parameters()

        if emb != None:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=cfg.train.lr, weight_decay=cfg.train.l2)
        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=cfg.train.lr, weight_decay=cfg.train.l2)
        best_valid = 0
        kill_cnt = 0
        
        for  epoch in range(1, 1 + cfg.train.epochs):
            loss = train(model, 
                         score_func, 
                         pos_train_edge, 
                         dataset._data, 
                         emb, 
                         optimizer, 
                         cfg.train.batch_size, 
                         train_edge_weight, 
                         device)

            # for attention score   
            # print(model.convs[0].att_src[0][0][:10])
            
            
            if epoch % 10 == 0:
                results_rank, score_emb = test(model, 
                                               score_func,
                                               splits['test'],
                                               evaluation_edges, 
                                               emb, evaluator_hit, 
                                               evaluator_mrr, 
                                               cfg.train.batch_size, 
                                               cfg.data.name, 
                                               cfg.train.use_valedges_as_input, 
                                               device)


                for key, _ in loggers.items():
                    loggers[key].add_result(run, results_rank[key])

                if epoch % 2 == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                
                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)
                log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')
                
                if len(loggers['AUC'].results[run]) > 0:
                    r = torch.tensor(loggers['AUC'].results[run])
                    best_valid_auc = round(r[:, 1].max().item(), 4)
                    best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                    
                    print('AUC')
                    log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                f'best test: {100*best_test_auc:.2f}%')
                
                print('---')

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if cfg.save: save_emb(score_emb, save_path)

                    
                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > cfg.train.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                loggers[key].print_statistics( run)
                print('\n')
        
      
    
    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            
            best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()
           

            if key == eval_metric:
                best_metric_valid_str = best_metric
                # best_valid_mean_metric = best_valid_mean


                
            if key == 'AUC':
                best_auc_valid_str = best_metric
                # best_auc_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]
            

        
    if cfg.train.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    
    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))
