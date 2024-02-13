import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

from typing import Dict
import numpy as np
import scipy.sparse as ssp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR
from heuristic.semantic_similarity import pairwise_prediction
import matplotlib.pyplot as plt
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj
from utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from heuristic.eval import evaluate_auc, evaluate_hits, evaluate_mrr, get_metric_score, get_prediction
from ge import Node2Vec
from yacs.config import CfgNode as CN
import networkx as nx 
import yaml
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg)
import graphgps 
from cora_emb   import eval_embed, set_cfg
from heuristic.pubmed_heuristic import get_pubmed_casestudy
from heuristic.cora_heuristic import get_cora_casestudy
from heuristic.arxiv2023_heuristic import get_raw_text_arxiv_2023

FILE_PATH = get_git_repo_root_path() + '/'

data_loader = {
    'cora': get_cora_casestudy,
    'pubmed': get_pubmed_casestudy,
    'arxiv_2023': get_raw_text_arxiv_2023
}

def eval_pubmed_mrr_acc(config) -> None:
    """load text attribute graph in link predicton setting
    """
    dataset, data_cited, splits = data_loader[config.data.name](config)
    
    # ust test edge_index as full_A
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    result_dict = {}
    # Access individual parameters
    walk_length = config.model.node2vec.walk_length
    num_walks = config.model.node2vec.num_walks
    p = config.model.node2vec.p
    q = config.model.node2vec.q
    workers = config.model.node2vec.workers
    use_rejection_sampling = config.model.node2vec.use_rejection_sampling
    embed_size = config.model.node2vec.embed_size
    ws = config.model.node2vec.window_size
    iter = config.model.node2vec.iter

    # model 
    for use_heuristic in ['node2vec']:
        G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.DiGraph())
        model = Node2Vec(G, walk_length=walk_length, 
                         num_walks=num_walks,
                        p=p, 
                        q=q, 
                        workers=workers, 
                        use_rejection_sampling=use_rejection_sampling)
        
        model.train(embed_size=embed_size,
                    window_size = ws, 
                    iter = iter)
        
        embeddings = model.get_embeddings()

    return eval_embed(embeddings, splits)

from sklearn.manifold import TSNE
from heuristic.pubmed_heuristic import get_pubmed_casestudy   




# main function 
if __name__ == "__main__":
    args = parse_args()
    # Load config file
    
    cfg = set_cfg(args)
    cfg.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    y_pred, results_acc, results_mrr, y_test  = eval_pubmed_mrr_acc(cfg)

    root = FILE_PATH + 'results'
    acc_file = root + f'/{cfg.data.name}_acc.csv'
    mrr_file = root +  f'/{cfg.data.name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    append_acc_to_excel(results_acc, acc_file, cfg.data.name)
    append_mrr_to_excel(results_mrr, mrr_file)

