import os
import sys
from typing import Dict
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization
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
from utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from heuristic.eval import (
    evaluate_auc,
    evaluate_hits,
    evaluate_mrr,
    get_metric_score,
    get_prediction
)
from ge import Node2Vec
from yacs.config import CfgNode as CN
import networkx as nx
import yaml
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
import graphgps
from heuristic.pubmed_heuristic import get_pubmed_casestudy
from heuristic.cora_heuristic import get_cora_casestudy
from heuristic.arxiv2023_heuristic import get_raw_text_arxiv_2023

import numba
from numba import njit, typed, types

import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import random 
from numba.typed import List

FILE_PATH = get_git_repo_root_path() + '/'

data_loader = {
    'cora': get_cora_casestudy,
    'pubmed': get_pubmed_casestudy,
    'arxiv_2023': get_raw_text_arxiv_2023
}


def set_cfg(args):
    with open(args.cfg_file, "r") as f:
        cfg = CN.load_cfg(f)
    return cfg

def node2vec(adj, embedding_dim=64, walk_length=30, walks_per_node=10,
                  workers=8, window_size=10, num_neg_samples=1, p=4, q=1):
    """
    参数说明
    -------------
    adj : 图的邻接矩阵
    embedding_dim : 图嵌入的维度
    walk_length : 随机游走的长度
    walks_per_node : 每个节点采样多少个随机游走
    workers: word2vec模型使用的线程数量
    window_size: word2vec模型中使用的窗口大小
    num_neg_samples : 负样本的数量
    p: node2vec的p参数
    q: node2vec的q参数
    """
    walks = sample_n2v_random_walks(adj, walk_length, walks_per_node, p=p, q=q) # 利用随机游走提取共现信息
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=embedding_dim, 
                     negative=num_neg_samples, compute_loss=True)   # 映射函数、重构器、目标
    embedding = model.wv.vectors[np.fromiter(map(int, model.wv.index_to_key), np.int32).argsort()] # 从词向量中取出节点嵌入
    return embedding

def sample_n2v_random_walks(adj, walk_length, walks_per_node, p, q):
    """
    返回值的类型
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        采样后的随机游走
    """
    adj = sp.csr_matrix(adj)
    random_walks = _n2v_random_walk(adj.indptr,
                                    adj.indices,
                                    walk_length,
                                    walks_per_node,
                                    p,
                                    q)
    return random_walks 

#@numba.jit(nopython=True)
from numba.typed import List 
from numba import njit 

# def _n2v_random_walk(indptr,
#                     indices,
#                     walk_length,
#                     walks_per_node,
#                     p,
#                     q):
#     N = len(indptr) - 1 # num of nodes
#     final_walks = List() # all walks of one node 
#     for _ in range(walks_per_node):
#         for n_iter in range(N):
#             walk = np.zeros(walk_length+1, dtype=np.int32)
#             walk[0] = n_iter          

#             visited_set = None
#             for il in range(walk_length):
#                 # v_iter 's neighbors
#                 neighbors = indices[indptr[n_iter]:indptr[n_iter+1]]
                
#                 sample_idx_arr = np.zeros(len(neighbors)+1, dtype=np.int32)
#                 sample_prob_arr = np.zeros(len(neighbors)+1, dtype=np.float32)
                
#                 for i, samples in enumerate(neighbors):
#                     sample_idx_arr[i] = samples
                
#                     if visited_set is not None:
#                         if samples in visited_set:
#                             sample_prob_arr[i] = 1
#                         else:
#                             sample_prob_arr[i] = 1/q
#                     else:
#                         sample_prob_arr[i] = 1/q 

#                 visited_set = neighbors.copy()
#                 sample_idx_arr[-1] = n_iter
#                 sample_prob_arr[-1] = 1/p
                    
#                 sample_prob_arr = sample_prob_arr / np.sum(sample_prob_arr)
#                 n_iter = random_choice(sample_idx_arr, sample_prob_arr)
                
#                 walk[il+1] = n_iter

#             final_walks.append(walk)
#     return np.array(final_walks)





# ### 用numba加速的版本
# # 建议debug阶段把下面这行注释掉，debug通过后再把取消下面这行的注释
@numba.jit(nopython=True)
def _n2v_random_walk(indptr,
                    indices,
                    walk_length,
                    walks_per_node,
                    p,
                    q):
    N = len(indptr) - 1 # 节点数量

    for _ in range(walks_per_node):
        for n_iter in range(N):
            walk = np.zeros(walk_length+1, dtype=np.int32)
            walk[0] = n_iter   

            visited_set = None
            for il in range(walk_length):
                # v_iter 's neighbors
                neighbors = indices[indptr[n_iter]:indptr[n_iter+1]]
                
                sample_idx_arr = np.zeros(len(neighbors)+1, dtype=np.int32)
                sample_prob_arr = np.zeros(len(neighbors)+1, dtype=np.float32)
                
                for i, samples in enumerate(neighbors):
                    sample_idx_arr[i] = samples
                
                    if visited_set is not None:
                        if samples in visited_set:
                            sample_prob_arr[i] = 1
                        else:
                            sample_prob_arr[i] = 1/q
                    else:
                        sample_prob_arr[i] = 1/q 

                visited_set = neighbors.copy()
                sample_idx_arr[-1] = n_iter
                sample_prob_arr[-1] = 1/p
                    
                sample_prob_arr = sample_prob_arr / np.sum(sample_prob_arr)
                n_iter = random_choice(sample_idx_arr, sample_prob_arr)
                
                walk[il+1] = n_iter
            yield walk # 用yield来构造一个generator


@numba.jit(nopython=True)
def random_choice(arr: np.int64, p):
    """
    params
    ----------
    arr : 1d tensor
    p : probability of each element in arr
    
    返回值
    -------
    samples : sampled samples 
    adopted from blog https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
    """
    return arr[np.searchsorted(np.cumsum(p), np.random.random(), side="right")]



def evaluate_node_classification(embedding_matrix, labels, train_mask, 
                                 test_mask, normalize_embedding=True, max_iter=1000):
        
    """训练一个线性模型（比如逻辑回归模型）来预测节点的标签
    
    返回值说明
    ----
    preds: 模型预测的标签
    test_acc: 模型预测的准确率
    """
    ######################################
    if normalize_embedding:
        embedding_matrix = normalize(embedding_matrix)
        
    # split embedding
    feature_train = embedding_matrix[train_mask]
    feature_test = embedding_matrix[test_mask]
    labels_train = labels[train_mask]
    labels_test = labels[test_mask]
    
    clf = LogisticRegression(solver='lbfgs',max_iter=max_iter, multi_class='auto')
    clf.fit(feature_train, labels_train)
    
    preds = clf.predict(feature_test)
    test_acc = accuracy_score(labels_test, preds)
    ######################################   
    return preds, test_acc

# 请大家完成下面这个测试函数
def evaluate_link_prediction(embed, splits, normalize_embedding=True, max_iter=1000):
        
    """训练一个线性模型（比如逻辑回归模型）来预测节点的标签
    
    返回值说明
    ----
    preds: 模型预测的标签
    test_acc: 模型预测的准确率
    """
    ######################################
    # train loop

    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    
    
    
    clf = LogisticRegression(solver='lbfgs',max_iter=max_iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    acc = clf.score(X_test, y_test)

    
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    results_mrr = {'node2vec_mrr': result_mrr}
    
    ######################################   
    return y_pred, results_acc, results_mrr, y_test 

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
        # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
        adj = to_scipy_sparse_matrix(full_edge_index)

        embedding = node2vec(adj, embedding_dim=64, p=0.5, q=0.5)
    


    return evaluate_link_prediction(embedding, splits)


# main function 
if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_scipy_sparse_matrix
    dataset = Planetoid(root='./data', name='Cora')# 将数据保存在data文件夹下
    data = dataset[0]
    adj = to_scipy_sparse_matrix(data.edge_index)

    embedding = node2vec(adj, embedding_dim=64, p=0.5, q=0.5)
    embedding.shape

    args = parse_args()
    # Load config file
    
    config = set_cfg(args)
    config.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(config.num_threads)
    
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
    
    # G = nx.from_scipy_sparse_matrix(full_A, create_using=nx.Graph())
    adj = to_scipy_sparse_matrix(full_edge_index)

    embed = node2vec(adj, embedding_dim=64, p=0.5, q=0.5)
        
    
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    
    
    
    clf = LogisticRegression(solver='lbfgs',max_iter=iter, multi_class='auto')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    acc = clf.score(X_test, y_test)

    
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    results_mrr = {'node2vec_mrr': result_mrr}

    print(results_acc, results_mrr)
