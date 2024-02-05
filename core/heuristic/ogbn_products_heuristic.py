# TODO: Not tested error 
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os, sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_geometric.transforms import RandomLinkSplit
from data_utils.dataset import CustomPygDataset, CustomLinkDataset
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR
from data_utils.load_pubmed import get_raw_text_pubmed, get_pubmed_casestudy, parse_pubmed
import matplotlib.pyplot as plt
from lpda.adjacency import construct_sparse_adj
import scipy.sparse as ssp
from lpda.adjacency import plot_coo_matrix, plot_pos_neg_adj
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from eval import evaluate_auc, evaluate_hits, evaluate_mrr, get_metric_score, get_prediction
from utils import get_git_repo_root_path

def get_raw_text_products(use_text=False, seed=0):
    root_path = get_git_repo_root_path()
    data = torch.load(root_path + '/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(root_path + '/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric().to_torch_sparse_coo_tensor().coalesce().indices()
    
    if not use_text:
        text = None

    dataset = CustomLinkDataset('./dataset', 'ogbn-products', transform=T.NormalizeFeatures())
    dataset._data = data
    del dataset._data.n_id, dataset._data.adj_t, dataset._data.e_id

    undirected = data.is_undirected()
    undirected = True
    include_negatives = True
    val_pct = 0.15
    test_pct = 0.05
    
    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    
    return dataset, text, splits



def ogbn_products_acc():
    dataset, text, splits = get_raw_text_products(use_text=False, seed=0)
    print(dataset._data)
    
    test_split = splits['test']
    labels = test_split.edge_label
    test_index = test_split.edge_label_index
    
    edge_index = splits['train'].edge_index
    edge_weight = torch.ones(edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')

    
    pos_test_index = splits['train'].edge_index[:, splits['train'].edge_label == 1]
    neg_test_index = splits['train'].edge_index[:, splits['train'].edge_label == 0]

    
    pos_m = construct_sparse_adj(pos_test_index)
    plot_coo_matrix(pos_m, f'test_pos_index.png')
    neg_m = construct_sparse_adj(neg_test_index)
    plot_coo_matrix(neg_m, f'test_neg_index.png')
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')


    results = eval_heuristic_mrr_hits(A, 
                                    pos_test_index, 
                                    neg_test_index, 
                                    labels, 
                                    evaluator_hit, 
                                    evaluator_mrr)

    for key, result in results.items():
        train_hits, valid_hits, test_hits = result
        print(key)
        print( f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
    return 


def eval_heuristic_mrr_hits(full_A, 
                            pos_test_index, 
                            neg_test_index, 
                            labels, 
                            evaluator_hit, 
                            evaluator_mrr):
    """eval heuristic using mrr and hits from ogb.evaluator"""

    result_heuristic = {}
    for use_heuristic in ['CN', 'AA', 'RA', 'InverseRA']:
        
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
        
    
        results = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_heuristic.update({f'{use_heuristic}': results})

    for use_heuristic in ['Ben_PPR', 'SymPPR']:
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)

        results = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_heuristic.update({f'{use_heuristic}': results})
        
        pass 
    
    for use_heuristic in ['shortest_path', 'katz_apro', 'katz_close']:
        pos_test_pred = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_index)
    
        results = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_heuristic.update({f'{use_heuristic}': results})
    
    with open('plots/ogbn-products/heuristic.json', 'w') as f:
        json.dump(result_heuristic, f, indent=4)
    
    return result_heuristic
    

def eval_heuristic_acc(A, test_index, labels):
    
    test_pred = {}
    for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
        pos_test_pred, edge_index = eval(use_lsf)(A, test_index)
         
        plt.figure()
        plt.plot(pos_test_pred)
        plt.plot(labels)
        plt.savefig(f'cora_{use_lsf}.png')
        
        acc = torch.sum(pos_test_pred == labels) / pos_test_pred.shape[0]
        
        
        print(f" {use_lsf}: accuracy: {acc}")
        test_pred.update({f'{use_lsf}_acc': acc})
        test_pred.update()
            
            
    # 'shortest_path', 'katz_apro', 'katz_close', 'Ben_PPR'
    for use_gsf in ['Ben_PPR', 'SymPPR']:
        scores, edge_reindex = eval(use_gsf)(A, test_index)
        
        # print(scores)
        # print(f" {use_heuristic}: accuracy: {scores}")
        pred = torch.zeros(scores.shape)
        cutoff = 0.05
        thres = scores.max()*cutoff 
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")
        test_pred.update({f'{use_lsf}_acc': acc})
        test_pred.update()
        
    
    for use_gsf in ['shortest_path', 'katz_apro', 'katz_close']:
        scores = eval(use_gsf)(A, test_index)
        
        pred = torch.zeros(scores.shape)
        thres = scores.min()*10
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")
        test_pred.update({f'{use_lsf}_acc': acc})
        test_pred.update()
        
    for key, val in test_pred.items():
        print(f'{key}: {val}')
        
    return test_pred
        




if __name__ == "__main__":
    ogbn_products_acc()
