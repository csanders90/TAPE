import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import random 
from tqdm import tqdm 
import numpy as np 

def plot_pessimistic_rank(y_pred_pos, y_pred_neg):
    plt.figure(figsize=(12, 8))
    # Plot distributions of probabilities
    plt.subplot(1, 2, 1)
    plt.imshow((y_pred_neg >= y_pred_pos))
    plt.xlabel('Sample Index')
    plt.ylabel('Neg >= Pos')
    plt.title('Distribution of Optimistic Rank')

    plt.subplot(1, 2, 2)
    plt.imshow((y_pred_neg > y_pred_pos))
    plt.xlabel('Sample Index')
    plt.ylabel('Pos > Pos')
    plt.title('Distribution of Pessimistic Rank')
    plt.savefig('optimistic_rank.png')
    


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    results = {}
    valid_auc = round(valid_auc, 4)
    results['AUC'] = valid_auc
    valid_ap = average_precision_score(val_true, val_pred)
    valid_ap = round(valid_ap, 4)
    results['AP'] = valid_ap
    return results

def error_mrr_pos(y_pred_pos, pos_edge_index, y_pred_neg, k_list):
    # add one axis
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    # ~> the positive is ranked last among those with equal score
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    plot_rank_list(ranking_list, 'pos')
    len = ranking_list.shape[0]
    result = {}
    for k in k_list:
        result[f'mrr_hit{round(k/len, 2)}'] = (ranking_list <= k).sum(0)/len
    
    
    ranking_list = ranking_list.cpu().numpy().astype(int)
    analysis_interval = [0.9, 1.0]
    rank_interval = ranking_list.min() + ((ranking_list.max() - ranking_list.min()) * np.array(analysis_interval)).astype(int)
    pos_edge_index_err = pos_edge_index[rank_interval[0]:rank_interval[1]]
    pos_rank_err = ranking_list[rank_interval[0]:rank_interval[1]]
        
    return result, pos_edge_index_err, pos_rank_err

# Define the reciprocal ranks for each query

def error_mrr_neg(y_pred_neg, neg_edge_index, y_pred_pos, k_list):
    # calculate ranks
    y_pred_neg = y_pred_neg.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg <= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg < y_pred_pos).sum(dim=1)
    
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) - 1
    
    plot_rank_list(ranking_list, 'neg')
    
    result = {}
    for k in k_list:
        result[f'mrr_hit{k}'] = (ranking_list <= k).sum(0).to(torch.float)/y_pred_pos.shape[0]
    
    ranking_list = ranking_list.cpu().numpy().astype(int)
    analysis_interval = [0, 0.1]
    rank_interval = ranking_list.min() + ((ranking_list.max() - ranking_list.min()) * np.array(analysis_interval)).astype(int)
    neg_edge_index_err = neg_edge_index[rank_interval[0]:rank_interval[1]]
    neg_rank_err = ranking_list[rank_interval[0]:rank_interval[1]]
     
    return result, neg_edge_index_err, neg_rank_err


def get_metric_invariant(pos_test_pred, pos_edge_index, neg_test_pred, neg_edge_index, k_list):

    k_rank = [int(k * neg_test_pred.shape[0]) for k in k_list] 

    mrr_pos2neg, pos_edge_index_err, pos_rank_err = error_mrr_pos(pos_test_pred, pos_edge_index, neg_test_pred.repeat(pos_test_pred.size(0), 1), k_rank)
    mrr_neg2pos, neg_edge_index_err, neg_rank_err = error_mrr_neg(neg_test_pred, neg_edge_index, pos_test_pred.repeat(neg_test_pred.size(0), 1), k_rank)
    
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_test = evaluate_auc(test_pred, test_true)

    result_auc_test['AUC'] = (result_auc_test['AUC'])
    result_auc_test['AP'] = (result_auc_test['AP'])
    
    return mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err


def plot_rank_list(position: torch.tensor, label: str, sample_size: int=20):
    """ Plot the ranks for each query in the dataset """
    pos = position.clone()
    
    index = np.random.randint(1, pos.shape[0], size=sample_size)
    pos = pos[index]
    # renormalize the ranks to be between 1 and 50
    pos = ((pos - pos.min())*(50 - 1)/(pos.max() - pos.min())).to(torch.int32)
    
    # Create the plot
    plt.figure()
    fig, axes = plt.subplots(1, pos.shape[0], figsize=(20, 5))  # Adjust figsize for better visualization

    for i, ax in tqdm(enumerate(axes)):
        # Create a bar chart for each query
        ax.barh(range(int(pos.max())), [0.5] * int(pos.max()), color='lightgrey', height=0.5)
        ax.barh(pos[i], 0.5, color='lightcoral', height=0.5)
        ax.invert_yaxis()

    for ax in axes:
        ax.axis('off')

    # Random ID for the plot filename
    id = random.randint(1, 100)
    plt.savefig(f'plot_rank_{label}_{id}.png')
    

def find_optimal_threshold(pos_probs, neg_probs, thresholds=None):
    if thresholds is None:
        # Generate thresholds from 0 to 1 with a step size
        thresholds = np.linspace(0, 1, num=100)
    
    y_true = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    y_scores = np.concatenate([pos_probs, neg_probs])
    
    best_threshold = 0
    best_accuracy = 0
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy

