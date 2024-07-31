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
    plt.figure(figsize=(10, 6))
    plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    plt.title('Histogram of Tensor Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('pos_2neg_hist.png')
    # Define the error range (x-axis range considered as error)
    norm_interval = [0.8, 1.0] 
    error_ranges = [[int(i*ranking_list.max().item())-1 for i in norm_interval]] # Example error ranges, adjust as needed
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    
    for i in range(len(patches)):
        bin_center = 0.5 * (bins[i] + bins[i+1])
        for error_range in error_ranges:
            if error_range[0] <= bin_center <= error_range[1]:
                patches[i].set_facecolor('r') 
                break  
            else:
                patches[i].set_facecolor('b') 
    # Add labels and title
    plt.title('Histogram of Tensor Distribution with Error Ranges Highlighted, Pos ranked lower than Neg')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('pos_2neg_hist.png')

    len_rank = ranking_list.shape[0]
    result = {}
    for k in k_list:
        result[f'mrr_hit{round(k/len_rank, 2)}'] = (ranking_list <= k).sum(0)/len_rank
    
    ranking_list = ranking_list.cpu().numpy().astype(int)
    # rank_interval = ranking_list.min() + ((ranking_list.max() - ranking_list.min()) * np.array(norm_interval)).astype(int)
    error_mask = np.zeros(ranking_list.shape[0], dtype=bool)
    for errors in error_ranges:
        error_mask = error_mask | ((ranking_list > errors[0]) & (ranking_list < errors[1]))
        
    pos_edge_index_err = pos_edge_index[error_mask, :]
    pos_rank_err = ranking_list[error_mask]
        
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
    
    len_rank = ranking_list.shape[0]
    result = {}
    for k in k_list:
        result[f'mrr_hit{round(k/len_rank, 2)}'] = (ranking_list <= k).sum(0)/len_rank
    
    plt.figure(figsize=(10, 6))
    plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    plt.title('Histogram of Tensor Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('pos_2neg_hist.png')

    norm_interval = [0.0, 0.4] 
    error_ranges = [[int(i*ranking_list.max().item())-1 for i in norm_interval]] # Example error ranges, adjust as needed
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(ranking_list.numpy(), bins=30, edgecolor='black')
    for i in range(len(patches)):
        bin_center = 0.5 * (bins[i] + bins[i+1])
        for error_range in error_ranges:
            if error_range[0] <= bin_center <= error_range[1]:
                patches[i].set_facecolor('r') 
                break  
            else:
                patches[i].set_facecolor('b') 
    plt.title('Histogram of Tensor Distribution with Error Ranges Highlighted, Neg ranked lower than Pos')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('pos_2neg_hist.png')
    
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

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: There was an issue parsing the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    import os
    import sys
    from IPython.display import display, Markdown
    import pandas as pd
    from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
    import torch 
    from visual import find_optimal_threshold, get_metric_invariant
    from matplotlib import pyplot as plt
    
    # Assuming your target directory is one level up from the current working directory
    notebook_dir = os.getcwd()  
    target_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

    sys.path.insert(0, target_dir)
    from core.data_utils.load import load_data_lp
    from core.graphgps.utility.utils import init_cfg_test

    cfg = init_cfg_test()
    splits, text, data = load_data_lp[cfg.data.name](cfg.data)
    

    # Example usage
    FILE_PATH = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/educational_demo/'
    file_path = FILE_PATH + 'err_ncnc_llama/llama-cora-origin_dot_AUC_0.903_MRR_0.228.csv'
    data = load_csv(file_path)

    # evaluator = Evaluator(name='ogbl-collab')
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    pred = data['pred'].tolist()
    gr = data['gr'].tolist()
    source = data['edge_index0'].tolist()
    target = data['edge_index1'].tolist()

    pos_mask = data[data['gr'] == 1.0]
    pos_index = pos_mask[['edge_index0', 'edge_index1']]

    neg_mask = data[data['gr'] == 0.0]
    neg_index = neg_mask[['edge_index0', 'edge_index1']]

    P2 = neg_mask['pred'].to_numpy()
    P1 = pos_mask['pred'].to_numpy()

    plt.figure(figsize=(12, 8))

    # Plot distributions of probabilities
    plt.hist(P2, bins=100, alpha=0.5, color='blue', label='Positive Class')
    plt.hist(P1, bins=100, alpha=0.5, color='red', label='Negative Class')
    best_threshold, best_accuracy= find_optimal_threshold(P2, P1)

    plt.axvline(best_threshold, color='green', linestyle='--', label=f'Optimal Threshold = {best_threshold:.2f}')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Probability Distributions with Optimal Threshold')
    plt.savefig('optimal_threshold.png')


    k_list  = [0.1, 0.2, 0.3, 0.5, 1]
    pos_index = torch.tensor(pos_index.to_numpy())
    neg_index = torch.tensor(neg_index.to_numpy())
    P1 = torch.tensor(P1)
    P2 = torch.tensor(P2)
    mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err = get_metric_invariant(P1, pos_index, P2, neg_index, k_list)

    print(mrr_pos2neg)
    print(result_auc_test)
    print(mrr_neg2pos)

    for row in pos_edge_index_err:
        src = text[row[0]]
        tgt = text[row[1]]
        display(Markdown(f"**Source:** {src}  \n**Target:** {tgt}"))