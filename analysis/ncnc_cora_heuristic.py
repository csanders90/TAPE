import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

import pandas as pd
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch 
from visual import find_opt_thres, get_metric_invariant, load_results
from matplotlib import pyplot as plt
from visual import load_csv
from torch_geometric import utils
import os
import sys
from core.heuristic.lsf import CN, AA, RA, InverseRA
from core.heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR

# Assuming your target directory is one level up from the current working directory
notebook_dir = os.getcwd()  
target_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

sys.path.insert(0, target_dir)
from core.data_utils.load import load_data_lp
from core.data_utils.lcc import use_lcc
from core.graphgps.utility.utils import init_cfg_test

# evaluator = Evaluator(name='ogbl-collab')
evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')

# Example usage
FILE_PATH = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/educational_demo/'
file_path = FILE_PATH + 'err_ncnc_llama/ncnc-cora_AUC_0.9669_MRR_0.5275.csv'
P1, P2, pos_index, neg_index = load_results(file_path)
best_thres_llama, best_acc_llama, pos_pred_llama, neg_pred_llama = find_opt_thres(P1, P2)

plt.figure(figsize=(12, 8))
# Plot distributions of probabilities
plt.hist(P2, bins=100, alpha=0.5, color='blue', label='Neg Class')
plt.hist(P1, bins=100, alpha=0.5, color='red', label='Pos Class')
best_thres, best_acc, pos_pred, neg_pred = find_opt_thres(P1, P2)


plt.axvline(best_thres, color='green', linestyle='--', label=f'Optimal Threshold = {best_thres:.2f}')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.title('Probability Distributions with Optimal Threshold')
plt.savefig('optimal_threshold.png')
print(f'best_accuracy: {best_acc}, best_threshold: {best_thres}')

k_list  = [0.1, 0.2, 0.3, 0.5, 1]
pos_index = torch.tensor(pos_index)
neg_index = torch.tensor(neg_index)
P1 = torch.tensor(P1)
P2 = torch.tensor(P2)
mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err = get_metric_invariant(P1, pos_index, P2, neg_index, k_list)

print(mrr_pos2neg)
print(result_auc_test)
print(mrr_neg2pos)

cfg = init_cfg_test()

splits, text, data = load_data_lp[cfg.data.name](cfg.data)
# new_data, lcc_index, G = use_lcc(data)
G = utils.to_dense_adj(data.edge_index).squeeze()
G

result_acc = {}
for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
    scores, edge_index = eval(use_lsf)(A, pos_edge_index_err)
