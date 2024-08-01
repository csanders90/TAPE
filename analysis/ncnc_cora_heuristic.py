import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import organization

import pandas as pd
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch 
from visual import find_opt_thres, get_metric_invariant, load_results, plot_pos_neg_histogram
from matplotlib import pyplot as plt
from visual import load_csv
from torch_geometric import utils
import os
import sys
from core.heuristic.lsf import CN, AA, RA, InverseRA
from core.heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close, SymPPR
from core.data_utils.load import load_data_lp
from core.data_utils.lcc import use_lcc
from core.graphgps.utility.utils import init_cfg_test
from torch_geometric.utils import to_torch_coo_tensor, coalesce
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

notebook_dir = os.getcwd()  
target_dir = os.path.abspath(os.path.join(notebook_dir, '..'))

sys.path.insert(0, target_dir)

evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')

# Example usage
FILE_PATH = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/educational_demo/'
ncnc_cora_path = FILE_PATH + 'err_ncnc_llama/ncnc-cora_AUC_0.9669_MRR_0.5275.csv'

P1, P2, pos_index, neg_index = load_results(ncnc_cora_path)
best_thres_llama, best_acc_llama, pos_pred_llama, neg_pred_llama = find_opt_thres(P1, P2)


def tensor_to_csr_matrix(edge_index: torch.tensor):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    num_nodes = edge_index.max().item() + 1  # Assuming node indices are 0-based
    adj_coo = csr_matrix((np.ones(src.shape[0]), (src, dst)), shape=(num_nodes, num_nodes))
    return adj_coo


best_thres, best_acc, pos_pred, neg_pred = find_opt_thres(P1, P2)

plot_pos_neg_histogram(pos_pred, neg_pred, best_thres)

k_list  = [0.1, 0.2, 0.3, 0.5, 1]

mrr_pos2neg, mrr_neg2pos, result_auc_test, pos_edge_index_err, pos_rank_err, neg_edge_index_err, neg_rank_err = get_metric_invariant(P1, pos_index, P2, neg_index, k_list)

print(mrr_pos2neg)
print(result_auc_test)
print(mrr_neg2pos)

cfg = init_cfg_test()

splits, text, data = load_data_lp[cfg.data.name](cfg.data)
# new_data, lcc_index, G = use_lcc(data)
G = utils.to_dense_adj(data.edge_index).squeeze()
G



heuristic_feat = []
for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
    edge_index = tensor_to_csr_matrix(data.edge_index)
    print(edge_index.shape)
    scores, edge_index = eval(use_lsf)(edge_index, pos_edge_index_err.T)
    print(f'{use_lsf}: {scores}')
    heuristic_feat.append(scores.numpy())
    
for use_gsf in ['Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR']:
    edge_index = tensor_to_csr_matrix(data.edge_index)
    print(edge_index.shape)
    scores, edge_index = eval(use_gsf)(edge_index, pos_edge_index_err.T)
    print(f'{use_gsf}: {scores}')
    heuristic_feat.append(scores.numpy())

feat_pro = []
for rows in pos_edge_index_err:
    feat_pro.append((data.x[rows[0]] @ data.x[rows[1]]).item())
feat_pro = np.array(feat_pro)
    
heuristic_feat.append(feat_pro)
print(heuristic_feat)

heuristic_feat = np.vstack(heuristic_feat)

row_sums = heuristic_feat.sum(axis=1, keepdims=True)
normalized_feat = heuristic_feat / row_sums

heuristic_feat = normalized_feat.min(axis=0)

feature_names = ['CN', 'AA', 'RA', 'InverseRA', 'Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR', 'feat_pro']
index_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

plt.figure(figsize=(8, 8))
plt.imshow(normalized_feat, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xticks(ticks=np.arange(len(index_names)), labels=index_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Feature Map Heatmap')
plt.tight_layout()
plt.savefig('normalized_feature_map.png')



heuristic_feat = []
for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
    edge_index = tensor_to_csr_matrix(data.edge_index)
    print(edge_index.shape)
    scores, edge_index = eval(use_lsf)(edge_index, neg_edge_index_err.T)
    print(f'{use_lsf}: {scores}')
    heuristic_feat.append(scores.numpy())
    
for use_gsf in ['Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR']:
    edge_index = tensor_to_csr_matrix(data.edge_index)
    print(edge_index.shape)
    scores, edge_index = eval(use_gsf)(edge_index, neg_edge_index_err.T)
    print(f'{use_gsf}: {scores}')
    heuristic_feat.append(scores.numpy())

feat_pro = []
for rows in neg_edge_index_err:
    feat_pro.append((data.x[rows[0]] @ data.x[rows[1]]).item())
feat_pro = np.array(feat_pro)
heuristic_feat.append(feat_pro)

print(heuristic_feat)

# heuristic_feat = np.vstack(heuristic_feat)

# row_sums = heuristic_feat.sum(axis=1, keepdims=True)
# normalized_feat = heuristic_feat / row_sums

# heuristic_feat = normalized_feat.min(axis=0)

normalized_feat = heuristic_feat
feature_names = ['CN', 'AA', 'RA', 'InverseRA', 'Ben_PPR', 'shortest_path', 'katz_apro', 'katz_close', 'SymPPR', 'feat_pro']
index_names = ['0']

plt.figure(figsize=(8, 8))
plt.imshow(normalized_feat, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xticks(ticks=np.arange(len(index_names)), labels=index_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Feature Map Heatmap')
plt.tight_layout()
plt.savefig('normalized_neg_feature_map.png')
