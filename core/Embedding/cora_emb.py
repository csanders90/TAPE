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
from torch_geometric.graphgym.config import (cfg, set_cfg)
import graphgps 


FILE_PATH = get_git_repo_root_path() + '/'


from sklearn.manifold import TSNE



def eval_cora_mrr_acc(config) -> None:
    
    dataset, data_cited, splits = get_cora_casestudy(undirected = True,
                                                include_negatives = True,
                                                val_pct = 0.15,
                                                test_pct = 0.05,
                                                split_labels = False
                                                )
    
    edge_index = splits['test'].edge_index
    edge_weight = torch.ones(edge_index.size(1))
    num_nodes = dataset._data.num_nodes

    m = construct_sparse_adj(edge_index)
    plot_coo_matrix(m, f'cora_test_edge_index.png')
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

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


    for use_emb in ['node2vec']:
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
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
            
        
        return  eval_embed(embeddings, splits)


        
    
        
def eval_embed(embed,  splits, visual=True):
    """train the classifier and return the pred

    Args:
        embed (np.array): embedding vector generated from the graph
        test_index (torch.tensor): test edge index
        edge_label (torch.tensor): test edge label

    Returns:
        prediction for edges 
    """
    # train loop
    embed = np.asarray(list(embed.values()))
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label
    # dot product
    X_train = embed[X_train_index]
    X_train = np.multiply(X_train[:, 1], (X_train[:, 0]))
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    # dot product 
    X_test = embed[X_test_index]
    X_test = np.multiply(X_test[:, 1], (X_test[:, 0]))
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(100,), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001, 
                        batch_size='auto', 
                        learning_rate_init=0.001, 
                        power_t=0.5, 
                        max_iter=200, 
                        shuffle=True, 
                        random_state=None, 
                        tol=0.0001, 
                        verbose=False, 
                        warm_start=False, 
                        momentum=0.9, 
                        nesterovs_momentum=True, 
                        early_stopping=False, 
                        validation_fraction=0.1, 
                        epsilon=1e-08, 
                        n_iter_no_change=10, 
                        max_fun=15000).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = clf.score(X_test, y_test)
    
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    results_mrr = {'node2vec_mrr': result_mrr}
    
    if visual:
        model = TSNE(n_components=2,
                    init="random",
                    random_state=0,
                    perplexity=100,
                    n_iter=300)
        node_pos = model.fit_transform(X_test)

        color_dict = {'0.0': 'r', '1.0': 'b'}
        color = [color_dict[str(i)] for i in y_test.tolist()]
        plt.figure()
        for idx in range(len(node_pos)):
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c=color[idx])
        plt.legend()
        plt.savefig(f'cora_node2vec.png')
    return y_pred, results_acc, results_mrr, y_test 


        
def parse_cora():
    # load original data from cora orig without text features
    path = FILE_PATH + 'dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_cora_casestudy(SEED=0,
                        undirected = True,
                        include_negatives = True,
                        val_pct = 0.15,
                        test_pct = 0.05,
                        split_labels = True
                        ) -> InMemoryDataset:
    
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./dataset', data_name,
                        transform=T.NormalizeFeatures())

    data = dataset[0]
    # check is data has changed and try to return dataset
    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).long()
    y = torch.tensor(data_Y).long()
    num_nodes = len(data_Y)

    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        
    dataset._data = data

    undirected = data.is_undirected()

    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}

    return dataset, data_citeid, splits

def set_cfg(args):
    with open(args.cfg_file, "r") as f:
        cfg = CN.load_cfg(f)
    return cfg



# main function 
if __name__ == "__main__":
    args = parse_args()
    # Load config file
    
    cfg = set_cfg(args)
    cfg.merge_from_list(args.opts)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    NAME = 'cora'
    y_pred, results_acc, results_mrr, y_test  = eval_cora_mrr_acc(cfg)

    root = FILE_PATH + 'results'
    acc_file = root + f'/{NAME}_acc.csv'
    mrr_file = root +  f'/{NAME}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    append_acc_to_excel(results_acc, acc_file, NAME)
    append_mrr_to_excel(results_mrr, mrr_file)

