import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import scipy.sparse as ssp
import torch
import argparse
import wandb
import time
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from ogb.linkproppred import Evaluator

from heuristic.eval import get_metric_score
from graph_embed.ge.models import Node2Vec
from tune_utils import save_parameters
from data_utils.load_data_lp import get_edge_split
from data_utils.load import load_graph_lp as data_loader
from data_utils.graph_stats import plot_coo_matrix, construct_sparse_adj
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from networkx import from_scipy_sparse_matrix as from_scipy_sparse_array
# Set the file path for the project
FILE_PATH = get_git_repo_root_path() + '/'


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    parser.add_argument('--sweep', type=str, required=False,
                        default='core/yamls/cora/gcns/ncn.yaml',
                        help='The configuration file path.')
    return parser.parse_args()


def eval_embed(embed, splits, visual=True):
    """Trains the classifier and returns predictions.

    Args:
        embed (np.array): Embedding vectors generated from the graph.
        splits (dict): Train/test edge splits.
        visual (bool): Whether to visualize the embeddings using TSNE.

    Returns:
        tuple: y_pred, results_acc, results_mrr, y_test
    """
    embed = np.asarray(list(embed.values()))
    X_train_index, y_train = splits['train'].edge_label_index.T, splits['train'].edge_label

    # Compute dot products for training and testing
    X_train = np.multiply(embed[X_train_index[:, 1]], embed[X_train_index[:, 0]])
    X_test_index, y_test = splits['test'].edge_label_index.T, splits['test'].edge_label
    X_test = np.multiply(embed[X_test_index[:, 1]], embed[X_test_index[:, 0]])

    # Train classifier
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = clf.score(X_test, y_test)

    # Evaluate predictions
    results_acc = {'node2vec_acc': acc}
    pos_test_pred = torch.tensor(y_pred[y_test == 1])
    neg_test_pred = torch.tensor(y_pred[y_test == 0])
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    result_mrr = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
    result_mrr['ACC'] = acc
    results_mrr = {'node2vec_mrr': result_mrr}

    # Visualization
    if visual:
        tsne = TSNE(n_components=2, random_state=0, perplexity=100, n_iter=300)
        node_pos = tsne.fit_transform(X_test)

        color_dict = {'0.0': 'r', '1.0': 'b'}
        colors = [color_dict[str(label)] for label in y_test.tolist()]
        plt.figure()
        for idx in range(len(node_pos)):
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c=colors[idx])
        plt.legend()
        plt.savefig('cora_node2vec.png')

    return y_pred, results_acc, results_mrr, y_test


def eval_mrr_acc(cfg) -> None:
    """Loads the graph data and evaluates embeddings using MRR and accuracy."""
    dataset, _ = data_loader[cfg.data.name](cfg)
    undirected = dataset.is_undirected()
    splits = get_edge_split(dataset, undirected, cfg.data.device,
                            cfg.data.split_index[1], cfg.data.split_index[2],
                            cfg.data.include_negatives, cfg.data.split_labels)

    # Create the full adjacency matrix from test edges
    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset.num_nodes
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), 
                             (full_edge_index[0], full_edge_index[1])), 
                             shape=(num_nodes, num_nodes))

    # Visualize the test edge adjacency matrix
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, 'test_edge_index.png')

    # Extract Node2Vec parameters from config
    node2vec_params = cfg.model.node2vec
    G = from_scipy_sparse_array(full_A, create_using=nx.Graph())

    model = Node2Vec(G, walk_length=node2vec_params.walk_length, 
                        num_walks=node2vec_params.num_walks,
                        p=node2vec_params.p, 
                        q=node2vec_params.q, 
                        workers=node2vec_params.workers, 
                        use_rejection_sampling=node2vec_params.use_rejection_sampling)
    
    start = time.time()
    epochs = 5
    model.train(embed_size=node2vec_params.embed_size,
                window_size=node2vec_params.window_size, 
                iter=node2vec_params.max_iter,
                epochs = epochs)
    end = time.time()
    
    embeddings = model.get_embeddings()
    root = os.path.join(FILE_PATH, 'results')
    save_parameters(root, model, start, end, epochs)

    return eval_embed(embeddings, splits)


if __name__ == "__main__":
    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    # Run the evaluation
    y_pred, results_acc, results_mrr, y_test = eval_mrr_acc(cfg)

    # Save the results
    root = os.path.join(FILE_PATH, 'results')
    os.makedirs(root, exist_ok=True)

    acc_file = os.path.join(root, f'{cfg.data.name}_acc.csv')
    mrr_file = os.path.join(root, f'{cfg.data.name}_mrr.csv')

    run_id = wandb.util.generate_id()
    append_acc_to_excel(run_id, results_acc, acc_file, cfg.data.name, 'node2vec')
    append_mrr_to_excel(run_id, results_mrr, mrr_file, cfg.data.name, 'node2vec')
