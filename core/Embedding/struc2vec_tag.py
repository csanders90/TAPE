import os, sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.linkproppred import Evaluator
from yacs.config import CfgNode as CN
from heuristic.eval import get_metric_score
from heuristic.pubmed_heuristic import get_pubmed_casestudy
from heuristic.cora_heuristic import get_cora_casestudy
from heuristic.arxiv2023_heuristic import get_raw_text_arxiv_2023
from lpda.adjacency import plot_coo_matrix, construct_sparse_adj
from utils import (
    get_git_repo_root_path,
    append_acc_to_excel,
    append_mrr_to_excel
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ge.classify import read_node_label, Classifier
from ge import Struc2Vec
import itertools


data_loader = {
    'cora': get_cora_casestudy,
    'pubmed': get_pubmed_casestudy,
    'arxiv_2023': get_raw_text_arxiv_2023
}

def evaluate_embeddings(embeddings):

    X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)

    tr_frac = 0.8

    print("Training classifier using {:.2f}% nodes...".format(

        tr_frac * 100))

    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())

    clf.split_train_evaluate(X, Y, tr_frac)





def plot_embeddings(embeddings,):

    X, Y = read_node_label('../data/flight/labels-brazil-airports.txt',skip_head=True)

    emb_list = []

    for k in X:

        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)



    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)



    color_idx = {}

    for i in range(len(X)):

        color_idx.setdefault(Y[i][0], [])

        color_idx[Y[i][0]].append(i)



    for c, idx in color_idx.items():

        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    plt.legend()

    plt.show()

if __name__ == "__main__":
    # G = nx.read_edgelist('../data/flight/brazil-airports.edgelist', create_using=nx.DiGraph(), nodetype=None,
    #                      data=[('weight', int)])
    #
    # # nx.draw(G, node_size=10, font_size=10, font_color="blue", font_weight="bold")
    # # plt.show()
    # # print(G)
    #
    # model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
    # model.train(embed_size=64)
    # embeddings = model.get_embeddings()
    #
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)

    # import pandas as pd

    # df = pd.DataFrame()
    # df['source'] = [str(i) for i in [0, 1, 2, 3, 4, 4, 6, 7, 7, 9]]
    # df['target'] = [str(i) for i in [1, 4, 4, 4, 6, 7, 5, 8, 9, 8]]

    # G = nx.from_pandas_edgelist(df, create_using=nx.Graph())

    FILE_PATH = get_git_repo_root_path() + '/'

    cfg_file = FILE_PATH + "core/configs/pubmed/node2vec.yaml"
    # # Load args file
    with open(cfg_file, "r") as f:
        cfg = CN.load_cfg(f)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    if torch.cuda.is_available():
        # Get the number of available CUDA devices
        num_cuda_devices = torch.cuda.device_count()

        if num_cuda_devices > 0:
            # Set the first CUDA device as the active device
            torch.cuda.set_device(0)
            device = 'cuda'
    else:
        device = 'cpu'
        
    dataset, data_cited, splits = data_loader[cfg.data.name](cfg)
    
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

    adj = to_scipy_sparse_matrix(full_edge_index)

    G = nx.from_scipy_sparse_array(adj)
    
    model = Struc2Vec(G, 10, 80, workers=20, verbose=40, )
    model.train(embed_size=2)

    embeddings = model.get_embeddings()
    # print(embeddings)
    x, y = [], []
    print(sorted(embeddings.items(), key=lambda x: x[0]))
    for k, i in embeddings.items():
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y)
    plt.savefig('structure2vec.png')