import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd
from torch_geometric.data import Data
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_git_repo_root_path
# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs

FILE_PATH = get_git_repo_root_path() + '/'


def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./generated_dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    x = torch.tensor(data_X)
    edge_index = torch.tensor(data_edges)
    y = torch.tensor(data_Y)
    num_nodes = data.num_nodes
    
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        train_id = [i for i in train_id if not is_mistake[i]]
        val_id = [i for i in val_id if not is_mistake[i]]
        test_id = [i for i in test_id if not is_mistake[i]]

    train_mask = torch.tensor(
        [x in train_id for x in range(data.num_nodes)])
    val_mask = torch.tensor(
        [x in val_id for x in range(data.num_nodes)])
    test_mask = torch.tensor(
        [x in test_id for x in range(data.num_nodes)])

    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        
    dataset._data = data
    
    return dataset, data_pubid


def parse_pubmed():

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(FILE_PATH + 'dataset/PubMed_orig/data/Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(FILE_PATH+ 'dataset/PubMed_orig/data/Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_pubmed(use_text=False, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
    if not use_text:
        return data, None

    f = open(FILE_PATH + 'dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)
    return data, text

# TEST CODE
# if __name__ == '__main__':
#     data, text = get_raw_text_pubmed(use_text=True)
#     print(data)
#     print(text[0])
    
#     data_A, data_X, data_Y, data_pubid, edge_index = parse_pubmed()
#     print(edge_index)
    
#     dataset, data_pubid = get_pubmed_casestudy(corrected=False, SEED=0)
#     print(dataset)