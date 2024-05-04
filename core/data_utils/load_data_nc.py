import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
import torch
import random
import json
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.datasets import Planetoid
from utils import get_git_repo_root_path, time_logger

FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'
FILE_PATH = get_git_repo_root_path() + '/'

def get_raw_text_arxiv_2023_nc(use_text=False, 
                            seed=0):
    """
    Load and process the arxiv_2023 dataset for node classification task. 
    # TODO add data resource
    Args:
        use_text (bool, False): If True, the raw text of the dataset will be used. 
        seed (int, 0): The seed RNG for reproducibility of the dataset split. 
    
    Default Vars:
    graph path: dataset/arxiv_2023/graph.pt
    text path:'dataset/arxiv_2023_orig/paper_info.csv'

    Returns:
        data (Data): A PyTorch Geometric Data object containing the processed dataset. 
        The dataset is split into training, validation, and test sets. 
        The split is determined by the provided seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    data = torch.load(FILE_PATH + 'core/dataset/arxiv_2023/graph.pt')

    # split data
    data.num_nodes = len(data.y)
    num_nodes = data.num_nodes
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(num_nodes)])

    if not use_text:
        return data, None
    df = pd.read_csv(FILE_PATH + 'core/dataset/arxiv_2023_orig/paper_info.csv')
    text = [
        f'Title: {ti}\nAbstract: {ab}'
        for ti, ab in zip(df['title'], df['abstract'])
    ]
    return data, text


def get_cora_nc(SEED=0) -> InMemoryDataset:
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    dataset = Planetoid('./generated_dataset', 'cora',
                        transform=T.NormalizeFeatures())

    data = dataset[0]
    # check is data has changed and try to return dataset
    x = torch.tensor(data_X).float()
    edge_index = torch.LongTensor(data_edges).long()
    y = torch.tensor(data_Y).long()
    num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    test_id = np.sort(node_id[int(data.num_nodes * 0.8):])
    
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
    
    return dataset, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun

def parse_cora():
    path = f'{FILE_PATH}core/dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text, seed=0):
    data, data_citeid = get_cora_nc(seed)
    if not use_text:
        return data, None

    with open(f'{FILE_PATH}core/dataset/cora_orig/mccallum/cora/papers') as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = f'{FILE_PATH}core/dataset/cora_orig/mccallum/cora/extractions/'

    # for debug
    # save file list
    # with open('extractions.txt', 'w') as txt_file:
    #     # Write each file name to the text file
    #     for file_name in os.listdir(path):
    #         txt_file.write(file_name + '\n')

    text = []
    not_loaded = []
    i = 0
    for pid in data_citeid:
        fn = pid_filename[pid]
        try:
            if os.path.exists(path+fn): 
                pathfn = path+fn
            elif os.path.exists(path+fn.replace(":", "_")):
                pathfn = path+fn.replace(":", "_")
            elif os.path.exists(path+fn.replace("_", ":")):
                pathfn = path+fn.replace("_", ":")

            with open(pathfn) as f:
                lines = f.read().splitlines()

            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
            text.append(ti+'\n'+ab)
        except Exception:
            not_loaded.append(pathfn)
            i += 1

    print(f"not loaded {i} papers.")
    print(f"not loaded papers: {not_loaded}")
    return data, text


def get_raw_text_ogbn_arxiv_nc(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(root='./generated_dataset',
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True

    train_mask = train_mask
    val_mask = val_mask
    test_mask = test_mask

    if data.adj_t.is_symmetric():
        is_symmetric = True
    else:
        edge_index = data.adj_t.to_symmetric()

    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        'generated_dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    tsv_path = FILE_PATH + 'core/dataset/ogbn_arixv_orig/titleabs.tsv'
    raw_text = pd.read_csv(tsv_path,
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    raw_text['paper id'] = pd.to_numeric(raw_text['paper id'], errors='coerce')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = [
        'Title: ' + ti + '\n' + 'Abstract: ' + ab
        for ti, ab in zip(df['title'], df['abs'])
    ]
    # recreate InMemoryDataset
    num_nodes = data.num_nodes
    x = data.x
    y = data.y

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

    return dataset, text



def get_raw_text_products_nc(use_text=False, seed=0):
    data = torch.load(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.pt')
    text = pd.read_csv(FILE_PATH + 'core/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    return (data, text) if use_text else (data, None)


@time_logger
def _process():
    """Process raw text data and convert it into a DataFrame for ogbn-products dataset.
        Download dataset from website http://manikvarma.org/downloads/XC/XMLRepository.html, 
        we utilize https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing
    Args:
        None

    Returns:
        None
    """
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")

    data = []
    files = [FILE_PATH + 'core/dataset/ogbn_products/Amazon-3M.raw/trn.json',
             FILE_PATH + 'core/dataset/ogbn_products/Amazon-3M.raw/tst.json']

    for f in files:
        # Read each line from the input file and parse JSON
        with open(f, "r") as input_file:
            for line in input_file:
                json_object = json.loads(line)
                data.append(json_object)
        

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    dataset = PygNodePropPredDataset(root='./generated_dataset',
        name='ogbn-products', transform=T.ToSparseTensor())
    
    nodeidx2asin = pd.read_csv(
        'generated_dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir(FILE_PATH + 'core/dataset/ogbn_products_orig'):
        os.mkdir(FILE_PATH + 'core/dataset/ogbn_products_orig')

    pd.DataFrame.to_csv(graph_df, FILE_PATH + FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])



def get_pubmed_nc(corrected=False, SEED=0):
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
    with open(FILE_PATH + 'core/dataset/PubMed_orig/data/Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
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

    with open(FILE_PATH+ 'core/dataset/PubMed_orig/data/Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
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
    data, data_pubid = get_pubmed_nc(SEED=seed)
    if not use_text:
        return data, None

    f = open(FILE_PATH + 'core/dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = ['Title: ' + ti + '\n'+'Abstract: ' + ab for ti, ab in zip(TI, AB)]
    return data, text


# TEST CODE
if __name__ == '__main__':
    data, text = get_raw_text_arxiv_2023_nc(use_text=True)
    print(data)
    print(text[0])
    
    data, text = get_raw_text_cora(use_text=True)
    print(data)
    print(text[:3])
    data, citeid = get_cora_nc()
    print(data)
    print(text[:3])
    data_X, data_Y, data_citeid, edge_index = parse_cora()
    print(data)
    print(text[:3])

    data, text = get_raw_text_ogbn_arxiv_nc(use_text=True)
    print(data)
    print(len(text))
    
    data, text = get_raw_text_products_nc(True)
    print(data)
    print(text[0])
    _process()
    