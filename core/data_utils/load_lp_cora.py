import torch 
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os
from torch_geometric.data import Data, InMemoryDataset
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import scipy.sparse as ssp
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR
import matplotlib.pyplot as plt
from lpda.adjacency import plot_adjacency_matrix, compare_adj, draw_adjacency_matrix, plot_coo_matrix
from lpda.adjacency import construct_sparse_adj

FILE_PATH = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/TAPE_chen/dataset'


def load_lp_data_cora(data_name: str, 
                 use_text: bool, 
                 use_gpt: bool, 
                 seed: int):
    """load text attribute graph in link predicton setting

    Args:
        dataset (_type_): _description_
        use_text (_type_): _description_
        use_gpt (_type_): _description_
        seed (_type_): _description_

    Returns:
        _type_: _description_
    """
    # type correction
    
    
    # load text and graph
    
    
    # generate link split based on pytorch random split
    
    
    # plot visualize link split
    
    
    # transform to output format
    
    
    
    data, labels = None, None
    return data, labels


def parse_cora():
    # load original data from cora orig without text features
    path = FILE_PATH + '/cora_orig/cora'
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



def get_cora_casestudy(SEED=0) -> InMemoryDataset:
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
    undirected = True
    include_negatives = True
    val_pct = 0.15
    test_pct = 0.05
    
    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}

    return dataset, data_citeid, splits


# main function 
if __name__ == "__main__":
    name = 'cora'
    use_heuristic = 'CN'
    dataset, data_cited, splits = get_cora_casestudy()
    test_split = splits['test']
    labels = test_split.edge_label
    test_index = test_split.edge_label_index
    
    edge_index = splits['train'].edge_index
    edge_weight = torch.ones(edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    # for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
    #     pos_test_pred, edge_index = eval(use_lsf)(A, test_index)
        
    #     plt.figure()
    #     plt.plot(pos_test_pred)
    #     plt.plot(labels)
    #     plt.savefig(f'{use_lsf}.png')
        
    #     acc = torch.sum(pos_test_pred == labels)/pos_test_pred.shape[0]
    #     print(f" {use_lsf}: accuracy: {acc}")
        
    m = construct_sparse_adj(edge_index)
    plot_coo_matrix(m, f'test_edge_index.png')
            
    # 'shortest_path', 'katz_apro', 'katz_close', 'Ben_PPR'
    for use_gsf in ['Ben_PPR', 'SymPPR']:
        scores, edge_reindex, labels = eval(use_gsf)(A, test_index, labels)
        
        # print(scores)
        # print(f" {use_heuristic}: accuracy: {scores}")
        pred = torch.zeros(scores.shape)
        cutoff = 0.05
        thres = scores.max()*cutoff 
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")
    
    
    for use_gsf in ['shortest_path', 'katz_apro', 'katz_close']:
        scores = eval(use_gsf)(A, test_index)
        
        # print(scores)
        # print(f" {use_heuristic}: accuracy: {scores}")
        pred = torch.zeros(scores.shape)
        thres = scores.min()*10
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        print(f" {use_gsf}: acc: {acc}")
        
        # plt.figure()
        # plt.plot(scores, '*')
        # plt.plot(labels, '-o')
        # plt.savefig(f'{use_gsf}_{use_gsf}.png')
        
        
