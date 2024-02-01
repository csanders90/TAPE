import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import networkx as nx
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def sort_edge_index(edge_index):
    """Sort the edge index in ascending order according to the source node index."""

    src_index, sort_indices = torch.sort(edge_index[:, 0])
    dst_index = edge_index[sort_indices, 1]
    edge_reindex = torch.stack([src_index, dst_index])
    return edge_reindex, sort_indices


def Ben_PPR(A, edge_index):
    """
    The Personalized PageRank heuristic score.
    Need to install fast_pagerank by "pip install fast-pagerank"
    Too slow for large datasets now.
    :param A: A CSR matrix using the 'message passing' edges
    :param edge_index: The supervision edges to be scored
    :return:
    """
    edge_index = edge_index.t()
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    edge_reindex, sort_indices = sort_edge_index(edge_index)
    
    
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_reindex.shape[1])):
        if i < j:
            continue
        
        src = edge_reindex[0, i]
        personalize = np.zeros(num_nodes)
        # get the ppr for the current source node
        personalize[src] = 1        
        
        # ppr initially start from srt code
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        
        j = i
        # avoid repeated calculation start at the same source 
        while edge_reindex[0, j] == src:
            j += 1
            if j == edge_reindex.shape[1]:
                break
            
        # all_dst 
        all_dst = edge_reindex[1, i:j]

        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    print(f'evaluated PPR for {len(scores)} edges')
    
    return torch.FloatTensor(scores), edge_reindex

def test_index_symmetric(edge_index):
    """symmetrize test edge split"""
    srt, dst = edge_index
    transpose = torch.stack((dst, srt))
    return torch.cat((edge_index, transpose), 1)



def SymPPR(A, edge_index):
    """
    The Personalized PageRank heuristic score.
    Need to install fast_pagerank by "pip install fast-pagerank"
    Too slow for large datasets now.
    :param A: A CSR matrix using the 'message passing' edges
    :param edge_index: The supervision edges to be scored
    :return:
    """
    
    edge_index = test_index_symmetric(edge_index)
    
    edge_index = edge_index.t()
    edge_index, sort_indices = sort_edge_index(edge_index)
    
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]

    visited = {}
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        # get the ppr for the current source node
        personalize[src] = 1        
        
        # ppr initially start from srt code
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
            
        # all_dst 
        j = i
        # avoid repeated calculation start at the same source 
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
            
        # all_dst 
        all_dst = edge_index[1, i:j]
        
        for dst in all_dst:
            visited.update({f"[{src}, {dst}]": ppr[dst]})
        
    scores = []
    length = int(edge_index.shape[1]/2)
    for idx in range(length):
        srt_dst = visited[f'[{edge_index[0, idx]}, {edge_index[1, idx]}]']
        dst_srt = visited[f'[{edge_index[1, idx]}, {edge_index[0, idx]}]']
        scores.append(dst_srt + srt_dst)

    return torch.FloatTensor(scores), edge_index


def shortest_path(A, edge_index, remove=False):

    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in tqdm(range(edge_index.size(1))):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        # if (s,t) in train_pos_list: train_pos_list.remove((s,t))
        # if (t,s) in train_pos_list: train_pos_list.remove((t,s))
        # G = nx.Graph(train_pos_list)
        if remove:
            if (s,t) in G.edges(): 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges(): 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):

            sp = nx.shortest_path_length(G, source=s, target=t)
            # if sp == 0:
            #     print(1)
        else:
            sp = 999
        

        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0

        scores.append(1/(sp))
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)
        
    return torch.FloatTensor(scores)

def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False):

    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    
    for i in range(len(betas)):
        
        betas[i] = np.power(beta, i+1)
    
    for i in tqdm(range(edge_index.size(1))):
        
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges(): 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges(): 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1


        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)


def katz_close(A, edge_index, beta=0.005):

    scores = []
    G = nx.from_scipy_sparse_matrix(A)

    adj = nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1+aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal()-1)

    for i in tqdm(range(edge_index.size(1))):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        scores.append(sim[s,t])
    
    return torch.FloatTensor(scores)


