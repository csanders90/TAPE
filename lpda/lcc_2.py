from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.data import Data, InMemoryDataset
from ogb.linkproppred import PygLinkPropPredDataset
import numpy as np
import torch
from typing import List
from adjacency import construct_sparse_adj
from adjacency import load_data
import matspy as spy


def get_component(adjacencyList: List[List[int]], start: int = 0) -> set:
  num_nodes = len(adjacencyList)
  visited = [False] * num_nodes
  queued_nodes = [start]
  visited_nodes = set()

  while len(queued_nodes) != 0:
    current_node = queued_nodes.pop()
    if visited[current_node]:
      continue
            
    visited[current_node] = True
    visited_nodes.add(current_node)
    neighbors = adjacencyList[current_node] # this call costed O(n) time before. Now it only needs O(#neighbors)
    queued_nodes.extend(neighbors)
  return visited_nodes

# NOTE: this assumes that the graph is undirected/ the adjacency matrix is symmetric
def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  row, col = dataset._data.edge_index.numpy()
  assert(len(row) == len(col))

  num_nodes = dataset._data.num_nodes # NOTE: there is maybe a better way to get the number of nodes?
  adjacencyList = [[] for x in range(num_nodes)] # one list for every node containing the neighbors
  for i in range(len(row)):
    adjacencyList[row[i]].append(col[i]) # fill the lists with the neighbors

  remaining_nodes = set(range(dataset._data.x.shape[0]))
  comps = []
  total_size = 0
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(adjacencyList, start)
    total_size += len(comp)

    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  
  print(len(comp))
  result = np.array(list(comps[np.argmax(list(map(len, comps)))]))
  return result

def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 14
  return mapper

def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]

def use_lcc(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset._data.x[lcc]
    y_new = dataset._data.y[lcc]

    row, col = dataset._data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        num_nodes=y_new.size()[0],
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        node_attrs=x_new, 
        edge_attrs = None, 
        graph_attrs = None
    )
    # original dataset._data = data
    dataset._data = data
    return dataset

if __name__ == '__main__':
  path = '.'
  for name in ['cora', 'citeseer', 'pubmed']:
    # dataset_name = 'cora' 
    # dataset_name = 'citeseer'

    #print("Working on Karateclub")
    #dataset = KarateClub()
    #print(dataset._data.x.shape[0])
    #
    #lcc_collab = use_lcc(dataset)
    #print(lcc_collab.x.shape[0])

    print(f"Working on {name}")
    dataset = Planetoid(path, name) 
    print(dataset.data.x.shape[0])

    lcc_collab = use_lcc(dataset)
    print(lcc_collab.x.shape[0])
    print(f"Finished {name}")


    print("Working on ogbl collab")
    dataset_name = 'ogbl-citation2'
    dataset_name = 'ogbl-collab'
    dataset = PygLinkPropPredDataset(name=dataset_name, root=path)
    print(dataset.data.x.shape[0])
    
    lcc_collab = use_lcc(dataset)
    print(lcc_collab.data.x.shape[0])
    print("Finished ogbl collab")


    # params
    path = '.'
    # 'cora', 'pubmed', 
    for name in ['ogbn-arxiv', 'ogbn-products', 'arxiv_2023']:
        
        if name in ['cora', 'pubmed', 'citeseer', 'ogbn-arxiv', 'ogbn-products', 'arxiv_2023']:
            use_lcc_flag = True
            
            # planetoid_data = Planetoid(path, name) 
            data, num_class, text = load_data(name, use_dgl=False, use_text=False, use_gpt=False, seed=0)
            
            print(data.num_nodes)
            
            if name  == 'ogbn-arxiv':
                edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
            else:
                edge_index = data.edge_index.numpy()
                
            m = construct_sparse_adj(edge_index)
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}/{name}_ori_data_index_spy.png", bbox_inches='tight')
            
            
            if use_lcc_flag:
                data_lcc = use_lcc(data)
            print(data_lcc.num_nodes)


            m = construct_sparse_adj(data_lcc.edge_index.numpy())
            fig, ax = spy.spy_to_mpl(m)
            fig.savefig(f"plots/{name}/{name}_lcc_data_index_spy.png", bbox_inches='tight')
        
