"""
utils for getting the largest connected component of a graph
"""
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import torch
from torch_sparse.tensor import SparseTensor


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset._data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    print("Size of component", len(comp))
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def get_row_col(edge_index: SparseTensor) -> list:
  if type(edge_index) is torch.Tensor:
    row, col = edge_index.numpy()
  elif type(edge_index) is SparseTensor:
    row, col = edge_index.to_torch_sparse_coo_tensor().coalesce().indices().numpy()
    return row, col
  
  
def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  # BFS
  # this is a bit slow, but it works
  # TODO: make this faster
  # TODO: make this work for directed graphs
  """this function detect the llc of a undirected graph with symmetric adjacency matrix

  Args:
      dataset (InMemoryDataset): modified graph from ogb or planetoid
      start (int, optional): start node index. Defaults to 0.
      dataset.data.edge_index is a tensor of shape [2, num_edges], default type is numpy.ndarray
      
  Returns:
      set: return a set of the node set of local connected component of the graph
  """
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = get_row_col(dataset._data.edge_index)
    
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def use_lcc(dataset: InMemoryDataset) -> InMemoryDataset:
    lcc = get_largest_connected_component(dataset)

    x_new = dataset._data.x[lcc]
    y_new = dataset._data.y[lcc]

    row, col = get_row_col(dataset._data.edge_index)
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    # TODO add updated masks after lcc to data
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

    dataset._data = data
    
    return dataset