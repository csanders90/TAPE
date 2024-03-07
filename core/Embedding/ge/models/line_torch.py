import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random

import numpy as np

from ..alias import create_alias_table, alias_sample
from ..utils import preprocess_nxgraph

class LineLoss(nn.Module):
    def forward(self, y_true, y_pred):
        return -torch.mean(torch.log(torch.sigmoid(y_true * y_pred)))

class LINEModel(nn.Module):
    def __init__(self, num_nodes, embedding_size, order='second'):
        super(LINEModel, self).__init__()

        self.first_emb = nn.Embedding(num_nodes, embedding_size)
        self.second_emb = nn.Embedding(num_nodes, embedding_size)
        self.context_emb = nn.Embedding(num_nodes, embedding_size)

        self.order = order

    def forward(self, v_i, v_j):
        v_i_emb = self.first_emb(v_i)
        v_j_emb = self.first_emb(v_j)

        v_i_emb_second = self.second_emb(v_i)
        v_j_context_emb = self.context_emb(v_j)

        first_order = torch.sum(v_i_emb * v_j_emb, dim=-1)
        second_order = torch.sum(v_i_emb_second * v_j_context_emb, dim=-1)

        if self.order == 'first':
            return first_order
        elif self.order == 'second':
            return second_order
        else:
            return [first_order, second_order]


class LINE_torch:
    def __init__(self, graph, embedding_size=8, negative_ratio=5, order='second',):
        """

        :param graph:
        :param embedding_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')

        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = True

        self.rep_size = embedding_size
        self.order = order

        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order

        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size*(1+negative_ratio)

        self._gen_sampling_table()
        self.model, self.embedding_dict = LINEModel(
            self.node_size, self.rep_size, self.order)

    def reset_training_args(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1)*times


    def _gen_sampling_table(self):

        # create sampling table for vertex
        power = 0.75
        numNodes = self.node_size
        node_degree = np.zeros(numNodes)  # out degree
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]
                        ] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([math.pow(node_degree[i], power)
                         for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[j], power)) /
                     total_sum for j in range(numNodes)]

        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        numEdges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0)
                         for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) *
                     numEdges / total_sum for edge in self.graph.edges()]

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)
        
    def batch_iter(self, node2idx):

        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]

        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:

                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h))*-1
                t = []
                for i in range(len(h)):

                    t.append(alias_sample(
                        self.node_accept, self.node_alias))

            if self.order == 'all':
                yield ([np.array(h), np.array(t)], [sign, sign])
            else:
                yield ([np.array(h), np.array(t)], [sign])
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)
                

    def get_embeddings(self,):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                   0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
            

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_args(batch_size, times)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(initial_epoch, initial_epoch + epochs):
            running_loss = 0.0
            for i, data in enumerate(self.batch_it, 0):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if verbose and (epoch + 1) % verbose == 0:
                print(f'Epoch {epoch + 1}, Loss: {running_loss / self.steps_per_epoch}')

        print('Finished Training')


# # Example usage:
# num_nodes = 100  # Replace with your actual number of nodes
# embedding_size = 64  # Replace with your desired embedding size
# order = 'second'  # Replace with 'first' or 'both' if needed

# model = LINEModel(num_nodes, embedding_size, order=order)
# criterion = LineLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# # Convert your TensorFlow inputs to PyTorch tensors
# v_i_tensor = torch.LongTensor([1, 2, 3])  # Replace with your actual data
# v_j_tensor = torch.LongTensor([4, 5, 6])  # Replace with your actual data

# # Convert tensors to PyTorch Variables if you're not using PyTorch 1.6+
# v_i_var = Variable(v_i_tensor)
# v_j_var = Variable(v_j_tensor)

# # Forward pass
# output = model(v_i_var, v_j_var)

# # Compute loss
# loss = criterion(output, torch.ones_like(output))

# # Backward pass and optimization
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
