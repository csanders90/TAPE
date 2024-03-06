import torch
from torch_geometric.transforms import BaseTransform

class CustomLinkSplit(BaseTransform):
    def __init__(self, split_ratio=0.8, is_undirected=True, **kwargs):
        # TODO [heuristic] (P3) Random Link Split : Add support for edge-level splitting in heterogeneous graphs
        self.split_ratio = split_ratio
        self.is_undirected = is_undirected
        super(CustomLinkSplit, self).__init__(**kwargs)

    def __call__(self, data):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # Calculate the number of edges to include in the training set
        num_train_edges = int(self.split_ratio * num_edges)

        # Randomly shuffle the edges
        perm = torch.randperm(num_edges)
        edge_index = edge_index[:, perm]

        # Split the edges into training and validation sets
        data.train_mask = torch.zeros(num_edges, dtype=torch.bool)
        data.train_mask[:num_train_edges] = 1

        if self.is_undirected:
            # If the graph is undirected, ensure symmetry in the train_mask
            data.train_mask = data.train_mask | data.train_mask[edge_index[1]]

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(split_ratio={self.split_ratio})'


if __name__ is "main":
    from torch_geometric.data import Data
    from torch_geometric.transforms import Compose

    # Create a dummy dataset
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data = Data(edge_index=edge_index)

    # Instantiate and apply the custom link split transform
    custom_link_split = CustomLinkSplit(split_ratio=0.8, is_undirected=True)
    transform = Compose([custom_link_split])
    data = transform(data)

    # Access the train_mask in the transformed data
    print(data.train_mask)
