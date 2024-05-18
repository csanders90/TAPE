import torch
from torch import nn
import torch_geometric
from torch import sigmoid


class InnerProductDecoder(torch.nn.Module):
    """解码器，用向量内积表示重建的图结构"""
    
    def forward(self, z, edge_index, sigmoid=True):
        """
        参数说明：
        z: 节点表示
        edge_index: 边索引，也就是节点对
        """
        # print(edge_index[0])
        # print(z)
        # raise(0)
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class RecLoss(nn.Module):
    def __init__(self, decoder=InnerProductDecoder(), acf='sigmoid'):
        super(RecLoss, self).__init__()
        self.decoder = decoder
        self.acf = acf
        self.eps = 1e-15

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        """
        params
        ----
        z: output of encoder
        pos_edge_index: positive edge index
        neg_edge_index: negative edge index
        """
        # Positive loss
        pos_loss = -torch.log(eval(self.acf)(self.decoder(z, pos_edge_index)) + self.eps).mean()

        # Negative sampling if neg_edge_index is not provided
        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0))

        # Negative loss
        neg_loss = -torch.log(1 - eval(self.acf)(self.decoder(z, neg_edge_index)) + self.eps).mean()

        return pos_loss + neg_loss

