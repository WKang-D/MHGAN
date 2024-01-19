import numpy as np
import torch.nn as nn
import torch
import scipy.sparse as sp
from torch_geometric.nn import AGNNConv
import torch.nn.functional as F
from torch_geometric.nn import GIN


class NodeAggregationConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(NodeAggregationConv, self).__init__()
        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=0.5, jk='cat')
     # ACM：0.5  DBLP:0.5  IMDB 0.5
    def forward(self, x, adj):
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        edge_index = torch.cuda.LongTensor(indices)
        x = self.gnn(x, edge_index)
        x = F.relu(x)
        return x

class SimilarAttentionConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, adj):
        H = adj
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        edge_index = torch.cuda.LongTensor(indices)
        x = F.dropout(x, p=0.5, training=self.training)  # ACM：0.5  DBLP:0.2  IMDB 0.5
        x = F.relu(self.linear1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.relu(x)
        return torch.mm(H, x)


