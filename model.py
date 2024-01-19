import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layer import SimilarAttentionConv
from layer import NodeAggregationConv

class MHGAN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, norm, args=None):
        super(MHGAN, self).__init__()
        self.num_edge = num_edge  # 5
        self.num_channels = num_channels  # 1
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.norm = norm
        self.args = args

        self.RelationAware = F_RelationAware(in_channels=num_edge, out_channels=num_channels)
        self.SimilarAttention = SimilarAttentionConv(in_channels=self.w_in, hidden_channels=self.w_out)
        self.NodeAggregation = NodeAggregationConv(in_channels=self.w_in, hidden_channels=self.w_out, num_layers=2)
        self.loss = nn.CrossEntropyLoss()
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        self.linear1 = nn.Linear(self.w_out*2, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def forward(self, A, H_adj, X, target_x, target):
        v = X
        Z_v = self.NodeAggregation(v, H_adj)  # GIN

        A = A.unsqueeze(0).permute(0, 3, 1, 2)  # A.unsqueeze(0)=[1,N,N,edgeType]=>[1,edgeType,N,N];
        Ws = []
        A_R, W = self.RelationAware(A)
        Ws.append(W)
        A_R = torch.squeeze(A_R)
        # RelationAttention
        Z_R = self.SimilarAttention(X, A_R)
        #
        Z = torch.cat((Z_R, Z_v), dim=1)

        if self.norm == 'True':
            Z = Z / (torch.max(torch.norm(Z, dim=1, keepdim=True), self.epsilon))  # Z--L2-norm
        else:
            Z=Z

        Z = self.linear1(Z)
        Z = F.relu(Z)
        y = self.linear2(Z[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class F_RelationAware(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(F_RelationAware, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = RelationAware(in_channels, out_channels)
    def forward(self, A):
        A = self.conv(A)
        W = [(F.softmax(self.conv.weight, dim=1)).detach()]
        return A, W
class RelationAware(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RelationAware, self).__init__()
        self.in_channels = in_channels  # >>>5
        self.out_channels = 1  # >>>1
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1), requires_grad=True)
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.2)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, A):
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A
