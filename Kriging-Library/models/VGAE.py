import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GCN_conv import GCNLayer


class Model(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
        super(Model, self).__init__()
        self.gae = gae
        self.gcn_base = GCNLayer(dim_feats, dim_h, None, 0, bias=False)
        self.gcn_mean = GCNLayer(dim_h, dim_z, activation, 0, bias=False)
        self.gcn_logstd = GCNLayer(dim_h, dim_z, activation, 0, bias=False)

    def forward(self, adj, features):
        # GCN encoder
        hidden = self.gcn_base(adj, features)
        self.mean = self.gcn_mean(adj, hidden)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = self.gcn_logstd(adj, hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise * torch.exp(self.logstd) + self.mean
            Z = F.relu(sampled_Z)
        # inner product decoder+
        B, N, D = Z.shape
        Z.permute(1, 0, 2)
        Z = Z.reshape(N, B * D)
        adj_logits = Z @ Z.T
        return adj_logits


class LearnableAdjacencyMatrix(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LearnableAdjacencyMatrix, self).__init__()

        # 初始化两个可学习的嵌入矩阵 E1 和 E2
        self.E1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))

    def forward(self):
        # 计算 E1 * E2^T
        adjacency_raw = torch.matmul(self.E1, self.E2.T)  # 矩阵乘法

        # 使用 ReLU 截断负值
        adjacency_relu = F.relu(adjacency_raw)

        # 对每一行应用 SoftMax 归一化，得到最终的邻接矩阵
        adjacency_matrix = F.softmax(adjacency_relu, dim=-1)

        return adjacency_matrix