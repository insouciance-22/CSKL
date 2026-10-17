import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, dropout=0, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj):
        if self.dropout:
            h = self.dropout(h)
        # normalization
        adj_hat = adj + torch.eye(adj.size(0)).to(adj.device)
        degree = torch.sum(adj_hat, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        adj_normalized = D_inv_sqrt @ adj_hat @ D_inv_sqrt

        x = h @ self.W
        # x = adj @ x
        x = adj_normalized @ x
        if self.b is not None:
            x = x + self.b
        return x


class GCNLayer_Batch(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, dropout=0, bias=True):
        super(GCNLayer_Batch, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj):
        # h: 输入特征，形状为 (B, N, in_features)
        # adj: 邻接矩阵，形状为 (B, N, N)

        if self.dropout:
            h = self.dropout(h)  # 应用 dropout

        # 对每个图的邻接矩阵进行归一化
        batch_size = adj.size(0)

        # 增加自连接项
        adj_hat = adj + torch.eye(adj.size(1), device=adj.device).unsqueeze(0)  # (B, N, N)

        # 计算度矩阵 D，注意批处理时需要按批次处理
        degree = torch.sum(adj_hat, dim=2)  # (B, N)
        D_inv_sqrt = torch.pow(degree, -0.5)  # (B, N)
        D_inv_sqrt = D_inv_sqrt.unsqueeze(2)  # (B, N, 1)
        D_inv_sqrt = D_inv_sqrt.expand(-1, -1, adj.size(1))  # (B, N, N)

        # 归一化邻接矩阵: D^-0.5 * A * D^-0.5
        adj_normalized = D_inv_sqrt * adj_hat * D_inv_sqrt  # (B, N, N)

        # 输入特征和权重矩阵的线性变换: h * W
        x = torch.matmul(h, self.W)  # (B, N, out_features)

        # 使用归一化邻接矩阵进行消息传递: A_hat * hW
        x = torch.matmul(adj_normalized, x)  # (B, N, out_features)

        # 加上偏置项
        x = x + self.b  # (B, N, out_features)

        return x