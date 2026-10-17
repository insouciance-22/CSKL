from torch import nn
from layers.DGCN_Block import D_GCN
from layers.GCN_conv import GCNLayer_Batch
import torch
import torch.nn.functional as F


class Model1(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m
    """
    def __init__(self, args):
        super(Model1, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimension = args.d
        self.order = args.k_DGCN

        # self.sigma = 2584.4502
        # self.sigma = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.sigma2 = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.ell_eq = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.GR = GraphRegressionModel(self.time_dimension, 64, 1)
        # self.SP = SigmaPredictor(self.time_dimension, 32)
        # self.SP.apply(weights_init)
        # self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=True)
        if args.use_alpha:
            self.alpha = nn.Parameter(torch.full((1,), args.kernel_alpha))
        else:
            self.alpha = args.kernel_alpha

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='relu')

        # self.PSG = PositiveSampleGenerator(drop_self_con=False)

    def forward(self, x, add_data):
        # load_data
        adj_sc = add_data['D_S']
        adj_matrix = torch.exp(-torch.square(adj_sc / (self.alpha * add_data['sigma'][0])))
        msk = adj_matrix > 0.1
        adj_matrix = adj_matrix * msk

        # IGNNK
        A_q = calculate_random_walk_matrix(adj_matrix).permute(0, 2, 1)
        A_h = calculate_random_walk_matrix(adj_matrix.permute(0, 2, 1)).permute(0, 2, 1)
        X_S = x.permute(0, 2, 1)  # to correct the input dims
        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_res = X_s3.permute(0, 2, 1)

        # adj_normal = self.PSG

        return X_res


class Model_1(nn.Module):
    """
    用GNN代替IGNNK
    """

    def __init__(self, args):
        super(Model_1, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimension = args.d
        self.order = args.k_DGCN

        self.fc1 = nn.Linear(self.time_dimension, self.hidden_dimension)
        self.GNN1 = GCNLayer_Batch(self.time_dimension, self.hidden_dimension)
        self.fc2 = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.GNN2 = GCNLayer_Batch(self.hidden_dimension, self.hidden_dimension)
        self.fc3 = nn.Linear(self.hidden_dimension, self.time_dimension)
        self.GNN3 = GCNLayer_Batch(self.hidden_dimension, self.time_dimension)

        # self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        # self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        # self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='relu')

    def forward(self, x, add_data):
        """
        :param x: Input data of shape (batch_size, num_timesteps, num_nodes)/(B, T, N)
        :A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)/(B, N, N)
        :A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        # adj_kernel
        adj_dist = add_data['D_S']
        adj_matrix = torch.exp(-torch.square(adj_dist / add_data['sigma'][0]))
        adj_matrix[adj_matrix < 0.1] = 0.

        X_S = x.permute(0, 2, 1)
        X_s1 = F.relu(self.fc1(X_S) + self.GNN1(X_S, adj_matrix))
        X_s2 = F.relu(self.fc2(X_s1) + self.GNN2(X_s1, adj_matrix)) + X_s1
        X_s3 = F.relu(self.fc3(X_s2) + self.GNN3(X_s2, adj_matrix))
        X_res = X_s3.permute(0, 2, 1)

        # # random walk
        # A_q = calculate_random_walk_matrix(adj_matrix).permute(0, 2, 1)
        # A_h = calculate_random_walk_matrix(adj_matrix.permute(0, 2, 1)).permute(0, 2, 1)
        # # IGNNK
        # X_S = x.permute(0, 2, 1)  # to correct the input dims
        # X_s1 = self.GNN1(X_S, A_q, A_h)
        # X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        # X_s3 = self.GNN3(X_s2, A_q, A_h)
        # X_res = X_s3.permute(0, 2, 1)

        return X_res


class Model_dy(nn.Module):
    """
    动态计算sigma
    """

    def __init__(self, args):
        super(Model_dy, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimension = args.d
        self.order = args.k_DGCN

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='relu')

    def forward(self, x, add_data):
        """
        :param x: Input data of shape (batch_size, num_timesteps, num_nodes)/(B, T, N)
        :A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)/(B, N, N)
        :A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        sigma = add_data['dy_sigma'].unsqueeze(1).unsqueeze(2)
        adj_dist = add_data['D_S']
        adj_matrix = torch.exp(-torch.square(adj_dist / sigma))
        adj_matrix[adj_matrix < 0.1] = 0.

        # random walk
        A_q = calculate_random_walk_matrix(adj_matrix).permute(0, 2, 1)
        A_h = calculate_random_walk_matrix(adj_matrix.permute(0, 2, 1)).permute(0, 2, 1)
        # self.A_q = A_q

        # IGNNK
        X_S = x.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)

        X_res = X_s3.permute(0, 2, 1)

        return X_res


def calculate_random_walk_matrix(adj_mx):
    # (B, N, N)
    degree = adj_mx.sum(dim=-1, keepdims=True)  # (B, N, 1)
    degree_inv = degree.pow(-1)  # (B, N, 1)
    degree_inv[torch.isinf(degree_inv)] = 0.  # Set inf values to 0
    random_walk_mx = degree_inv * adj_mx  # (B, N, N)

    return random_walk_mx


class GraphRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphRegressionModel, self).__init__()
        # self.GNN1 = GCNLayer_Batch(input_dim, hidden_dim)
        # self.GNN2 = GCNLayer_Batch(hidden_dim, hidden_dim)
        # 一个简单的全连接层，用于回归
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj=0):
        if adj:
            x = x.permute(0, 2, 1)
            x = F.relu(self.GNN1(x, adj))
            x = F.relu(self.GNN2(x, adj))
            x = torch.mean(x, dim=1)
        else:
            x = torch.mean(x, dim=-1)  # (B, T)
            x = F.relu(self.fc1(x))
        output = self.fc2(x) # 最终回归输出

        return output


class SigmaPredictor(nn.Module):
    """
    神经网络模型，用于预测每个图（或每个批次图）的 sigma。
    输入：节点特征、邻接矩阵，输出：对应的 sigma 值。
    """
    def __init__(self, in_features, hidden_features, out_features=1):
        super(SigmaPredictor, self).__init__()
        self.fc1 = nn.Linear(in_features+1, hidden_features)
        # self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, X, A):
        """
        X: 节点特征 (B, T, N)
        A: 邻接矩阵 (B, N, N)
        """
        # 可以根据节点特征和邻接矩阵来设计特征提取
        # 比如聚合节点特征或计算图的某些统计信息
        X = self.fc1(X)
        node_feature_sum = X.sum(dim=2)  # 聚合节点特征作为图的表示
        adj_sum = A.sum(dim=[1, 2])  # 聚合邻接矩阵，表示图的稠密度

        # 拼接节点特征和邻接矩阵的信息
        combined_features = torch.cat([node_feature_sum, adj_sum.unsqueeze(-1)], dim=-1)

        # 通过神经网络预测 sigma
        x = F.relu(self.fc1(combined_features))
        # x = F.relu(self.fc2(x))
        sigma = torch.exp(self.fc3(x))  # 使用exp函数保证sigma为正数

        return sigma


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class adjPredictor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1):
        super(adjPredictor, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.GNN1 = GCNLayer_Batch(in_features, hidden_features)

    def forward(self, X, A):
        """
        X: 节点特征 (B, T, N)
        A: 邻接矩阵 (B, N, N)
        """
        # 可以根据节点特征和邻接矩阵来设计特征提取
        # 比如聚合节点特征或计算图的某些统计信息
        X = self.fc1(X) + self.GNN1(X)
        node_feature_sum = X.sum(dim=2)  # 聚合节点特征作为图的表示
        adj_sum = A.sum(dim=[1, 2])  # 聚合邻接矩阵，表示图的稠密度

        # 拼接节点特征和邻接矩阵的信息
        combined_features = torch.cat([node_feature_sum, adj_sum.unsqueeze(-1)], dim=-1)

        # 通过神经网络预测 sigma
        x = F.relu(self.fc1(combined_features))
        # x = F.relu(self.fc2(x))
        sigma = torch.exp(self.fc3(x))  # 使用exp函数保证sigma为正数

        return sigma



from layers.Kernel_Block import Kernel_Layer

class Model(nn.Module):
    """
    根据attn调整自学习矩阵
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimension = args.d
        self.order = args.k_DGCN
        # if args.use_alpha:
        #     self.alpha = nn.Parameter(torch.full((1,), args.kernel_alpha))
        # else:
        #     self.alpha = args.kernel_alpha
        self.kernel_layer = Kernel_Layer(args.use_alpha, args.kernel_alpha)

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='relu')
        # self.PG = PositiveGenerator(drop_self_con=False)

    def forward(self, x, add_data):
        # load_data
        dist = add_data['D_S']
        sigma = add_data['sigma'][0]
        adj_matrix = self.kernel_layer(dist, sigma)

        # IGNNK
        A_q = calculate_random_walk_matrix(adj_matrix).permute(0, 2, 1)
        A_h = calculate_random_walk_matrix(adj_matrix.permute(0, 2, 1)).permute(0, 2, 1)
        X_S = x.permute(0, 2, 1)  # to correct the input dims
        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_res = X_s3.permute(0, 2, 1)

        return X_res


from torch import nn
import torch

from layers.DGCN_Block import D_GCN
from layers.GraphAttention_Block import Node_Attention


class Model0(nn.Module):
    """
    IGNNK attention baseline for kriging.

    This keeps the DCRNN-like / IGNNK message-passing backbone unchanged and
    only replaces the CSKL kernel adjacency with a learned attention adjacency.
    """

    def __init__(self, args):
        print("attention")
        super(Model0, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimension = args.d
        self.order = args.k_DGCN
        self.use_self_loop = getattr(args, "attn_self_loop", True)

        attn_hidden = getattr(args, "attn_hidden", self.hidden_dimension)
        self.attn_layer = Node_Attention(self.time_dimension, attn_hidden)
        self.kernel_layer = Kernel_Layer(args.use_alpha, args.kernel_alpha)

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='relu')

    def _build_attention_adj(self, x, add_data):
        # x: [B, T, N] -> [B, N, T]
        node_feat = x.permute(0, 2, 1)
        adj_matrix = self.attn_layer(node_feat)  # [B, N, N]


        if not self.use_self_loop:
            eye = torch.eye(adj_matrix.size(-1), device=adj_matrix.device).unsqueeze(0)
            adj_matrix = adj_matrix * (1.0 - eye)
            adj_sum = adj_matrix.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            adj_matrix = adj_matrix / adj_sum

        return adj_matrix

    def forward(self, x, add_data):
        adj_matrix = self._build_attention_adj(x, add_data)

        # Use the same IGNNK random-walk propagation as the kernel baseline.
        A_q = calculate_random_walk_matrix(adj_matrix).permute(0, 2, 1)
        A_h = calculate_random_walk_matrix(adj_matrix.permute(0, 2, 1)).permute(0, 2, 1)

        X_S = x.permute(0, 2, 1)
        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1
        X_s3 = self.GNN3(X_s2, A_q, A_h)
        X_res = X_s3.permute(0, 2, 1)

        return X_res


def calculate_random_walk_matrix(adj_mx):
    # (B, N, N)
    degree = adj_mx.sum(dim=-1, keepdims=True)
    degree_inv = degree.pow(-1)
    degree_inv[torch.isinf(degree_inv)] = 0.
    random_walk_mx = degree_inv * adj_mx
    return random_walk_mx


