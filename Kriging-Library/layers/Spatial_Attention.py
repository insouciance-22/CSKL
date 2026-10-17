import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_time_steps):
        super(SpatioTemporalSelfAttention, self).__init__()

        # 定义查询（Q），键（K），值（V）的线性变换
        self.query_proj = nn.Linear(2, hidden_dim)  # 2 是 S 的维度，hidden_dim 是隐藏维度
        self.key_proj = nn.Linear(2, hidden_dim)  # 2 是 S 的维度
        self.value_proj = nn.Linear(1, hidden_dim)  # 1 是 X 的特征维度，hidden_dim 是值的隐藏维度

        # 最终输出的线性变换
        self.output_proj = nn.Linear(hidden_dim, 1)  # 输出维度与 X 相同

        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps

    def forward(self, X, S):
        """
        前向传播
        :param X: 输入数据 (B, N, T, 1)
        :param S: 来自 GNN 的空间节点表示 (B, N, 2)
        :return: 输出特征 (B, N, T, 1)
        """
        B, N, T, F = X.shape  # X 的形状是 (B, N, T, 1)
        _, _, hidden_dim = S.shape  # S 的形状是 (B, N, 2)

        # 计算查询（Q），键（K），值（V）
        Q = self.query_proj(S)  # (B, N, hidden_dim)
        K = self.key_proj(S)  # (B, N, hidden_dim)
        V = self.value_proj(X)  # (B, N, T, hidden_dim)

        # 计算查询和键的点积，得到注意力分数
        attention_scores = torch.einsum('bnd,bnd->bnn', Q, K) / (Q.shape[-1] ** 0.5)  # (B, N, N)

        # 对注意力分数应用 softmax 进行归一化，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, N)

        # 对值（V）进行加权求和，得到输出
        attention_output = torch.einsum('bnn,bntd->bntd', attention_weights, V)  # (B, N, T, hidden_dim)

        # 最后通过线性变换得到最终的输出特征
        output = self.output_proj(attention_output)  # (B, N, T, 1)

        return output, attention_weights

