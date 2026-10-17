import math
import torch
from torch import nn
import torch.nn.functional as F
from models.IGNNK import Model as IGNNK
from models.VGAE import Model as VGAE
import pyro


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.temperature = 1
        self.GNN = IGNNK(args)
        self.ep_net = VGAE(args.seq_len, args.dim_h, args.dim_z, F.relu, args.gae)

    def forward(self, x, add_data):
        B, N, N = add_data['A'].shape
        x = x.permute(0, 2, 1)
        adj_list = []
        for i in range(B):
            adj = add_data['A'][i]
            adj_logits = self.ep_net(adj, x)
            adj_enhanced = self.sample_adj(adj_logits)
            adj_list.append(adj_enhanced)
        adj_batch = torch.stack(adj_list, dim=0)
        # adj_enhanced = self.enhance_graph(adj, adj_logits)
        if self.training:
            add_data = {
                'A_q': adj_batch,
                'A_h': adj_batch.transpose(1, 2)
            }
        else:
            add_data = {
                'A_q': add_data['A'],
                'A_h': add_data['A'].transpose(1, 2)
            }
        x = x.permute(0, 2, 1)
        return self.GNN(x, add_data), adj_batch

    def sample_adj(self, adj_logits):
        """ sample an adj from the predicted edge probabilities of ep_net """
        # edge_probs = adj_logits / torch.max(adj_logits)
        # sampling
        # adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
        #                                                                  probs=edge_probs).rsample()

        # making adj_sampled symmetric
        adj_sampled = adj_logits.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        adj_sampled = F.softmax(adj_sampled)
        return adj_sampled

    def enhance_graph(self, adj, adj_logits, alpha=0.5, thr=0.1):
        """
        使用 GAE 的输出 logits 和原始距离邻接矩阵进行图增强。

        Parameters:
            adj (np.ndarray): 原始距离构建的邻接矩阵 (n, n)。
            adj_logits (torch.Tensor): GAE 输出的预测邻接矩阵 logits。
            alpha (float): 混合比例，控制原始邻接矩阵和 GAE 输出的影响。
            thr (float): 阈值，小于该值的边被设为 0。

        Returns:
            adj_enhanced (np.ndarray): 增强后的邻接矩阵。
        """
        # 将 GAE 的输出 logits 转换为概率 [0, 1]
        adj_probs = torch.sigmoid(adj_logits)
        # 混合原始邻接矩阵和 GAE 的输出
        adj_enhanced = alpha * adj + (1 - alpha) * adj_probs

        # 阈值化，去除小权重的边
        adj_enhanced[adj_enhanced < thr] = 0.0

        # 保证对称性
        # adj_enhanced = torch.max(adj_enhanced, adj_enhanced.T)

        return adj_enhanced




