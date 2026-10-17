# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:26:36 2024

@author: AA
"""

from __future__ import division
import math
import torch
from torch import nn
import torch.nn.functional as F


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    each of a same batch has the same Adj
    """

    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                     out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A)

        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
        # print(x.shape)
        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x


'''
Buitld the GNN
'''


class Model(nn.Module):
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
        super(Model, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimnesion = args.d
        self.order = args.k_DGCN
        num_nodes_tra = args.num_nodes_tra

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimnesion, self.order)
        self.GNN2 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN3 = D_GCN(self.hidden_dimnesion, self.time_dimension, self.order, activation='relu')

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes_tra, 64), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(64, num_nodes_tra), requires_grad=True)

    def forward(self, x):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """

        A = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        X_S = x.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, A)
        X_s2 = self.GNN2(X_s1, A) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A)

        X_res = X_s3.permute(0, 2, 1)

        return X_res, A

    def evaluate(self, x, A):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X_S = x.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, A)
        X_s2 = self.GNN2(X_s1, A) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A)

        X_res = X_s3.permute(0, 2, 1)

        return X_res


