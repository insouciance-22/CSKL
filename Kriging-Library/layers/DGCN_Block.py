import math
import torch
from torch import nn
import torch.nn.functional as F


class D_GCN_1(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN_1, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
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

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_timesteps)
        """
        B = X.shape[0] # batch_size
        N = X.shape[1]
        T = X.size(2)  # time_length

        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0)  # (T, N, B)
        x0 = torch.reshape(x0, shape=[N, T * B])
        x = torch.unsqueeze(x0, 0)  # (1, N, T*B)
        for support in supports:
            x1 = torch.mm(support, x0)  # (N, N) @ (N, T*B)->(N, T*B)
            x = self._concat(x, x1)  # (2, N, T*B)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x0) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, N, T, B])
        x = x.permute(3, 1, 2, 0)  # (B, N, T, order)
        x = torch.reshape(x, shape=[B, N, T * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (T * order, output_size)->(B, N, out)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)
        elif self.activation == 'softplus':
            x = F.softplus(x)

        return x


class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
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
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                     out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)
        :A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_timesteps)
        """
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X  # (num_nodes, num_times, batch_size)
        x = X
        # x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        # x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.bmm(support, x0)
            x = torch.cat([x, x1], -1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.bmm(support, x0) - x0
                x = torch.cat([x, x2], -1)
                x1, x0 = x2, x1

        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)
        elif self.activation == 'softplus':
            x = F.softplus(x)

        return x