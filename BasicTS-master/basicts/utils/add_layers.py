import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


# 单一核
class Kernel_Layer_single(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_single, self).__init__()
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use single learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.theta = 0.
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.theta = 0.1
        self.no_self_loop = no_self_loop

    def forward(self, dist, sigma):
        a1 = nn.functional.relu(self.alpha)

        adj = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))

        # scaled_distance_RQ = (dist ** 2) / (2 * a1 * sigma ** 2 + 1e-9)
        # adj = (1 + scaled_distance_RQ).pow(-1)

        # scaled_distance_MT = dist / (sigma * a1 + 1e-9)
        # adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
        #     -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)

        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 组合核2+2
class Kernel_Layer_zuhe(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_zuhe, self).__init__()
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use double-learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.theta = 0.
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.theta = 0.1
        self.no_self_loop = no_self_loop

        self.logits = nn.Parameter(torch.zeros(num_nodes, num_nodes, 2).float())

    def forward(self, dist, sigma):
        a1 = nn.functional.relu(self.alpha)
        a2 = nn.functional.relu(self.alpha2)

        # adj_1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))

        scaled_distance_RQ = (dist ** 2) / (2 * a1 * sigma ** 2 + 1e-9)
        adj_1 = (1 + scaled_distance_RQ).pow(-1)

        scaled_distance_MT = dist / (sigma * a2 + 1e-9)
        adj_2 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)

        pi = F.softmax(self.logits, dim=-1)
        adj = pi[..., 0] * adj_1 + pi[..., 1] * adj_2

        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 通用融合
class Kernel_Layer_ty(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_ty, self).__init__()
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multi-learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            # self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            # self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            # self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            self.theta = 0.
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
            self.theta = 0.1
        # self.w_1 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        # self.w_2 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        # self.w_3 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        self.no_self_loop = no_self_loop
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_nodes, 3).float())

    def forward(self, dist, sigma):
        a1 = nn.functional.relu(self.alpha)
        a2 = nn.functional.relu(self.alpha2)
        a3 = nn.functional.relu(self.alpha3)
        # a1 = F.softplus(self.alpha) + 1e-6
        # a2 = F.softplus(self.alpha2) + 1e-6
        # a3 = F.softplus(self.alpha3) + 1e-6
        adj_1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj_2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj_3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        # adj = self.w_1 * torch.matmul(adj_1, adj_3)
        pi = F.softmax(self.logits, dim=-1)
        adj = pi[..., 0] * adj_1 + pi[..., 1] * adj_2 + pi[..., 2] * adj_3
        # adj = self.w_1 * torch.matmul(adj_1, adj_2) + \
        #       self.w_2 * torch.matmul(adj_2, adj_3) + \
        #       self.w_3 * torch.matmul(adj_1, adj_3)
        # adj = adj_1 * adj_2 * adj_3

        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 可以读取已有alpha
class Kernel_Layer00(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer00, self).__init__()
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.theta = 0.
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            if kernel_alpha == 0:
                alpha = torch.from_numpy(np.load(os.path.join('kernel', 'grin', 'alpha_bay.npy')).astype('float32'))
                self.register_buffer('alpha', alpha)
            else:
                self.alpha = torch.tensor(kernel_alpha)
            self.theta = 0.1
        self.no_self_loop = no_self_loop

    def forward(self, dist, sigma):
        self.alpha_ = nn.functional.relu(self.alpha)
        adj = torch.exp(-torch.square(dist / (self.alpha_ * sigma + 1e-9)))
        # adj = torch.exp(-torch.square(dist / (self.alpha * sigma + 1e-6)))
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# adakernel
class Kernel_Layer(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer, self).__init__()
        self.use_alpha = use_alpha
        if use_alpha:
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            self.alpha = kernel_alpha
        self.no_self_loop = no_self_loop
        if use_alpha == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")

    def forward(self, dist, sigma, theta=0.1):
        scal = F.relu(self.alpha * sigma) + 1e-8
        adj = torch.exp(-torch.square(dist / scal))
        # adj = torch.exp(-torch.square(dist / (self.alpha * sigma + 1e-6)))
        msk = adj > theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


class Kernel_Layer1(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer1, self).__init__()
        if use_alpha:
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            self.alpha = kernel_alpha
        self.no_self_loop = no_self_loop
        self.a = 1.
        if use_alpha == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        print(">>>>>>>>>>>>>>>>>>>>>>>use Rational Quadratic kernel!!!<<<<<<<<<<<<<<<<<<<<<<<")

    def forward(self, dist, sigma, theta=0.1):
        scale = F.relu(2 * self.a * self.alpha * sigma ** 2) + 1e-8
        scaled_distance = (dist ** 2) / scale
        adj = (1 + scaled_distance).pow(-self.a)
        msk = adj > theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


class Kernel_Layer2(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer2, self).__init__()
        if use_alpha:
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            self.alpha = kernel_alpha
        self.no_self_loop = no_self_loop
        if use_alpha == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        print(">>>>>>>>>>>>>>>>>>>>>>>use Matern kernel!!!<<<<<<<<<<<<<<<<<<<<<<<")

    def forward(self, dist, sigma, theta=0.1):
        scaled_distance = dist / (F.relu(sigma * self.alpha) + 1e-8) # 归一化距离
        adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance)
        msk = adj > theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj