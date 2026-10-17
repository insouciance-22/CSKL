import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import numpy as np


# 单一核
class Kernel_Layer_si(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_si, self).__init__()
        self.theta = 0.
        self.no_self_loop = no_self_loop
        if use_alpha:
            # self.alpha = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.alpha = nn.Parameter(torch.tensor([kernel_alpha]))
            print(">>>>>>>>>>>>>>>>>>>>>>>use single learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            self.alpha = torch.tensor(kernel_alpha)
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")

    def forward(self, dist, sigma):
        # a1 = F.softplus(self.alpha) + 1e-6
        a1 = nn.functional.relu(self.alpha)

        # adj = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))

        scaled_distance_RQ = (dist ** 2) / (2 * a1 * sigma ** 2 + 1e-9)
        adj = (1 + scaled_distance_RQ).pow(-1)

        # scaled_distance_MT = dist / (sigma * a1 + 1e-9)
        # adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
        #     -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj

# 2+组合
class Kernel_Layer_z(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_z, self).__init__()
        self.theta = 0.
        self.no_self_loop = no_self_loop
        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.alpha2 = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            print(">>>>>>>>>>>>>>>>>>>>>>>use double learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        self._pi_logits = nn.Parameter(torch.zeros(2).float())

    def forward(self, dist, sigma):
        a1 = F.softplus(self.alpha) + 1e-6
        a2 = F.softplus(self.alpha2) + 1e-6

        # adj1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))

        scaled_distance_RQ = (dist ** 2) / (2 * a1 * sigma ** 2 + 1e-9)
        adj1 = (1 + scaled_distance_RQ).pow(-1)

        scaled_distance_MT = dist / (sigma * a2 + 1e-9)
        adj2 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)

        pi = F.softmax(self._pi_logits, dim=0)
        adj = pi[0] * adj1 + pi[1] * adj2
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 2+特化组合
class Kernel_Layer_zt(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_zt, self).__init__()
        self.theta = 0.
        self.no_self_loop = no_self_loop
        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([kernel_alpha]))
            self.alpha2 = nn.Parameter(torch.tensor([kernel_alpha]))
            print(">>>>>>>>>>>>>>>>>>>>>>>use double learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        self.w1 = nn.Parameter(torch.full((1,), 1.))
        self.w2 = nn.Parameter(torch.full((1,), 1.))
        # self._pi_logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        a1 = nn.functional.relu(self.alpha)
        a2 = nn.functional.relu(self.alpha2)

        # adj1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))

        scaled_distance_RQ = (dist ** 2) / (2 * a1 * sigma ** 2 + 1e-9)
        adj1 = (1 + scaled_distance_RQ).pow(-1)

        scaled_distance_MT = dist / (sigma * a2 + 1e-9)
        adj2 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)

        adj = (self.w1 * adj1) * (self.w2 * adj2)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 通用
class Kernel_Layer00(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer00, self).__init__()
        self.theta = 0.
        self.no_self_loop = no_self_loop
        if use_alpha:
            # self.alpha = nn.Parameter(torch.tensor([kernel_alpha]))
            # self.alpha2 = nn.Parameter(torch.tensor([kernel_alpha]))
            # self.alpha3 = nn.Parameter(torch.tensor([kernel_alpha]))
            self.alpha = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.alpha2 = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.alpha3 = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        # self.w1 = nn.Parameter(torch.full((1,), 1.))
        # self.w2 = nn.Parameter(torch.full((1,), 1.))
        # self.w3 = nn.Parameter(torch.full((1,), 1.))
        self._pi_logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        # a1 = nn.functional.relu(self.alpha)
        # a2 = nn.functional.relu(self.alpha2)
        # a3 = nn.functional.relu(self.alpha3)
        a1 = F.softplus(self.alpha) + 1e-6
        a2 = F.softplus(self.alpha2) + 1e-6
        a3 = F.softplus(self.alpha3) + 1e-6
        adj1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        # adj = (self.w1 * adj1) + (self.w2 * adj2) + (self.w3 * adj3)
        # adj = (self.w1 * adj1) * (self.w2 * adj2) * (self.w3 * adj3)
        # adj = self.w1*(adj1 * adj2) + self.w2*(adj2 * adj3) + self.w3*(adj1 * adj3)
        # adj = self.w1 * torch.matmul(adj1, adj2) + self.w2 * torch.matmul(adj2, adj3) + self.w3 * torch.matmul(adj1, adj3)
        pi = F.softmax(self._pi_logits, dim=0)
        adj = pi[0] * adj1 + pi[1] * adj2 + pi[2] * adj3
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 特化
class Kernel_Layer(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer, self).__init__()
        self.theta = 0.
        self.no_self_loop = no_self_loop
        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([kernel_alpha]))
            self.alpha2 = nn.Parameter(torch.tensor([kernel_alpha]))
            self.alpha3 = nn.Parameter(torch.tensor([kernel_alpha]))
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        self.w1 = nn.Parameter(torch.full((1,), 1.))
        self.w2 = nn.Parameter(torch.full((1,), 1.))
        self.w3 = nn.Parameter(torch.full((1,), 1.))
        # self._pi_logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        a1 = nn.functional.relu(self.alpha)
        a2 = nn.functional.relu(self.alpha2)
        a3 = nn.functional.relu(self.alpha3)
        adj1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        adj = (self.w1 * adj1) * (self.w2 * adj2) * (self.w3 * adj3)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# adakernel
class Kernel_Layer_ada(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer_ada, self).__init__()
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha(adakernel)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            if kernel_alpha == 0:
                self.alpha = nn.Parameter(torch.abs(torch.randn(1,) * 0.1 + 1.0))
            elif kernel_alpha == -1:
                self.alpha = nn.Parameter(0.9 * torch.rand(1,) + 0.1)
            else:
                self.alpha = nn.Parameter(torch.full((1,), kernel_alpha))
            self.theta = 0.
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = kernel_alpha
            self.theta = 0.1
        self.no_self_loop = no_self_loop
        print("use Gaussian kernel")

    def forward(self, dist, sigma):
        adj = torch.exp(-torch.square(dist / (self.alpha * sigma)))
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


class Kernel_Layer1(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer1, self).__init__()
        if use_alpha:
            self.alpha = nn.Parameter(torch.full((1,), kernel_alpha))
        else:
            self.alpha = kernel_alpha
        self.no_self_loop = no_self_loop
        self.a = 1.
        print("use RQ kernel")

    def forward(self, dist, sigma, theta=0.1):
        scaled_distance = (dist ** 2) / (2 * self.a * self.alpha * sigma ** 2)
        adj = (1 + scaled_distance).pow(-self.a)
        msk = adj > theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


class Kernel_Layer2(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer2, self).__init__()
        if use_alpha:
            self.alpha = nn.Parameter(torch.full((1,), kernel_alpha))
        else:
            self.alpha = kernel_alpha
        self.no_self_loop = no_self_loop
        print("use matern kernel")

    def forward(self, dist, sigma, theta=0.1):
        scaled_distance = dist / (sigma * self.alpha)  # 归一化距离
        adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance)
        msk = adj > theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# class Kernel_Layer(nn.Module):
#     def __init__(self, use_alpha=True, kernel_alpha=1.):
#         super(Kernel_Layer, self).__init__()
#         self.alpha = kernel_alpha
#         self.period = 1.
#
#     def forward(self, dist, sigma, theta=0.1):
#         self.lengthscale = 1
#         scaled_distance = (dist / self.period)  # 计算标准化距离
#         adj = torch.exp(-2 * (torch.sin(torch.pi * scaled_distance) ** 2) / (self.lengthscale ** 2))
#         msk = adj > theta
#         adj = adj * msk
#         return adj


