import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#cskl+sm
class Kernel_Layer_csklsm(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer_csklsm, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha(cskl+sm)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha4 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            # self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            # self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            # self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_nodes, 4).float())
        # self.logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        a1 = F.relu(self.alpha)
        a2 = F.relu(self.alpha2)
        a3 = F.relu(self.alpha3)
        a4 = F.relu(self.alpha4)

        adj_1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj_2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj_3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)

        d = dist / (sigma + 1e-9)
        envelope = torch.exp(-torch.square(d))
        spectral = 0.5 * (1.0 + torch.cos(2.0 * math.pi * a4 * d))
        adj_4 = envelope * spectral

        pi = F.softmax(self.logits, dim=-1)
        # adj = pi[0] * adj_1 + pi[1] * adj_2 + pi[2] * adj_3
        adj = pi[..., 0] * adj_1 + pi[..., 1] * adj_2 + pi[..., 2] * adj_3 + pi[..., 3] * adj_4
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj

class Kernel_Layer_sm(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1.0, no_self_loop=False, theta=0.1):
        super(Kernel_Layer_sm, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        self.no_self_loop = no_self_loop

        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable mu!!! (SM kernel)<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>>>>>init_mu =", kernel_alpha, "<<<<<<<")
            if kernel_alpha == 0:
                self.alpha = nn.Parameter(torch.abs(torch.randn(num_nodes, num_nodes) * 0.1 + 1.0))
            elif kernel_alpha == -1:
                self.alpha = nn.Parameter(0.9 * torch.rand(num_nodes, num_nodes) + 0.1)
            else:
                self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static mu:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)

        print(">>>>>>>>>>theta =", theta, "<<<<<<<<<<<<<<")

    def forward(self, dist, sigma):
        # 保证 mu 非负
        mu_ = F.relu(self.alpha)

        # 归一化距离，和你的高斯核风格保持一致
        d = dist / (sigma + 1e-9)

        # Gaussian envelope
        envelope = torch.exp(-torch.square(d))

        # Spectral oscillation term
        # 映射到 [0, 1]，避免负权邻接
        spectral = 0.5 * (1.0 + torch.cos(2.0 * math.pi * mu_ * d))

        adj = envelope * spectral

        # threshold sparsification
        msk = adj > self.theta
        adj = adj * msk

        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))

        return adj
# rq
class Kernel_Layer_rq(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer_rq, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use single learnable alpha(rq)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha(rq):", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop

    def forward(self, dist, sigma):
        a = F.relu(self.alpha)
        scaled_distance_RQ = (dist ** 2) / (2 * a * sigma ** 2 + 1e-9)
        adj = (1 + scaled_distance_RQ).pow(-1)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj
#mt
class Kernel_Layer_mt(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer_mt, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use single learnable alpha(mt)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha(mt):", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop

    def forward(self, dist, sigma):
        a = F.relu(self.alpha)
        scaled_distance_MT = dist / (sigma * a + 1e-9)
        adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj

# 组合核2+2
class Kernel_Layer_22(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer_22, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha(2个排列组合)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha(2个排列组合):", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop

        self.logits = nn.Parameter(torch.zeros(2).float())

    def forward(self, dist, sigma):
        a = F.relu(self.alpha)
        a2 = F.relu(self.alpha2)

        # adj1 = torch.exp(-torch.square(dist / (a * sigma + 1e-9)))

        scaled_distance_RQ = (dist ** 2) / (2 * a * sigma ** 2 + 1e-9)
        adj1 = (1 + scaled_distance_RQ).pow(-1)

        scaled_distance_MT = dist / (sigma * a2 + 1e-9)
        adj2 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)

        pi = F.softmax(self.logits, dim=-1)
        adj = pi[0] * adj1 + pi[1] * adj2

        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj

# 特化
class Kernel_Layer_th(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer_th, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha(cskl_th)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
        self.w_1 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        self.w_2 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        self.w_3 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        self.no_self_loop = no_self_loop

    def forward(self, dist, sigma):
        a1 = F.relu(self.alpha)
        a2 = F.relu(self.alpha2)
        a3 = F.relu(self.alpha3)
        adj_1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj_2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj_3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        adj = self.w_1 * torch.matmul(adj_1, adj_2) + \
                      self.w_2 * torch.matmul(adj_2, adj_3) + \
                      self.w_3 * torch.matmul(adj_1, adj_3)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj

# 通用
class Kernel_Layer(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha(cskl)!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            # self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            # self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
            # self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), math.log(math.exp(1.0) - 1.0)))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
        # self.w_1 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        # self.w_2 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        # self.w_3 = nn.Parameter(torch.full((num_nodes, num_nodes), 1.))
        self.no_self_loop = no_self_loop
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_nodes, 3).float())
        # self.logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        # a1 = F.softplus(self.alpha) + 1e-6
        # a2 = F.softplus(self.alpha2) + 1e-6
        # a3 = F.softplus(self.alpha3) + 1e-6
        a1 = F.relu(self.alpha)
        a2 = F.relu(self.alpha2)
        a3 = F.relu(self.alpha3)
        adj_1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj_2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj_3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        pi = F.softmax(self.logits, dim=-1)
        # adj = pi[0] * adj_1 + pi[1] * adj_2 + pi[2] * adj_3
        adj = pi[..., 0] * adj_1 + pi[..., 1] * adj_2 + pi[..., 2] * adj_3
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj

# adakernel
class Kernel_Layer_ada(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer_ada, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        self.use_alpha = use_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!(adakernel)<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>>>>>init_alpha=", kernel_alpha, "<<<<<<<")
            if kernel_alpha == 0:
                self.alpha = nn.Parameter(torch.abs(torch.randn(num_nodes, num_nodes) * 0.1 + 1.0))
            elif kernel_alpha == -1:
                self.alpha = nn.Parameter(0.9 * torch.rand(num_nodes, num_nodes) + 0.1)
            else:
                self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop
        print(">>>>>>>>>>theta=", theta, "<<<<<<<<<<<<<<")

    def forward(self, dist, sigma):
        alpha_ = nn.functional.relu(self.alpha)
        adj = torch.exp(-torch.square(dist / (alpha_ * sigma + 1e-9)))
        # adj = torch.exp(-torch.square(dist / (self.alpha * sigma)))
        # adj = self.alpha * torch.exp(-torch.square(dist / (sigma)))
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# class Kernel_Layer1(nn.Module):
#     def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
#         super(Kernel_Layer1, self).__init__()
#         if use_alpha:
#             self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
#         else:
#             self.alpha = torch.tensor(kernel_alpha)
#         self.no_self_loop = no_self_loop
#         self.a = 1.
#         if use_alpha == 1:
#             print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
#         else:
#             print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
#         print(">>>>>>>>>>>>>>>>>>>>>>>use Rational Quadratic kernel!!!<<<<<<<<<<<<<<<<<<<<<<<")
#
#     def forward(self, dist, sigma, theta=0.1):
#         self.alpha_ = nn.functional.relu(self.alpha)
#         scaled_distance_RQ = (dist ** 2) / (2 * self.alpha_ * sigma ** 2 + 1e-9)
#         adj = (1 + scaled_distance_RQ).pow(-1)
#         msk = adj > theta
#         adj = adj * msk
#         if self.no_self_loop:
#             adj = adj - torch.diag(torch.diag(adj))
#         return adj
#
#
# class Kernel_Layer2(nn.Module):
#     def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False):
#         super(Kernel_Layer2, self).__init__()
#         if use_alpha:
#             self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
#         else:
#             self.alpha = torch.tensor(kernel_alpha)
#         self.no_self_loop = no_self_loop
#         if use_alpha == 1:
#             print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
#         else:
#             print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
#         print(">>>>>>>>>>>>>>>>>>>>>>>use Matern kernel!!!<<<<<<<<<<<<<<<<<<<<<<<")
#
#     def forward(self, dist, sigma, theta=0.1):
#         self.alpha_ = nn.functional.relu(self.alpha)
#         scaled_distance_MT = dist / (sigma * self.alpha_ + 1e-9)
#         adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
#             -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
#         msk = adj > theta
#         adj = adj * msk
#         if self.no_self_loop:
#             adj = adj - torch.diag(torch.diag(adj))
#         return adj


# def normalize_kernel(K, mode='row', eps=1e-8):
#     if mode == 'global':
#         norm = torch.sum(K) + eps
#         return K / norm
#     elif mode == 'row':
#         norm = torch.sum(K, dim=1, keepdim=True) + eps
#         return K / norm
#     else:
#         raise ValueError("未识别的归一化模式：请选择 'global' 或 'row'")