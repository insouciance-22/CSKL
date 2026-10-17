import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 适用于imputation、forecasting
class Kernel_Layer1(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer1, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha2 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
            self.alpha3 = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop
        # 初始化比例参数
        # self.logits = nn.Parameter(torch.zeros(num_nodes, num_nodes, 3).float())
        self.logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        self.alpha_ = F.relu(self.alpha)
        self.alpha2_ = F.relu(self.alpha2)
        self.alpha3_ = F.relu(self.alpha3)
        adj_1 = torch.exp(-torch.square(dist / (self.alpha_ * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * self.alpha2_ * sigma ** 2 + 1e-9)
        adj_2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * self.alpha3_ + 1e-9)
        adj_3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        pi = F.softmax(self.logits, dim=-1)
        adj = pi[..., 0] * adj_1 + pi[..., 1] * adj_2 + pi[..., 2] * adj_3
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 与前一个方法相同，单个变量取代矩阵，适用于kriging（仅对IGNNK有效）
class Kernel_Layer2(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer2, self).__init__()
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
        self._pi_logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        # a1 = nn.functional.relu(self.alpha)+ 1e-6
        # a2 = nn.functional.relu(self.alpha2)+ 1e-6
        # a3 = nn.functional.relu(self.alpha3)+ 1e-6
        # 采用softplus代替relu，避免梯度截断
        a1 = F.softplus(self.alpha) + 1e-6
        a2 = F.softplus(self.alpha2) + 1e-6
        a3 = F.softplus(self.alpha3) + 1e-6
        adj1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        pi = F.softmax(self._pi_logits, dim=0)
        adj = pi[0] * adj1 + pi[1] * adj2 + pi[2] * adj3

        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 适用于kriging（对ignnk、kits有效，但不能稳定提升）
class Kernel_Layer3(nn.Module):
    def __init__(self, use_alpha=True, kernel_alpha=1., no_self_loop=False):
        super(Kernel_Layer3, self).__init__()
        self.theta = 0.
        self.no_self_loop = no_self_loop
        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.alpha2 = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            self.alpha3 = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1.0)]))
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            self.alpha = torch.tensor(kernel_alpha)
            self.alpha2 = torch.tensor(kernel_alpha)
            self.alpha3 = torch.tensor(kernel_alpha)
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        self._pi_logits = nn.Parameter(torch.zeros(3).float())

    def forward(self, dist, sigma):
        a1 = F.softplus(self.alpha) + 1e-6
        a2 = F.softplus(self.alpha2) + 1e-6
        a3 = F.softplus(self.alpha3) + 1e-6
        adj1 = torch.exp(-torch.square(dist / (a1 * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * a2 * sigma ** 2 + 1e-9)
        adj2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * a3 + 1e-9)
        adj3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        # 融合方式
        adj = (self.w1 * adj1) * (self.w2 * adj2) * (self.w3 * adj3)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj


# 适用于imputation（仅在imputation上最优）
class Kernel_Layer4(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer4, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        if use_alpha:
            print(">>>>>>>>>>>>>>>>>>>>>>>use multy learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
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
        self.alpha_ = F.relu(self.alpha)
        self.alpha2_ = F.relu(self.alpha2)
        self.alpha3_ = F.relu(self.alpha3)
        adj_1 = torch.exp(-torch.square(dist / (self.alpha_ * sigma + 1e-9)))
        scaled_distance_RQ = (dist ** 2) / (2 * self.alpha2_ * sigma ** 2 + 1e-9)
        adj_2 = (1 + scaled_distance_RQ).pow(-1)
        scaled_distance_MT = dist / (sigma * self.alpha3_ + 1e-9)
        adj_3 = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
       # 融合方式
        adj = self.w_1 * torch.matmul(adj_1, adj_2) + \
                      self.w_2 * torch.matmul(adj_2, adj_3) + \
                      self.w_3 * torch.matmul(adj_1, adj_3)
        msk = adj > self.theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj