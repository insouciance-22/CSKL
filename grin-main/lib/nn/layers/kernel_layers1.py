import torch
import torch.nn as nn


class Kernel_Layer(nn.Module):
    def __init__(self, num_nodes, use_alpha=True, kernel_alpha=1., no_self_loop=False, theta=0.1):
        super(Kernel_Layer, self).__init__()
        self.theta = theta
        self.kernel_alpha = kernel_alpha
        if use_alpha:
            if kernel_alpha == 0:
                self.alpha = nn.Parameter(torch.abs(torch.randn(num_nodes, num_nodes) * 0.1 + 1.0))
            elif kernel_alpha == -1:
                self.alpha = nn.Parameter(0.9 * torch.rand(num_nodes, num_nodes) + 0.1)
            else:
                self.alpha = nn.Parameter(torch.full((num_nodes, num_nodes), kernel_alpha))
        else:
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop
        if use_alpha == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>>>>>init_alpha=", kernel_alpha, "<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        print(">>>>>>>>>>theta=", theta, "<<<<<<<<<<<<<<")


    def forward(self, dist, sigma):
        self.alpha_ = nn.functional.relu(self.alpha)
        adj = torch.exp(-torch.square(dist / (self.alpha_ * sigma + 1e-9)))
        # adj = torch.exp(-torch.square(dist / (self.alpha * sigma)))
        # adj = self.alpha * torch.exp(-torch.square(dist / (sigma)))
        msk = adj > self.theta
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
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop
        self.a = 1.
        if use_alpha == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        print(">>>>>>>>>>>>>>>>>>>>>>>use Rational Quadratic kernel!!!<<<<<<<<<<<<<<<<<<<<<<<")

    def forward(self, dist, sigma, theta=0.1):
        self.alpha_ = nn.functional.relu(self.alpha)
        scaled_distance_RQ = (dist ** 2) / (2 * self.alpha_ * sigma ** 2 + 1e-9)
        adj = (1 + scaled_distance_RQ).pow(-1)
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
            self.alpha = torch.tensor(kernel_alpha)
        self.no_self_loop = no_self_loop
        if use_alpha == 1:
            print(">>>>>>>>>>>>>>>>>>>>>>>use learnable alpha!!!<<<<<<<<<<<<<<<<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>use static alpha:", kernel_alpha, "!!!<<<<<<<<<<<<<<<<<<<<<<<")
        print(">>>>>>>>>>>>>>>>>>>>>>>use Matern kernel!!!<<<<<<<<<<<<<<<<<<<<<<<")

    def forward(self, dist, sigma, theta=0.1):
        self.alpha_ = nn.functional.relu(self.alpha)
        scaled_distance_MT = dist / (sigma * self.alpha_ + 1e-9)
        adj = (1 + torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT) * torch.exp(
            -torch.sqrt(torch.tensor(3.0)) * scaled_distance_MT)
        msk = adj > theta
        adj = adj * msk
        if self.no_self_loop:
            adj = adj - torch.diag(torch.diag(adj))
        return adj
