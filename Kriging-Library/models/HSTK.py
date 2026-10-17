# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:14:13 2024

@author: AA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalKernel(nn.Module):
    def __init__(self):
        super(TemporalKernel, self).__init__()
        # Define learnable parameters
        self.sigma2 = nn.Parameter(torch.rand(1))
        self.ell_p = nn.Parameter(torch.rand(1))
        self.p = nn.Parameter(torch.rand(1))
        self.ell_eq = nn.Parameter(torch.rand(1))

    def forward(self, d):
        # Compute kernel using learnable parameters
        return compute_temporal(d, [self.sigma2, self.ell_p, self.p, self.ell_eq])


class SpatialKernel(nn.Module):
    def __init__(self):
        super( SpatialKernel, self).__init__()
        # Define learnable parameters
        self.sigma2 = nn.Parameter(torch.rand(1))
        self.ell_eq = nn.Parameter(torch.rand(1))

    def forward(self, d, inf_msk):
        # Compute kernel using learnable parameters
        return compute_spatial(d, inf_msk, [self.sigma2, self.ell_eq])


# Updated compute_kernel function to use matrix input `d`
def compute_temporal(d, params):
    sigma2, ell_p, p, ell_eq = params
    periodic_part = torch.exp(-2 * torch.sin(torch.pi * torch.abs(d) / p)**2 / ell_p**2)
    exp_part = torch.exp(-(d)**2 / (2 * ell_eq**2))
    return sigma2 * periodic_part * exp_part


def compute_spatial(d, inf_msk, params):
    sigma2,  ell_eq = params
    exp_part = torch.exp(-(d) ** 2 / (2 * ell_eq ** 2))
    exp_part = exp_part * inf_msk
    return sigma2 * exp_part


def generate_random_data(batch_size, num_timesteps, num_nodes):
    X = torch.rand(batch_size, num_timesteps, num_nodes)
    D_T = torch.rand(num_timesteps, num_timesteps)  # 时间距离矩阵
    D_S = torch.rand(num_nodes, num_nodes)          # 空间距离矩阵
    return X, D_T, D_S


'''
Buitld the GNN
'''
class Model(nn.Module):
    """
    L the layer of ST kernel
    """
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.L = args.L_HSTK  # Number of layers
        # Initialize the temporal and spatial kernels for each layer
        self.temporal_kernels = nn.ModuleList([TemporalKernel() for _ in range(self.L)])
        self.spatial_kernels = nn.ModuleList([SpatialKernel() for _ in range(self.L)])

    def forward(self, x, add_data):
        D_T = add_data['D_T']
        D_S = add_data['D_S']
        inf_msk = add_data['inf_msk']

        for l in range(self.L):
            x_ipt = x

            # Compute temporal and spatial kernels for the current layer
            T_kernel = self.temporal_kernels[l](D_T)  # Temporal kernel of shape (num_timesteps, num_timesteps)
            S_kernel = self.spatial_kernels[l](D_S, inf_msk)   # Spatial kernel of shape (num_nodes, num_nodes)
        
            # Update X by applying spatial and temporal transformations
            # Spatial dimension: matrix multiplication with S_kernel
            x = torch.matmul(x, S_kernel)  # Shape: (batch_size, num_timesteps, num_nodes)
        
            # Temporal dimension: matrix multiplication with T_kernel along time steps
            x = torch.matmul(T_kernel, x)   # Shape: (batch_size, num_timesteps, num_nodes)
            x = F.relu(x)
            x = x + x_ipt

        # T_kernel = self.temporal_kernels[-1](D_T)  # Temporal kernel of shape (num_timesteps, num_timesteps)
        # S_kernel = self.spatial_kernels[-1](D_S, inf_msk)  # Spatial kernel of shape (num_nodes, num_nodes)
        # x = torch.matmul(x, S_kernel)  # Shape: (batch_size, num_timesteps, num_nodes)
        # x = torch.matmul(T_kernel, x)

        return x

# model_T = TemporalKernel()
# model_S = SpatialKernel()

# def generate_time_distance_matrix(size, scale=1.0):
#     indices = torch.arange(size).float().unsqueeze(1)
#     time_distance_matrix = torch.abs(indices - indices.T) * scale
#     return time_distance_matrix

# # 定义空间距离矩阵的生成函数（使用随机数）
# def generate_space_distance_matrix(size, scale=1.0):
#     return torch.rand(size, size) * scale

# n, m = 5, 4  # 可调整
# time_distance_matrix = generate_time_distance_matrix(n, scale=1.0)
# space_distance_matrix = generate_space_distance_matrix(m, scale=1.0)

# time_kernel_matrix = model_T(time_distance_matrix)
# space_kernel_matrix = model_S(space_distance_matrix)



# # 超参数设置
# batch_size = 2       # 批大小
# num_timesteps = 5    # 时间步数
# num_nodes = 4        # 节点数
# L = 3                # 时空层数
#
# # 生成随机数据
# X, D_T, D_S = generate_random_data(batch_size, num_timesteps, num_nodes)
#
# # 初始化模型
# model = HSTK(L=L)
#
# # 执行前向传播
# output = model(X, D_T, D_S)
#
# # 打印输入和输出
# print("Input X:\n", X)
# print("Temporal Distance Matrix D_T:\n", D_T)
# print("Spatial Distance Matrix D_S:\n", D_S)
# print("Output X_res:\n", output)


