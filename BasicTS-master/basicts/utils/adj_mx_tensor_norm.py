import torch
import numpy as np


def calculate_adj(adj_sc, adj_type):
    adj = []
    if adj_type == 'scalap':
        adj.append(calculate_scaled_laplacian(adj_sc))  # 转换为tensor
    elif adj_type == 'normlap':
        adj.append(calculate_symmetric_normalized_laplacian(adj_sc))  # 转换为tensor
    elif adj_type == 'symnadj':
        adj.append(calculate_symmetric_message_passing_adj(adj_sc))  # 转换为tensor
    elif adj_type == 'transition':
        adj.append(calculate_transition_matrix(adj_sc).T)  # 转换为tensor
    elif adj_type == 'doubletransition':
        adj.append(calculate_transition_matrix(adj_sc).T)  # 转换为tensor
        adj.append(calculate_transition_matrix(adj_sc.T).T)  # 转换为tensor
    elif adj_type == 'identity':
        adj.append(torch.diag(torch.ones(adj_sc.shape[0], dtype=torch.float32)))  # 创建tensor类型的单位矩阵
    elif adj_type == 'original':
        adj.append(adj_sc)  # 确保转换为tensor类型
    elif adj_type == 'doubletransition1':
        adj.append(calculate_transition_matrix(adj_sc))  # 转换为tensor
        adj.append(calculate_transition_matrix(adj_sc.T))  # 转换为tensor
    return adj, adj_sc


def calculate_symmetric_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Calculate the symmetric normalized Laplacian using PyTorch tensors for dense matrices.
    The symmetric normalized Laplacian is computed as:
    L^{Sym} = I - D^{-1/2} A D^{-1/2},
    where A is the adjacency matrix, D is the degree matrix.

    Args:
        adj (torch.Tensor): Adjacency matrix A (n x n).

    Returns:
        torch.Tensor: Symmetric normalized Laplacian L^{Sym} (n x n).
    """
    # 计算度数向量 D (行和)
    degree = adj.sum(dim=1)  # 每行的和，即节点的度数

    # 计算 D^{-1/2}
    degree_inv_sqrt = degree.pow(-0.5)  # 度数倒数平方根
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 处理 degree 为零的情况，防止除零错误

    # 创建 D^{-1/2} 对角矩阵
    degree_inv_sqrt_mat = torch.diag(degree_inv_sqrt)

    # 计算对称归一化拉普拉斯矩阵 L^{Sym} = I - D^{-1/2} A D^{-1/2}
    identity = torch.eye(adj.shape[0], device=adj.device)  # 单位矩阵 I
    laplacian = identity - degree_inv_sqrt_mat @ adj @ degree_inv_sqrt_mat  # 矩阵乘法

    return laplacian


def calculate_scaled_laplacian(adj: torch.Tensor, lambda_max: float = 2.0, undirected: bool = True) -> torch.Tensor:
    """
    Calculate the scaled Laplacian for Chebyshev polynomials using PyTorch tensors for dense matrices.

    Args:
        adj (torch.Tensor): Adjacency matrix A (n x n).
        lambda_max (float, optional): Maximum eigenvalue for scaling. Default is 2.
        undirected (bool, optional): If True, the graph is treated as undirected.

    Returns:
        torch.Tensor: Scaled Laplacian matrix.
    """
    if undirected:
        adj = torch.maximum(adj, adj.t())  # Ensure symmetry for undirected graph

    # 计算对称归一化拉普拉斯矩阵
    laplacian = calculate_symmetric_normalized_laplacian(adj)

    # 计算最大特征值
    if lambda_max is None:
        # 这里我们假设已知 lambda_max，通常可以通过谱分解求得
        lambda_max = torch.max(torch.eig(laplacian, eigenvectors=False)[0][:, 0])  # 取最大特征值

    # 缩放拉普拉斯矩阵
    identity = torch.eye(adj.shape[0], device=adj.device)
    scaled_laplacian = (2 / lambda_max) * laplacian - identity  # 归一化拉普拉斯缩放

    return scaled_laplacian


def calculate_symmetric_message_passing_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Calculate the symmetric message-passing adjacency matrix using PyTorch tensors for dense matrices.

    The message-passing adjacency matrix is defined as A' = D^{-1/2} (A + I) D^{-1/2}.

    Args:
        adj (torch.Tensor): Adjacency matrix A (n x n).

    Returns:
        torch.Tensor: Renormalized message-passing adjacency matrix A' (n x n).
    """
    # 为了自环，加上单位矩阵 I
    adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=torch.float32)

    # 计算每行的度数
    degree = adj.sum(dim=1)  # 每行的和
    degree_inv_sqrt = degree.pow(-0.5)  # 度数的倒数平方根
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 处理度数为零的情况

    # 构造 D^{-1/2}
    degree_inv_sqrt_mat = torch.diag(degree_inv_sqrt)

    # 计算消息传递邻接矩阵
    mp_adj = degree_inv_sqrt_mat @ adj @ degree_inv_sqrt_mat

    return mp_adj


def calculate_transition_matrix(adj: torch.Tensor) -> torch.Tensor:
    """
    Calculate the transition matrix P using PyTorch tensors for dense matrices.

    The transition matrix is defined as P = D^{-1} A, where D is the degree matrix.

    Args:
        adj (torch.Tensor): Adjacency matrix A (n x n).

    Returns:
        torch.Tensor: Transition matrix P (n x n).
    """
    # 计算每行的度数
    degree = adj.sum(dim=1)  # 每行的和
    degree_inv = degree.pow(-1)  # 度数的倒数
    degree_inv[torch.isinf(degree_inv)] = 0.  # 处理度数为零的情况

    # 构造 D^{-1}
    degree_inv_mat = torch.diag(degree_inv)

    # 计算转移矩阵
    transition_matrix = degree_inv_mat @ adj

    return transition_matrix