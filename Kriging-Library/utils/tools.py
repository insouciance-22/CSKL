import numpy as np
import scipy.sparse as sp
import torch
import os
import math
import logging
import copy
import random
import matplotlib.pyplot as plt


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def plot_time_series(data,
                     title='Time Series Visualization',
                     xlabel='Time Steps',
                     ylabel='PM2.5',
                     colors=None,
                     linewidth=2,
                     linestyle='-'):
    """
    绘制时间序列图

    Parameters:
        data (numpy.ndarray): 输入的矩阵数据，形状为(n_sequences, timesteps)
        title (str): 图表标题，默认为'Time Series Visualization'
        xlabel (str): x轴标签，默认为'Time Steps'
        ylabel (str): y轴标签，默认为'Values'
        colors (list or None): 颜色列表，可以是自定义颜色或预设调色板，默认为tab10
        linewidth (float): 线条宽度，默认为2
        linestyle (str): 线条样式，默认为'-'

    Returns:
        fig: matplotlib Figure 对象，方便进一步修改或保存

    Example:
        data = np.random.randn(5, 20)
        plot_time_series(data, title='My Time Series', xlabel='Time', ylabel='Value')
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    data = data.T
    n_sequences, timesteps = data.shape

    # 自动选择颜色方案
    if colors is None:
        # 使用 tab10 调色板，若序列数较多则使用 tab20
        if n_sequences <= 10:
            colors = plt.cm.tab10(np.arange(n_sequences))
        else:
            colors = plt.cm.tab20(np.arange(n_sequences))

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制每一条序列
    for i in range(n_sequences):
        ax.plot(data[i],
                label=f'Node {i+1}',
                color=colors[i],
                linewidth=linewidth,
                linestyle=linestyle)

    # 设置标题和标签
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    # 添加图例、网格并调整布局
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("PM25.png",
                dpi=300,
                format="png",
                bbox_inches='tight',  # 自动裁剪白边
                pad_inches=0)
    plt.show()


    return fig


def adj_aug(adj, x, val_msk, sub_entry_num):
    n1 = adj.shape[0]
    obs_neighbors = {}
    for i in range(n1):
        row_nonzero = set(np.where(adj[i, :] > 0)[0].tolist())
        # Find nonzero elements in the ith column
        col_nonzero = set(np.where(adj[:, i] > 0)[0].tolist())
        nonzero = row_nonzero.union(col_nonzero)
        obs_neighbors[i] = list(nonzero)  # 1-hop neighbors
    n2 = n1 + sub_entry_num
    adj_aug = np.random.rand(n2, n2)
    adj_aug[:n1, :n1] = adj
    adj_diag = 1. - np.eye(n2)
    adj_aug = adj_aug * adj_diag
    neighbors = copy.deepcopy(obs_neighbors)  # initially has n1 entries' 1-hop neighbors
    adj_aug_mask = np.zeros_like(adj_aug)  # n2, n2
    adj_aug_mask[:n1, :n1] = 1
    for i in range(n1, n2):
        n_current = range(len(neighbors.keys()))  # number of current entries (obs and already added virtual)
        rand_entry = random.sample(n_current, 1)[0]  # randomly sample 1 entry (obs or already added virtual)
        rand_neighbors = neighbors[rand_entry]  # get 1-hop neighbors of sampled entry
        p = np.random.rand(1)  # randomly generate a probability

        # randomly select neighbors
        valid_neighbors = (np.random.rand(len(rand_neighbors)) < p).astype(int)
        valid_neighbors = np.where(valid_neighbors == 1)[0].tolist()
        valid_neighbors = [rand_neighbors[idx] for idx in valid_neighbors]
        all_entries = [rand_entry]
        all_entries.extend(valid_neighbors)

        # add current virtual entry to the 1-hop neighbors of selected entries
        for entry in all_entries:
            neighbors[entry].append(i)

        # add selected entries to the 1-hop neighbors of current virtual entry
        neighbors[i] = all_entries

        options = [0, 1, 2]  # 0: forward; 1: backward; 2: bi-direction
        connected_conditions = [random.choice(options) for _ in range(len(all_entries))]
        for j in range(len(all_entries)):
            entry = all_entries[j]
            condition = connected_conditions[j]

            if condition == 0 or condition == 2:
                adj_aug_mask[entry, i] = 1
            if condition == 1 or condition == 2:
                adj_aug_mask[i, entry] = 1

    adj_aug = adj_aug * adj_aug_mask
    # print(adj_aug[100, 101])
    x_aug = np.zeros([x.shape[0], n2])
    x_aug[:, :n1] = x
    val_msk_aug = np.ones([val_msk.shape[0], n2])
    val_msk_aug[:, :n1] = val_msk
    A = adj_aug + adj_aug.T
    mask = (x_aug == 0).astype(float)
    neighbor_sum = np.einsum('ij,kj->ki', A, x_aug)
    valid_neighbor_count = np.einsum('ij,kj->ki', A, (x_aug != 0).astype(float))
    neighbor_avg = neighbor_sum / (valid_neighbor_count + 1e-10)
    # col_mean = np.mean(x, axis=1)

    # Find columns in neighbor_avg that are all zeros
    # zero_cols = np.all(neighbor_avg == 0, axis=0)
    # #print(zero_cols)
    # # Replace the all-zero columns with the corresponding column mean from x
    # for idx in np.where(zero_cols)[0]:
    #     neighbor_avg[:, idx] = col_mean
    x_filled = np.where(mask == 1, neighbor_avg, x_aug)
    for i in range(n2):
        adj_aug[i, i] = 1

    return adj_aug, x_filled, val_msk_aug


def normalize_meta(*args):
    # 接收多个参数组成的元组，用列表推导式对每个参数进行标准化处理
    normalized_arrays = [np.expand_dims((arg - np.mean(arg)) / np.std(arg), -1) for arg in args]
    # 将所有标准化后的数组沿最后一个轴拼接起来
    return np.concatenate(normalized_arrays, axis=-1)


def fill_zeros_idw(X, A, k=5):
    # 获取时间序列长度 T 和节点数量 N
    T, N = X.shape
    # 初始化输出数组为原始数据的副本
    X_filled = np.copy(X)
    A_copy = np.copy(A)
    np.fill_diagonal(A_copy, -1)
    neighbors_idx = np.argsort(A_copy)[:, -k:]
    for i in range(N):
        # 使用 advanced indexing 获取所有时间点的K个邻居的值
        neighbors_values = X[:, neighbors_idx[i]] #(T, K)
        # 获取点i与k个近邻的关系
        neighbors_relation = A_copy[i, neighbors_idx[i]]
        zero_mask_sub = (neighbors_values == 0)
        neighbors_relation = np.tile(neighbors_relation, (neighbors_values.shape[0], 1))
        neighbors_relation[zero_mask_sub] = 0
        # 对所有点的关系归一化
        neighbors_relation = neighbors_relation / (np.sum(neighbors_relation, axis=1, keepdims=True) + 1e-6)
        # 根据关系进行加权平均
        mean_values = np.sum(neighbors_values * neighbors_relation, axis=1)
        # 仅更新原始矩阵中为零的位置
        zero_mask = (X_filled[:, i] == 0)
        X_filled[:, i][zero_mask] = mean_values[zero_mask]
    return X_filled


def fill_zeros_knn(X, A, k=5):
    # 获取时间序列长度 T 和节点数量 N
    T, N = X.shape
    # 初始化输出数组为原始数据的副本
    X_filled = np.copy(X)
    A_copy = np.copy(A)
    np.fill_diagonal(A_copy, -1)
    neighbors_idx = np.argsort(A_copy)[:, -k:]
    for i in range(N):
        # 使用 advanced indexing 获取所有时间点的K个邻居的值
        neighbors_values = X[:, neighbors_idx[i]] #(T, K)
        # neighbors_relation = A_copy[i, neighbors_idx[i]]
        # neighbors_relation = neighbors_relation / (np.sum(neighbors_relation) + 1e-6)
        # mean_values = neighbors_values @ neighbors_relation
        # 计算这些邻居在每个时间点的平均值，忽略零值
        # 使用 np.where 来避免除以零
        sum_neighbors = np.sum(neighbors_values, axis=1) #(T,)
        count_non_zeros = np.count_nonzero(neighbors_values, axis=1)
        mean_values = np.where(count_non_zeros != 0, sum_neighbors / count_non_zeros, 0)
        # 仅更新原始矩阵中为零的位置
        zero_mask = (X_filled[:, i] == 0)
        X_filled[:, i][zero_mask] = mean_values[zero_mask]
    return X_filled


def fill_nan_knn(X, A, k=5):
    T, N = X.shape
    X_filled = np.copy(X)
    nan_msks = np.isnan(X)
    X[nan_msks] = 0.
    A_copy = np.copy(A)
    np.fill_diagonal(A_copy, -1)
    neighbors_idx = np.argsort(A_copy)[:, -k:]
    for i in range(N):
        neighbors_values = X[:, neighbors_idx[i]] #(T, K)
        sum_neighbors = np.sum(neighbors_values, axis=1) #(T,)
        count_non_nan = k - np.count_nonzero(nan_msks, axis=1)
        mean_values = np.where(count_non_nan != 0, sum_neighbors / count_non_nan, 0)
        # 仅更新原始矩阵中为nan的位置
        nan_msk = nan_msks[:, i]
        X_filled[:, i][nan_msk] = mean_values[nan_msk]
    return X_filled


def calculate_random_walk_matrix(adj_mx):
    """
    This is for IGNNK
    1. Returns the random walk adjacency matrix.(×)
    2. Normalize adj with degree.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def get_similarity(adj_sc, thr=0.1, force_symmetric=False, sparse=False, distance=False):
    """
    This is for KITS
    1. Convert Distance matrix to Probability matrix
    2. Remove self connection
    """
    if distance:
        finite_dist =adj_sc.reshape(-1)
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        sigma = finite_dist.std()
        adj = np.exp(-np.square(adj_sc / sigma))
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
    else:
        adj = adj_sc
    np.fill_diagonal(adj, 0.)
    return adj


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, pre_training=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.pre_training = pre_training

    def __call__(self, val_loss, model, path, add_data=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, add_data)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, add_data)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, add_data):
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.pre_training:
            torch.save(model.state_dict(), os.path.join(path, 'pre_training_checkpoint.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        if add_data is not None:
            np.save(os.path.join(path, 'add_data.npy'), add_data)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type0':
        # print('Skipping learning rate adjustment for this epoch')
        return
    elif args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info('Updating learning rate to {}'.format(lr))


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def softmax_non_zero(x, axis):
    # 创建一个掩码，标识非零元素的位置
    non_zero_mask = x != 0
    # 创建一个输出数组，初始化为原数组的拷贝
    # 这步是可选的，取决于你是否需要保留原数组不变
    output = np.copy(x)
    # 在非零元素上应用 softmax
    # 使用掩码选择非零元素，对这部分元素应用 softmax
    # 然后将结果放回对应的位置
    output[non_zero_mask] = softmax(x[non_zero_mask])

    return output



