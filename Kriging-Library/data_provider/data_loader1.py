import numpy as np
from data_provider.data_set import NormalDataset
from utils.tools import calculate_random_walk_matrix, softmax, adj_aug, fill_zeros_knn, fill_zeros_idw
from utils.time_features import time_encoder
import warnings
import random
import torch
import pandas as pd
# from data_provider.data_set import GNNDataset
# from torch_geometric.data import Data
import os.path


warnings.filterwarnings('ignore')


class Dataset_IGNNK(NormalDataset):
    def __init__(self, args, device,
                tra_s, tra_m,
                dataset, N_test, N_val, T_test, T_val, seq_len,
                flag='train', ignore_zero=True, fill_zero=False, test_name="",
                scaler_method='fixed', scaler_value=50,
                train_unknow_set=None, train_val_set=None):

        self.tra_s = tra_s
        self.tra_m = tra_m
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)

    def __getitem__(self, index):
        ground_truth, valid_msk = super().__getitem__(index)  # (T, N)
        if self.flag == 'train':
            know_msk = list(
                set(random.sample(range(0, ground_truth.shape[1]), int(self.tra_s * ground_truth.shape[1]))))
            seq_y = ground_truth[:, know_msk]
            seq_x = seq_y.copy()
            unknow_msk = set(random.sample(range(0, seq_x.shape[1]), int(self.tra_m * seq_x.shape[1])))
            A_dynamic = self.adj[know_msk, :][:, know_msk]
            seq_x[:, list(unknow_msk)] = 0
            # seq_msk = np.zeros_like(seq_x)
            # seq_msk[:, list(unknow_msk)] = 1
            seq_valid_msk = valid_msk[:, know_msk]
            seq_msk = seq_valid_msk
        elif self.flag == 'val':
            seq_y = ground_truth
            seq_x = seq_y.copy()
            seq_x[:, list(self.val_set)] = 0
            A_dynamic = self.adj
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(self.val_set)] = 1
            seq_valid_msk = valid_msk
            # seq_msk = seq_valid_msk
        else:
            seq_y = ground_truth
            seq_x = seq_y.copy()
            seq_x[:, list(self.unknow_set)] = 0
            A_dynamic = self.adj
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(self.unknow_set)] = 1
            seq_valid_msk = valid_msk

        seq_msk = seq_msk * seq_valid_msk
        A_q = calculate_random_walk_matrix(A_dynamic).T
        A_h = calculate_random_walk_matrix(A_dynamic.T).T
        # A_q = A_dynamic.T.T
        # A_h = A_dynamic.T

        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        A_q = torch.from_numpy(A_q.astype('float32')).to(self.device)
        A_h = torch.from_numpy(A_h.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        # seq_valid_msk = torch.from_numpy(seq_valid_msk.astype(float)).to(self.device)

        add_data = {
            'A_q': A_q,
            'A_h': A_h
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }


class Dataset_IGNNK_1(NormalDataset):
    # each sample of the same batch uses the same adj
    def __init__(self, args, device,
                tra_s, tra_m,
                dataset, N_test, N_val, T_test, T_val, seq_len,
                flag='train', ignore_zero=True, fill_zero=False, test_name="",
                scaler_method='standard', scaler_value=50,
                train_unknow_set=None, train_val_set=None):

        self.tra_s = tra_s
        self.tra_m = tra_m
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)

    def __getitem__(self, index):
        seq_y, seq_valid_msk = super().__getitem__(index)  # (T, N)
        return seq_y, seq_valid_msk

    def collate_fn(self, batch):
        seq_y, seq_valid_msk = map(np.stack, zip(*batch)) # (B, T, N)

        if self.flag == 'train':
            know_msk = sorted(random.sample(range(0, seq_y.shape[-1]), int(self.tra_s * seq_y.shape[-1])))
            seq_y = seq_y[:, :, know_msk]
            seq_x = seq_y.copy()
            unknow_msk = sorted(random.sample(range(0, seq_x.shape[-1]), int(self.tra_m * seq_x.shape[-1])))
            A_dynamic = self.adj[know_msk, :][:, know_msk]
            seq_x[:, :, unknow_msk] = 0
            # seq_msk = np.zeros_like(seq_x)
            # seq_msk[:, list(unknow_msk)] = 1
            seq_valid_msk = seq_valid_msk[:, :, know_msk]
            seq_msk = seq_valid_msk
        elif self.flag == 'val':
            seq_x = seq_y.copy()
            seq_x[:, :, self.val_set] = 0
            seq_msk = np.zeros_like(seq_y)
            seq_msk[:, :, self.val_set] = 1
            A_dynamic = self.adj
        else:
            seq_x = seq_y.copy()
            seq_x[:, :, self.unknow_set] = 0
            seq_msk = np.zeros_like(seq_y)
            seq_msk[:, :, self.unknow_set] = 1
            A_dynamic = self.adj
        seq_msk = seq_msk * seq_valid_msk
        A_q = calculate_random_walk_matrix(A_dynamic).T
        A_h = calculate_random_walk_matrix(A_dynamic.T).T

        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        A_q = torch.from_numpy(A_q.astype('float32')).to(self.device)
        A_h = torch.from_numpy(A_h.astype('float32')).to(self.device)
        # seq_valid_msk = torch.from_numpy(seq_valid_msk.astype(float)).to(self.device)

        add_data = {
            'A_q': A_q,
            'A_h': A_h
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }


class Dataset_KITS(NormalDataset):
    def __init__(self, args, device,
                 dataset, N_test, N_val, T_test, T_val, seq_len,
                 flag='train', ignore_zero=True, fill_zero=False, test_name="",
                 scaler_method='standard', scaler_value=None,
                 train_unknow_set=None, train_val_set=None):
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)
        self.ratio = float(len(self.tra_set) / (len(self.unknow_set) + len(self.tra_set)))

    def __getitem__(self, index):
        seq_x, seq_valid_msk = super().__getitem__(index)  # (T, N)
        seq_y = seq_x.copy()
        return seq_x, seq_y, seq_valid_msk

    def collate_fn(self, batch):
        # kits输入的msk_input不同于用于criterion的msk
        seq_x, seq_y, seq_valid_msk = map(np.stack, zip(*batch)) # (B, T, N)
        seq_msk = np.zeros_like(seq_x)
        if self.flag == 'train':
            dynamic_ratio = self.ratio + 0.2 * random.random()
            cur_node_num = seq_x.shape[2]  # n1
            aug_node_num = int(cur_node_num / dynamic_ratio)  # n2
            sub_node_num = aug_node_num - cur_node_num  # n2 - n1
            seq_sub_x = np.zeros((seq_x.shape[0], seq_x.shape[1], sub_node_num))
            seq_x = np.concatenate((seq_x, seq_sub_x), axis=2)
            seq_y = np.concatenate((seq_y, seq_sub_x), axis=2)
            seq_valid_msk = np.concatenate((seq_valid_msk, seq_sub_x), axis=2)
            seq_msk = np.concatenate((seq_msk, 1-seq_sub_x), axis=2)
        elif self.flag == 'val':
            seq_msk[:, :, list(self.val_set)] = 1
        else:
            seq_msk[:, :, list(self.unknow_set)] = 1

        seq_x = seq_x * (1 - seq_msk)
        seq_msk_input = (1 - seq_msk) * seq_valid_msk
        seq_msk = seq_msk * seq_valid_msk

        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        # seq_valid_msk = torch.from_numpy(seq_valid_msk.astype(float)).to(self.device)
        adj = torch.from_numpy(self.adj.astype('float32')).to(self.device)
        seq_msk_input = torch.from_numpy(seq_msk_input.astype('float32')).to(self.device)

        add_data = {
            'A': adj,
            'msk_input': seq_msk_input
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }


class Dataset_INCREASE(NormalDataset):
    def __init__(self, args, device,
                 k_KNN, interval,
                 dataset, N_test, N_val, T_test, T_val, seq_len,
                 flag='train', ignore_zero=True, fill_zero=False, test_name="",
                 scaler_method='standard', scaler_value=None,
                 train_unknow_set=None, train_val_set=None):
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)
        self.K = k_KNN
        self.interval = interval
        if self.flag == 'train':
            self.target_set = np.arange(len(self.tra_set))
            self.relation, self.neighbor = self.heterogeneous_relations(self.adj, self.K)
            self.relation2, self.neighbor2 = self.heterogeneous_relations(self.adj.T, self.K)
        elif self.flag == 'val':
            self.target_set = list(self.val_set)
            self.relation, self.neighbor = self.heterogeneous_relations(self.adj, self.K, self.tra_set)
            self.relation2, self.neighbor2 = self.heterogeneous_relations(self.adj.T, self.K, self.tra_set)
        else:
            self.target_set = list(self.unknow_set)
            self.relation, self.neighbor = self.heterogeneous_relations(self.adj, self.K, self.know_set)
            self.relation2, self.neighbor2 = self.heterogeneous_relations(self.adj, self.K, self.know_set)
        # (N, K)->(N, 1, 1, K)
        self.relation = self.relation[self.target_set, np.newaxis, np.newaxis]
        self.relation2 = self.relation2[self.target_set, np.newaxis, np.newaxis]
        # (T, N_train)->(T, N_tar, K, 1)
        # (T, N_train+N_val)->(T, N_val, K, 1)
        # (T, N_all)->(T, N_test, K, 1)
        self.x = self.gt[:, self.neighbor[self.target_set], np.newaxis]
        self.x2 = self.gt[:, self.neighbor2[self.target_set], np.newaxis]
        self.gt = self.gt[:, self.target_set]
        self.valid_mask = self.valid_mask[:, self.target_set]

        # add data--TimeEncoder(TE)
        length = self.data_sc.shape[0]
        enc_type = 'timeofday'
        TE = time_encoder(length, self.interval, enc_type)
        TE = TE[np.newaxis, :]
        TE_test = TE[:, self.split_line1:]
        TE_ave = TE[:, :self.split_line1]
        TE_val = TE_ave[:, self.split_line2:]
        TE_train = TE_ave[:, :self.split_line2]
        if self.flag == 'train':
            self.TE = TE_train
        elif self.flag == 'val':
            self.TE = TE_val
        else:
            self.TE = TE_test

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.x[s_begin: s_end]
        seq_x2 = self.x2[s_begin: s_end]
        seq_y, seq_valid_msk = super().__getitem__(index)  # (T, N)
        seq_TE = self.TE[:, s_begin: s_end]
        relation = self.relation
        relation2 = self.relation2
        seq_msk = np.ones_like(seq_y) * seq_valid_msk

        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_x2 = torch.from_numpy(seq_x2.astype('float32')).to(self.device)
        seq_y =  torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        seq_TE = torch.from_numpy(seq_TE.astype('int64')).to(self.device)
        relation = torch.from_numpy(relation.astype('float32')).to(self.device)
        relation2 = torch.from_numpy(relation2.astype('float32')).to(self.device)

        add_data = {
            'x2': seq_x2,
            'relation': relation,
            'relation2': relation2,
            'TE': seq_TE
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }

    def heterogeneous_relations(self, A, K, known_node=None):
        if known_node:
            relation = np.zeros_like(A) - 1
            relation[:, list(known_node)] = A[:, list(known_node)]  # only use the known nodes
        else:
            relation = A
        np.fill_diagonal(relation, val=-1)  # delete self-loop connections
        Neighbor = np.argsort(-relation, axis=1)  # 返回降序索引
        Neighbor = Neighbor[:, :K]
        relation = -np.sort(-relation, axis=-1)  # 返回降序数值
        relation = relation[:, :K]
        relation[relation < 0] = 0
        relation = relation / (1e-10 + np.sum(relation, axis=1, keepdims=True))
        relation = relation.astype(np.float32)
        # (N, K),(N, K)
        # Neighbor表示每个点最相近的K个点的索引，relation代表每个点与其K个最近似点的关系相似度（权重）
        return relation, Neighbor


class Dataset_KNN(NormalDataset):
    def __init__(self, args, device,
                k,
                dataset, N_test, N_val, T_test, T_val, seq_len,
                flag='train', ignore_zero=True, test_name="",
                scaler_method='standard', scaler_value=50,
                train_unknow_set=None, train_val_set=None):
        # ignore_zero = True
        fill_zero = 'knn'
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)
        if self.flag == 'test':
            self.k = k
            self.x = self.gt.copy()
            self.x[:, list(self.unknow_set)] = 0
            if args.model == 'IDW':
                self.x = fill_zeros_idw(self.x, self.adj, args.k_mean)
            else:
                self.x = fill_zeros_knn(self.x, self.adj, args.k_mean)
        # if self.flag != 'train':
        #     self.gt[:, list(self.val_set)] = 0
        # self.gt = fill_zeros_with_neighbors(self.gt, self.adj, 5)

    def __getitem__(self, index):
        ground_truth, valid_msk = super().__getitem__(index)  # (T, N)
        seq_y = ground_truth
        seq_x = self.x[index: index + self.seq_len]
        if self.flag == 'train':
            seq_msk = np.zeros_like(seq_x)
            seq_valid_msk = valid_msk
        elif self.flag == 'val':
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(self.val_set)] = 1
            seq_valid_msk = valid_msk
            seq_msk = seq_valid_msk
        else:
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(self.unknow_set)] = 1
            seq_valid_msk = valid_msk

        seq_msk = seq_msk * seq_valid_msk
        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        # seq_valid_msk = torch.from_numpy(seq_valid_msk.astype(float)).to(self.device)

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
        }


class Dataset_IGNNK_kernel_0(NormalDataset):
    def __init__(self, args, device,
                 tra_s, tra_m,
                 dataset, N_test, N_val, T_test, T_val, seq_len,
                 flag='train', ignore_zero=True, fill_zero=False, test_name="",
                 scaler_method='standard', scaler_value=50,
                 train_unknow_set=None, train_val_set=None):
        self.tra_s = tra_s
        self.tra_m = tra_m
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)
        self.D_T = self.build_distance_matrix().to(self.device)
        self.L = args.L_HSTK
        self.D_S = np.load(os.path.join("dataset", dataset, "adj_dist.npy"))
        # self.D_S = self.normalize_matrix(self.D_S)
        self.inf_msk = 1 - np.isinf(self.D_S)
        self.D_S[np.isinf(self.D_S)] = np.max(self.D_S[~np.isinf(self.D_S)]) + 50

    def __getitem__(self, index):
        seq_y, seq_valid_msk = super().__getitem__(index)  # (T, N)
        return seq_y, seq_valid_msk

    def build_distance_matrix(self):
        T = self.seq_len
        indices = torch.arange(T).float().unsqueeze(1)
        time_distance_matrix = torch.abs(indices - indices.T)
        max_value = torch.max(time_distance_matrix)
        time_distance_matrix = time_distance_matrix / max_value
        return time_distance_matrix

    def normalize_matrix(self, matrix):
        finite_dist = matrix.copy()
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        max_value = np.max(finite_dist)
        normalized_matrix = matrix / max_value

        return normalized_matrix

    def collate_fn(self, batch):
        seq_y, seq_valid_msk = map(np.stack, zip(*batch))  # (B, T, N)
        if self.flag == 'train':
            know_msk = sorted(random.sample(range(0, seq_y.shape[-1]), int(self.tra_s * seq_y.shape[-1])))
            seq_y = seq_y[:, :, know_msk]
            seq_x = seq_y.copy()
            unknow_msk = sorted(random.sample(range(0, seq_x.shape[-1]), int(self.tra_m * seq_x.shape[-1])))
            D_S_dynamic = self.D_S[know_msk, :][:, know_msk]
            inf_msk = self.inf_msk[know_msk, :][:, know_msk]
            seq_x[:, :, unknow_msk] = 0
            # seq_msk = np.zeros_like(seq_x)
            # seq_msk[:, list(unknow_msk)] = 1
            seq_valid_msk = seq_valid_msk[:, :, know_msk]
            seq_msk = seq_valid_msk
        elif self.flag == 'val':
            seq_x = seq_y.copy()
            seq_x[:, :, self.val_set] = 0
            seq_msk = np.zeros_like(seq_y)
            seq_msk[:, :, self.val_set] = 1
            D_S_dynamic = self.D_S[self.know_set, :][:, self.know_set]
            inf_msk = self.inf_msk[self.know_set, :][:, self.know_set]
        else:
            seq_x = seq_y.copy()
            seq_x[:, :, self.unknow_set] = 0
            seq_msk = np.zeros_like(seq_y)
            seq_msk[:, :, self.unknow_set] = 1
            D_S_dynamic = self.D_S
            inf_msk = self.inf_msk
        seq_msk = seq_msk * seq_valid_msk

        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        # D_S = self.normalize_matrix(D_S_dynamic)
        D_S = torch.from_numpy(D_S_dynamic.astype('float32')).to(self.device)
        D_T = self.D_T
        inf_msk = torch.from_numpy(inf_msk.astype('float32')).to(self.device)
        dist = torch.from_numpy(self.D_S.astype('float32')).to(self.device)
        # seq_valid_msk = torch.from_numpy(seq_valid_msk.astype(float)).to(self.device)

        add_data = {
            'D_T': D_T,
            'D_S': D_S,
            'inf_msk': inf_msk,
            'dist': dist
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }


class Dataset_IGNNK_kernel(NormalDataset):
    def __init__(self, args, device,
                tra_s, tra_m,
                dataset, N_test, N_val, T_test, T_val, seq_len,
                flag='train', ignore_zero=True, fill_zero=False, test_name="",
                scaler_method='fixed', scaler_value=50,
                train_unknow_set=None, train_val_set=None):

        self.tra_s = tra_s
        self.tra_m = tra_m
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)

        self.D_S = self.dist_sc.copy()
        # self.D_S = self.trans_to_kernel(self.dist)
        # self.D_S = self.normalize_matrix(self.D_S)
        # self.D_S[np.isinf(self.D_S)] = 1.0
        self.inf_msk = ~np.isinf(self.D_S)
        self.D_S[~self.inf_msk] = np.max(self.D_S[np.isfinite(self.D_S)]) * 10

    def normalize_matrix(self, matrix):
        finite_dist = matrix.copy()
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        max_value = np.max(finite_dist)
        normalized_matrix = matrix / max_value
        return normalized_matrix

    def trans_to_kernel(self, dist):
        sigma = np.std(dist[~np.isinf(dist)])
        matrix = np.exp(-np.square(dist / sigma))
        return matrix

    def build_distance_matrix(self):
        T = self.seq_len
        indices = torch.arange(T).float().unsqueeze(1)
        time_distance_matrix = torch.abs(indices - indices.T)
        max_value = torch.max(time_distance_matrix)
        time_distance_matrix = time_distance_matrix / max_value
        return time_distance_matrix

    def __getitem__(self, index):
        ground_truth, valid_msk = super().__getitem__(index)  # (T, N)
        if self.flag == 'train':
            know_msk = list(
                set(random.sample(range(0, ground_truth.shape[1]), int(self.tra_s * ground_truth.shape[1]))))
            seq_y = ground_truth[:, know_msk]
            seq_x = seq_y.copy()
            unknow_msk = set(random.sample(range(0, seq_x.shape[1]), int(self.tra_m * seq_x.shape[1])))
            D_S_dynamic = self.D_S[self.know_set, :][:, self.know_set][self.tra_set, :][:, self.tra_set][know_msk, :][:, know_msk]
            inf_msk = self.inf_msk[self.know_set, :][:, self.know_set][self.tra_set, :][:, self.tra_set][know_msk, :][:, know_msk]
            seq_x[:, list(unknow_msk)] = 0
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(unknow_msk)] = 1
            seq_valid_msk = valid_msk[:, know_msk]
            # seq_msk = seq_valid_msk
        elif self.flag == 'val':
            seq_y = ground_truth
            seq_x = seq_y.copy()
            seq_x[:, list(self.val_set)] = 0
            D_S_dynamic = self.D_S[self.know_set, :][:, self.know_set]
            inf_msk = self.inf_msk[self.know_set, :][:, self.know_set]
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(self.val_set)] = 1
            seq_valid_msk = valid_msk
            # seq_msk = seq_valid_msk
        else:
            seq_y = ground_truth
            seq_x = seq_y.copy()
            seq_x[:, list(self.unknow_set)] = 0
            D_S_dynamic = self.D_S
            inf_msk = self.inf_msk
            seq_msk = np.zeros_like(seq_x)
            seq_msk[:, list(self.unknow_set)] = 1
            seq_valid_msk = valid_msk

        # dynamic sigma
        dy_sigma = np.std(D_S_dynamic[~np.isinf(D_S_dynamic)])
        # D_S_dynamic = self.trans_to_kernel(D_S_dynamic)

        seq_msk = seq_msk * seq_valid_msk
        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        D_S_dynamic = torch.from_numpy(D_S_dynamic.astype('float32')).to(self.device)
        inf_msk = torch.from_numpy(inf_msk.astype('float32')).to(self.device)
        dy_sigma = torch.from_numpy(np.array(dy_sigma).astype('float32')).to(self.device)
        sigma = torch.from_numpy(np.array(self.sigma).astype('float32')).to(self.device)

        add_data = {
            'D_S': D_S_dynamic,
            'inf_msk': inf_msk,
            'sigma': sigma,
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }


class Dataset_KITS_kernel(NormalDataset):
    def __init__(self, args, device,
                 dataset, N_test, N_val, T_test, T_val, seq_len,
                 flag='train', ignore_zero=True, fill_zero=False, test_name="",
                 scaler_method='standard', scaler_value=None,
                 train_unknow_set=None, train_val_set=None):
        self.device = device
        super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
                         test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)
        self.ratio = float(len(self.tra_set) / (len(self.unknow_set) + len(self.tra_set)))
        self.D_S = self.dist_sc.copy()
        max_to_inf = np.max(self.D_S[np.isfinite(self.D_S)]) * 10
        self.dist[np.isinf(self.dist)] = max_to_inf

    def __getitem__(self, index):
        seq_x, seq_valid_msk = super().__getitem__(index)  # (T, N)
        seq_y = seq_x.copy()
        return seq_x, seq_y, seq_valid_msk

    def collate_fn(self, batch):
        # kits输入的msk_input不同于用于criterion的msk
        seq_x, seq_y, seq_valid_msk = map(np.stack, zip(*batch)) # (B, T, N)
        seq_msk = np.zeros_like(seq_x)
        if self.flag == 'train':
            dynamic_ratio = self.ratio + 0.2 * random.random()
            cur_node_num = seq_x.shape[2]  # n1
            aug_node_num = int(cur_node_num / dynamic_ratio)  # n2
            sub_node_num = aug_node_num - cur_node_num  # n2 - n1
            seq_sub_x = np.zeros((seq_x.shape[0], seq_x.shape[1], sub_node_num))
            seq_x = np.concatenate((seq_x, seq_sub_x), axis=2)
            seq_y = np.concatenate((seq_y, seq_sub_x), axis=2)
            seq_valid_msk = np.concatenate((seq_valid_msk, seq_sub_x), axis=2)
            seq_msk = np.concatenate((seq_msk, 1-seq_sub_x), axis=2)
        elif self.flag == 'val':
            seq_msk[:, :, list(self.val_set)] = 1
            sub_node_num = 0
        else:
            seq_msk[:, :, list(self.unknow_set)] = 1
            sub_node_num = 0

        seq_x = seq_x * (1 - seq_msk)
        seq_msk_input = (1 - seq_msk) * seq_valid_msk
        seq_msk = seq_msk * seq_valid_msk

        seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
        seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
        seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
        # seq_valid_msk = torch.from_numpy(seq_valid_msk.astype(float)).to(self.device)
        adj = torch.from_numpy(self.adj.astype('float32')).to(self.device)
        seq_msk_input = torch.from_numpy(seq_msk_input.astype('float32')).to(self.device)
        dist = torch.from_numpy(self.dist.astype('float32')).to(self.device)
        sigma = torch.from_numpy(np.array(self.sigma).astype('float32')).to(self.device)
        add_data = {
            'A': adj,
            'msk_input': seq_msk_input,
            'dist': dist,
            'sigma': sigma,
            'sub_node_num': sub_node_num
        }

        return {
            'x': seq_x,
            'y': seq_y,
            'msk': seq_msk,
            'add_data': add_data
        }
