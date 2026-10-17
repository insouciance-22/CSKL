import os.path
from torch.utils.data import Dataset
import numpy as np
from utils.scalers import Scaler
import pandas as pd
import pickle
from utils.tools import fill_zeros_knn, fill_nan_knn, plot_time_series
from utils.dataset_tools import get_dist
from utils.kernels import *
import logging
import h5py


def get_data(dataset_name):
    """
    :param: dataset_name(String)
    :return: adj[N, N], data[T, N], path(String)
    """
    root_path = os.path.join("dataset", dataset_name)
    if dataset_name == 'metr_la':
        # adj = np.load(os.path.join(root_path, "adj_mat.npy"))
        data = np.load(os.path.join(root_path, "node_values.npy"))[:, :, 0]  # (34272, 207, 2)
        dist = np.load(os.path.join(root_path, "metr_la_dist.npy"))  # 含有nan值
        adj, sigma = rfb_kernel(dist, l=1.)
        # adj = rfb_kernel_prime(dist)
        # adj = matern_kernel(dist, nu=2.5)
        # adj = rational_quadratic_kernel(dist, alpha=2., l=1., theta=0.1)
    elif dataset_name == 'pems_bay':
        with open(os.path.join(root_path, "adj_mx_bay_unix.pkl"), 'rb') as file:
            adj_data = pickle.load(file, encoding='latin1')
        # adj_ = adj_data[2]
        data = np.load(os.path.join(root_path, "node_values.npy")).transpose()  # (52116, 325)
        id_map = adj_data[1]
        df = pd.read_csv(os.path.join(root_path, "distances_bay_2017.csv"), header=None, names=['start', 'end', 'distance'])
        dist = np.inf * np.ones((325, 325))
        for _, row in df.iterrows():
            start = id_map[str(int(row['start']))]
            end = id_map[str(int(row['end']))]
            distance = row['distance']
            dist[start, end] = distance
        np.fill_diagonal(dist, 0)
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1)
    elif dataset_name == 'AQI_36':
        path = os.path.join(root_path, "small36.h5")
        # eval_mask = pd.DataFrame(pd.read_hdf(path, 'eval_mask'))
        # with pd.HDFStore(path) as store:
        #     available_keys = store.keys()
        data = pd.DataFrame(pd.read_hdf(path, 'pm25')).values
        stations = pd.DataFrame(pd.read_hdf(path, 'stations')).reset_index(drop=True)
        dist = get_dist(stations, 'longitude', 'latitude')
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1) #(8759, 36)
    elif dataset_name == 'AQI': #
        path = os.path.join(root_path, "full437.h5")
        data = pd.DataFrame(pd.read_hdf(path, 'pm25')).values
        stations = pd.DataFrame(pd.read_hdf(path, 'stations')).reset_index(drop=True)
        dist = get_dist(stations, 'longitude', 'latitude')
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1, decisive_index=36) #(8760, 437)
    elif dataset_name == 'PEMS03':
        data = np.load(os.path.join(root_path, "PEMS03.npz"))['data'][:, :, 0]
        dist = np.load(os.path.join(root_path, "adj_dist.npy"))
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1)
    elif dataset_name == 'PEMS04':
        data = np.load(os.path.join(root_path, "PEMS04.npz"))['data'][:, :, 0]
        dist = np.load(os.path.join(root_path, "adj_dist.npy"))
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1)
    elif dataset_name == 'PEMS07':
        data = np.load(os.path.join(root_path, "PEMS07.npz"))['data'][:, :, 0]
        # dist = np.load(os.path.join(root_path, "adj_dist.npy"))
        dist = np.load(os.path.join(root_path, "dist0.npy"))
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1)
    elif dataset_name == 'PEMS08':
        data = np.load(os.path.join(root_path, "PEMS08.npz"))['data'][:, :, 0]
        dist = np.load(os.path.join(root_path, "adj_dist.npy"))
        adj, sigma = rfb_kernel(dist, l=1., theta=0.1)
    else:
        raise ValueError("Dataset No Found!")

    return adj, data, dist, root_path, sigma


class NormalDataset(Dataset):
    def __init__(self, args, dataset, N_test, N_val, T_test, T_val, seq_len, flag='train', ignore_zero=True,
                 fill_zero=False, test_name="", scaler_method='standard', scaler_value=None,
                 train_unknow_set=None, train_val_set=None):
        self.dataset = dataset
        self.N_test = N_test
        self.N_val = N_val
        self.T_test = T_test
        self.T_val = T_val
        self.seq_len = seq_len
        self.flag = flag
        self.ignore_zero = ignore_zero
        self.fill_zero = fill_zero
        self.scaler_method = scaler_method
        self.scaler_value = scaler_value
        self.time_stamp = args.time_stamp
        self.model = args.model
        self.save_path = args.save_path
        self.test_name = test_name
        self.adj_sc, self.data_sc, self.dist_sc, self.root_path, self.sigma = get_data(dataset) # (N, N), (T, N)
        logging.info("data shape:%s", self.data_sc.shape)
        # Preprocess data
        mask = ~np.isnan(self.data_sc)
        if ignore_zero: # 数据中的0值是否为无效值（若为无效值则需要用其他值补充）
            self.data_sc[~mask] = 0
            mask = mask * (self.data_sc != 0.).astype(float)
            if fill_zero: # 对0进行补充
                if fill_zero == 'knn':
                    self.data_sc = fill_zeros_knn(self.data_sc, self.adj_sc, k=10)
                df = pd.DataFrame(self.data_sc)
                df = df.replace(0, np.nan).fillna(method='ffill', axis=0)
                df = df.fillna(method='bfill', axis=0)
                self.data_sc = df.values.astype(float)
        else:
            if fill_zero: # 对nan进行补充
                if fill_zero == 'knn':
                    self.data_sc = fill_nan_knn(self.data_sc, self.adj_sc, k=10)
                df = pd.DataFrame(self.data_sc)
                df = df.fillna(method='ffill', axis=0)
                df = df.fillna(method='bfill', axis=0)
                self.data_sc = df.values.astype(float)
        # Preprocess adj
        if self.model == 'KITS' or self.model == 'KITS_kernel':
            np.fill_diagonal(self.adj_sc, 0.)
        # Nodes split
        test_nodes_num = int(N_test * self.data_sc.shape[1])
        ave_nodes_num = self.data_sc.shape[1] - test_nodes_num
        val_nodes_num = int(ave_nodes_num * N_val)
        test_node_path = os.path.join(self.root_path, "test_node_index")
        if not os.path.exists(test_node_path):
            os.makedirs(test_node_path)
        if self.test_name == "":
            if flag != 'train':
                raise ValueError("Test_Node File No Found!")
            self.test_name = self.time_stamp + "_" + str(int(N_test * 100)) + "%.npz"
            test_node = np.random.choice(list(range(0, self.data_sc.shape[1])), int(test_nodes_num), replace=False)
            val_node = np.random.choice(list(range(0, ave_nodes_num)), int(val_nodes_num), replace=False)
            unknow_set = set(test_node)
            val_set = set(val_node)
            logging.info("The test_nodes file's name is %s", os.path.join(test_node_path, self.test_name))
            np.savez(os.path.join(test_node_path, self.test_name), test_node=test_node, val_node=val_node)
            np.savez(os.path.join(self.save_path, self.test_name), test_node=test_node, val_node=val_node)
        else:
            assert int(self.test_name[-7:-5]) == int(N_test * 100)
            if train_unknow_set:
                unknow_set = train_unknow_set
                val_set = train_val_set
            else:
                test_node = np.load(os.path.join(test_node_path, self.test_name))['test_node']
                unknow_set = set(test_node)
                val_node = np.load(os.path.join(test_node_path, self.test_name))['val_node']
                val_set = set(val_node)
                logging.info("Testing node is loaded")
                np.savez(os.path.join(self.save_path, self.test_name), test_node=test_node, val_node=val_node)
        full_set = set(range(0, self.data_sc.shape[1]))
        know_set = full_set - set(unknow_set)
        ave_set = set(range(0, ave_nodes_num))
        tra_set = ave_set - set(val_set)
        # sort
        know_set = sorted(know_set)
        unknow_set = sorted(unknow_set)
        val_set = sorted(val_set)
        tra_set = sorted(tra_set)
        self.know_set = know_set
        self.unknow_set = unknow_set
        self.val_set = val_set
        self.tra_set = tra_set
        logging.info("%s dataset initialized, test_node: %s...%s, vali_node: %s...%s", flag, unknow_set[:5], unknow_set[-5:], val_set[:5], val_set[-5:])
        # Time split
        test_len = int(self.data_sc.shape[0] * self.T_test)
        val_len = int(self.data_sc.shape[0] * self.T_val)
        split_line1 = self.data_sc.shape[0] - test_len
        ave_data = self.data_sc[:split_line1, know_set]
        ave_mask = mask[:split_line1, know_set]
        test_data = self.data_sc[split_line1:, :]
        test_mask = mask[split_line1:, :]
        split_line2 = split_line1 - val_len
        self.split_line1 = split_line1
        self.split_line2 = split_line2
        val_data = ave_data[split_line2:, :]
        val_mask = ave_mask[split_line2:, :]
        train_data = ave_data[:split_line2, tra_set]
        train_mask = ave_mask[: split_line2, tra_set]
        # Adjacency matrix
        adj_val = self.adj_sc[:, know_set][know_set, :].copy()
        adj_tra = adj_val[:, tra_set][tra_set, :].copy()
        dist_val = self.dist_sc[:, know_set][know_set, :].copy()
        dist_tra = dist_val[:, tra_set][tra_set, :].copy()
        # Normalization
        self.scaler = Scaler(scaler_method, scaler_value)
        self.scaler.fit(train_data)
        assert flag in ['train', 'test', 'val']
        if self.flag == 'train':
            self.gt = train_data
            self.adj = adj_tra
            self.valid_mask = train_mask
            self.dist = dist_tra
            logging.info("num_train_node: %s", len(tra_set))
        elif self.flag == 'val':
            self.gt = val_data
            self.adj = adj_val
            self.valid_mask = val_mask
            self.dist = dist_val
            logging.info("num_val_node: %s", len(val_set))
        else:
            self.gt = test_data
            self.adj = self.adj_sc.copy()
            self.valid_mask = test_mask
            self.dist = self.dist_sc.copy()
            logging.info("num_test_node: %s", len(unknow_set))
        if args.model not in {'KNN', 'IDW'}:
            self.gt = self.scaler.transform(self.gt)
            logging.info("%s shape: %s, scale_method: %s", self.flag, self.gt.shape, self.scaler.method)
        else:
            logging.info("%s shape: %s, no scale_method", self.flag, self.gt.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_gt = self.gt[s_begin: s_end]
        seq_valid_mask = self.valid_mask[s_begin: s_end]
        return seq_gt, seq_valid_mask

    def __len__(self):
        return len(self.gt) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_set_index(self):
        # 根据训练集、验证集在know_set中的索引推导出在原始数据中的索引，以便直接从所有节点中找到训练和验证集节点
        indices = np.array(range(self.data_sc.shape[1]))
        indices_ave = indices[self.know_set]
        indices_val = set(indices_ave[self.val_set])
        indices_tra = set(indices_ave[self.tra_set])
        # a = self.data_sc[:, list(indices_tra)]
        # b = self.data_sc[:, list(self.know_set)][:, list(self.tra_set)]
        # c = np.array_equal(a, b)
        return indices_tra, indices_val


if __name__ == '__main__':
    root_path = os.path.join("../dataset", "AQI_36")
    # file = h5py.File(os.path.join(root_path, "small36.h5"), 'r')
    # print("Keys in the HDF5 file:", list(file.keys()))
    # eval_mask = pd.DataFrame(pd.read_hdf(path, 'eval_mask'))
    # df = pd.DataFrame(pd.read_hdf(path, 'pm25'))
    # stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
    path = os.path.join(root_path, "small36.h5")
    data = pd.DataFrame(pd.read_hdf(path, 'pm25')).values
    plot_time_series(data[40:100, 0:5], xlabel=None, ylabel=None)
    # print()
    # root_path = os.path.join("../dataset", "metr_la")
    # data = np.load(os.path.join(root_path, "node_values.npy"))[:, :, 0]
    # plot_time_series(data[40:100, 0:5], xlabel=None, ylabel=None)