# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.metrics import metric
# from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.losses import mse_loss_masked, mae_loss_masked, mse_loss_KITS
# import os.path
# import torch
# import torch.nn as nn
# from torch import optim
# import numpy as np
# from tqdm import tqdm
# import logging
# from torch.utils.data import Dataset
# # from torch_geometric.data import Dataset as PyG_Dataset, Data
# from utils.scalers import Scaler
# import pandas as pd
# import pickle
# from utils.tools import get_similarity, fill_zeros_with_neighbors
# import logging
# import argparse
# # import torch_geometric

#
# def get_data(dataset_name):
#     """
#     :param: dataset_name(String)
#     :return: adj[N, N], data[T, N], path(String)
#     """
#     root_path = os.path.join("dataset", dataset_name)
#     if dataset_name == 'metr_la':
#         adj = np.load(os.path.join(root_path, "adj_mat.npy"))
#         data = np.load(os.path.join(root_path, "node_values.npy"))[:, :, 0]  # (34272, 207, 2)
#     elif dataset_name == 'pems_bay':
#         with open(os.path.join(root_path, "adj_mx_bay_unix.pkl"), 'rb') as file:
#             adj_data = pickle.load(file, encoding='latin1')
#         adj = adj_data[2]
#         data = np.load(os.path.join(root_path, "node_values.npy"))
#         data = data.transpose() # (52116, 325)
#     else:
#         raise ValueError("Dataset No Found!")
#
#     return adj, data, root_path
#
#
# class NormalDataset(Dataset):
#     def __init__(self, args, dataset, N_test, N_val, T_test, T_val, seq_len, flag='train', ignore_zero=True,
#                  fill_zero=False, test_name="", scaler_method='standard', scaler_value=None,
#                  train_unknow_set=None, train_val_set=None):
#         self.dataset = dataset
#         self.N_test = N_test
#         self.N_val = N_val
#         self.T_test = T_test
#         self.T_val = T_val
#         self.seq_len = seq_len
#         self.flag = flag
#         self.ignore_zero = ignore_zero
#         self.fill_zero = fill_zero
#         self.scaler_method = scaler_method
#         self.scaler_value = scaler_value
#         self.time_stamp = args.time_stamp
#         self.model = args.model
#         self.save_path = args.save_path
#         self.test_name = test_name
#         self.adj_sc, self.data_sc, self.root_path = get_data(dataset) # (N, N), (T, N)
#         # Preprocess data
#         mask = ~np.isnan(self.data_sc)
#         if ignore_zero:
#             mask = mask * (self.data_sc != 0.).astype(float)
#             if fill_zero: # 这里还可以改进，实现更合理的补缺
#                 if fill_zero == 'knn-mean':
#                     self.data_sc = fill_zeros_with_neighbors(self.data_sc, self.adj_sc, k=10)
#                     df = pd.DataFrame(self.data_sc)
#                     df = df.replace(0, np.nan).fillna(method='ffill', axis=0)
#                     self.data_sc = df.values.astype(float)
#                 elif fill_zero == 'ffill':
#                     df = pd.DataFrame(self.data_sc)
#                     df = df.replace(to_replace=0., method='ffill')
#                     self.data_sc = df.values.astype(float)
#         # Preprocess adj
#         if self.model == 'KITS':
#             self.adj_sc = get_similarity(self.adj_sc)
#         # Nodes split
#         test_nodes_num = int(N_test * self.data_sc.shape[1])
#         ave_nodes_num = self.data_sc.shape[1] - test_nodes_num
#         val_nodes_num = int(ave_nodes_num * N_val)
#         test_node_path = os.path.join(self.root_path, "test_node_index")
#         if not os.path.exists(test_node_path):
#             os.makedirs(test_node_path)
#         if self.test_name == "":
#             if flag != 'train':
#                 raise ValueError("Test_Node File No Found!")
#             logging.info("Not use history test_nodes. Randomly divide nodes for testing!")
#             self.test_name = self.time_stamp + "_%" + str(int(N_test * 100)) + ".npz"
#             test_node = np.random.choice(list(range(0, self.data_sc.shape[1])), int(test_nodes_num), replace=False)
#             val_node = np.random.choice(list(range(0, ave_nodes_num)), int(val_nodes_num), replace=False)
#             unknow_set = set(test_node)
#             val_set = set(val_node)
#             logging.info("The test_nodes file's name is %s", os.path.join(test_node_path, self.test_name))
#             np.savez(os.path.join(test_node_path, self.test_name), test_node=test_node, val_node=val_node)
#             np.savez(os.path.join(self.save_path, self.test_name), test_node=test_node, val_node=val_node)
#         else:
#             assert int(self.test_name[-6:-4]) * 0.01 == N_test
#             if train_unknow_set:
#                 unknow_set = train_unknow_set
#                 val_set = train_val_set
#             else:
#                 test_node = np.load(os.path.join(test_node_path, self.test_name))['test_node']
#                 unknow_set = set(test_node)
#                 val_node = np.load(os.path.join(test_node_path, self.test_name))['val_node']
#                 val_set = set(val_node)
#                 logging.info("Testing node is loaded")
#                 np.savez(os.path.join(self.save_path, self.test_name), test_node=test_node, val_node=val_node)
#         logging.info("%s, test_node: %s, vali_node: %s", flag, list(unknow_set)[:10], list(val_set)[:10])
#
#         full_set = set(range(0, self.data_sc.shape[1]))
#         know_set = full_set - unknow_set
#         self.know_set = know_set
#         ave_set = set(range(0, ave_nodes_num))
#         tra_set = ave_set - val_set
#         self.unknow_set = unknow_set
#         self.val_set = val_set
#         self.tra_set = tra_set
#         # Time split
#         split_line1 = self.data_sc.shape[0] - int(self.data_sc.shape[0] * self.T_test)
#         self.split_line1 = split_line1
#         ave_data = self.data_sc[:split_line1, list(know_set)]
#         ave_mask = mask[:split_line1, list(know_set)]
#         test_data = self.data_sc[split_line1:, :]
#         test_mask = mask[split_line1:, :]
#         split_line2 = split_line1 - int(split_line1 * self.T_val)
#         self.split_line2 = split_line2
#         val_data = ave_data[split_line2:, :]
#         val_mask = ave_mask[split_line2:, :]
#         train_data = ave_data[:split_line2, list(tra_set)]
#         train_mask = ave_mask[: split_line2, list(tra_set)]
#         # Adjacency matrix
#         adj_val = self.adj_sc[:, list(know_set)][list(know_set), :].copy()
#         adj_tra = adj_val[:, list(tra_set)][list(tra_set), :].copy()
#         # Normalization
#         self.scaler = Scaler(scaler_method, scaler_value)
#         self.scaler.fit(train_data)
#         assert flag in ['train', 'test', 'val']
#         if self.flag == 'train':
#             self.gt = self.scaler.transform(train_data)
#             self.adj = adj_tra
#             self.valid_mask = train_mask
#         elif self.flag == 'val':
#             self.gt = self.scaler.transform(val_data)
#             self.adj = adj_val
#             self.valid_mask = val_mask
#         else:
#             self.gt = self.scaler.transform(test_data)
#             self.adj = self.adj_sc.copy()
#             self.valid_mask = test_mask
#         logging.info("%s: %s", self.flag, self.gt.shape)
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         seq_gt = self.gt[s_begin: s_end]
#         seq_valid_mask = self.valid_mask[s_begin: s_end]
#         return seq_gt, seq_valid_mask
#
#     def __len__(self):
#         return len(self.gt) - self.seq_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#     def get_set_index(self):
#         # 根据训练集、验证集在know_set中的索引推导出在原始数据中的索引，以便直接从所有节点中找到训练和验证集节点
#         indices = np.array(range(self.data_sc.shape[1]))
#         indices_ave = indices[list(self.know_set)]
#         indices_val = set(indices_ave[list(self.val_set)])
#         indices_tra = set(indices_ave[list(self.tra_set)])
#         # a = self.data_sc[:, list(indices_tra)]
#         # b = self.data_sc[:, list(self.know_set)][:, list(self.tra_set)]
#         # c = np.array_equal(a, b)
#         return indices_tra, indices_val

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parser = argparse.ArgumentParser(description='Kriging')
    # args = parser.parse_args()
    # args.time_stamp = ''
    # args.model = 'IGNNK'
    # dataset = 'metr_la'
    # args.save_path = ""
    # test_name = "2024-07-27_19-44-36_%50.npz"
    # dataset = NormalDataset(args, dataset,
    #                       0.2, 0.1, 0.3, 0.1, 16, 'train', test_name="")
    # adj_sc, data_sc, root_path = get_data('pems_bay')  # (N, N), (T, N)
    # a = np.array([[1, 2], [3, 4]])
    # print(a)
    # print(np.sum(np.isnan(data_sc)))
    # print(2222)
    # nan_mask = np.isnan(data_sc)
    # not_nan_mask = ~nan_mask
    # print(3333)
    # # Preprocess data
    # mask = ~np.isnan(data_sc)
    # print(mask)
    # print(1111)
    import torch
    import numpy as np

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(np.__version__)
    # print(torch_geometric.__version__)

    import pandas as pd
