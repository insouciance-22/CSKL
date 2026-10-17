# import os.path
# from torch_geometric.data import Dataset as PyG_Dataset, Data
# import numpy as np
# from utils.scalers import Scaler
# import pandas as pd
# from utils.tools import get_similarity, fill_zeros_knn
# from data_provider.data_set import get_data
# import logging
#
#
# class GNNDataset(PyG_Dataset):
#     def __init__(self, args, dataset, N_test, N_val, T_test, T_val, seq_len, flag='train', ignore_zero=True,
#                  fill_zero=False, test_name="", scaler_method='standard', scaler_value=None,
#                  train_unknow_set=None, train_val_set=None):
#         super(GNNDataset, self).__init__()
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
#         logging.info("data shape:%s", self.data_sc.shape)
#         # Preprocess data
#         mask = np.isnan(self.data_sc)
#         mask = ~mask
#         if ignore_zero:
#             mask = mask * (self.data_sc != 0.).astype(float)
#             # non_zero = (self.data_sc != 0.)  # 检查非零元素
#             # non_zero = non_zero.astype(float)  # 转换类型
#             # mask = mask * non_zero  # 应用掩码
#             if fill_zero:
#                 if fill_zero == 'knn':
#                     self.data_sc = fill_zeros_knn(self.data_sc, self.adj_sc, k=10)
#                     df = pd.DataFrame(self.data_sc)
#                     df = df.replace(0, np.nan).fillna(method='ffill', axis=0)
#                     df = df.fillna(method='bfill', axis=0)
#                     self.data_sc = df.values.astype(float)
#                 elif fill_zero == 'ffill':
#                     df = pd.DataFrame(self.data_sc)
#                     df = df.replace(0, np.nan).fillna(method='ffill', axis=0)
#                     df = df.fillna(method='bfill', axis=0)
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
#             self.test_name = self.time_stamp + "_" + str(int(N_test * 100)) + "%.npz"
#             test_node = np.random.choice(list(range(0, self.data_sc.shape[1])), int(test_nodes_num), replace=False)
#             val_node = np.random.choice(list(range(0, ave_nodes_num)), int(val_nodes_num), replace=False)
#             unknow_set = set(test_node)
#             val_set = set(val_node)
#             logging.info("The test_nodes file's name is %s", os.path.join(test_node_path, self.test_name))
#             np.savez(os.path.join(test_node_path, self.test_name), test_node=test_node, val_node=val_node)
#             np.savez(os.path.join(self.save_path, self.test_name), test_node=test_node, val_node=val_node)
#         else:
#             assert int(self.test_name[-7:-5]) == int(N_test * 100)
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
#         full_set = set(range(0, self.data_sc.shape[1]))
#         know_set = full_set - set(unknow_set)
#         ave_set = set(range(0, ave_nodes_num))
#         tra_set = ave_set - set(val_set)
#         # sort
#         know_set = sorted(know_set)
#         unknow_set = sorted(unknow_set)
#         val_set = sorted(val_set)
#         tra_set = sorted(tra_set)
#         self.know_set = know_set
#         self.unknow_set = unknow_set
#         self.val_set = val_set
#         self.tra_set = tra_set
#         logging.info("%s dataset initialized, test_node: %s...%s, vali_node: %s...%s", flag, unknow_set[:5],
#                      unknow_set[-5:], val_set[:5], val_set[-5:])
#         # Time split
#         test_len = int(self.data_sc.shape[0] * self.T_test)
#         val_len = int(self.data_sc.shape[0] * self.T_val)
#         split_line1 = self.data_sc.shape[0] - test_len
#         ave_data = self.data_sc[:split_line1, know_set]
#         ave_mask = mask[:split_line1, know_set]
#         test_data = self.data_sc[split_line1:, :]
#         test_mask = mask[split_line1:, :]
#         split_line2 = split_line1 - val_len
#         self.split_line1 = split_line1
#         self.split_line2 = split_line2
#         val_data = ave_data[split_line2:, :]
#         val_mask = ave_mask[split_line2:, :]
#         train_data = ave_data[:split_line2, tra_set]
#         train_mask = ave_mask[: split_line2, tra_set]
#         # Adjacency matrix
#         adj_val = self.adj_sc[:, know_set][know_set, :].copy()
#         adj_tra = adj_val[:, tra_set][tra_set, :].copy()
#         # Normalization
#         self.scaler = Scaler(scaler_method, scaler_value)
#         self.scaler.fit(train_data)
#         assert flag in ['train', 'test', 'val']
#         if self.flag == 'train':
#             self.gt = train_data
#             self.adj = adj_tra
#             self.valid_mask = train_mask
#             logging.info("num_train_node: %s", len(tra_set))
#         elif self.flag == 'val':
#             self.gt = val_data
#             self.adj = adj_val
#             self.valid_mask = val_mask
#             logging.info("num_val_node: %s", len(val_set))
#         else:
#             self.gt = test_data
#             self.adj = self.adj_sc.copy()
#             self.valid_mask = test_mask
#             logging.info("num_test_node: %s", len(unknow_set))
#         if args.model != 'KNN':
#             self.gt = self.scaler.transform(self.gt)
#             logging.info("%s shape: %s, scale_method: %s", self.flag, self.gt.shape, self.scaler.method)
#         else:
#             logging.info("%s shape: %s, no scale_method", self.flag, self.gt.shape)
#
#     def get(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         seq_gt = self.gt[:, s_begin: s_end]
#         seq_valid_mask = self.valid_mask[:, s_begin: s_end]
#         return Data(x=seq_gt, seq_valid_mask=seq_valid_mask)
#
#     def len(self):
#         return self.gt.shape[-1] - self.seq_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#     def get_set_index(self):
#         # 根据训练集、验证集在know_set中的索引推导出在原始数据中的索引，以便直接从所有节点中找到训练和验证集节点
#         indices = np.array(range(self.data_sc.shape[1]))
#         indices_ave = indices[self.know_set]
#         indices_val = set(indices_ave[self.val_set])
#         indices_tra = set(indices_ave[self.tra_set])
#         # a = self.data_sc[:, list(indices_tra)]
#         # b = self.data_sc[:, list(self.know_set)][:, list(self.tra_set)]
#         # c = np.array_equal(a, b)
#         return indices_tra, indices_val
#
#
# class Dataset_KCP(GNNDataset):
#     def __init__(self, args, device,
#                 tra_aug, rep_num, pre_training,
#                 dataset, N_test, N_val, T_test, T_val, seq_len,
#                 flag='train', ignore_zero=True, fill_zero=False, test_name="",
#                 scaler_method='standard', scaler_value=50,
#                 train_unknow_set=None, train_val_set=None):
#         self.tra_aug = tra_aug
#         self.rep_num = rep_num
#         self.pre_training = pre_training
#         self.device = device
#         super().__init__(args, dataset, N_test, N_val, T_test, T_val, seq_len, flag, ignore_zero, fill_zero,
#                          test_name, scaler_method, scaler_value, train_unknow_set, train_val_set)
#         # 自监督
#         if self.flag == 'train':
#             self.gt = np.tile(self.gt, [1, self.rep_num])
#             self.valid_mask = np.tile(self.valid_mask, [1, self.rep_num])
#
#     def len(self):
#         return super(Dataset_KCP, self).len()
#
#     def get(self, index):
#         data_sc = super().get(index)
#         ground_truth, valid_msk = data_sc.x, data_sc.seq_valid_mask
#         # (T, N)
#         if self.flag == 'train':
#             seq_y = ground_truth
#             seq_x = seq_y.copy()
#             num_aug = int(seq_x.shape[0] * self.tra_aug)
#             aug_idx = np.random.choice(np.arange(seq_x.shape[0]), num_aug, replace=False)
#             aug_idx_sc = np.array(list(self.tra_set))[aug_idx]
#             adj = self.adj
#             seq_x[list(aug_idx), :] = 0
#             seq_msk = np.zeros_like(seq_x)
#             seq_msk[list(aug_idx), :] = 1
#             seq_valid_msk = valid_msk
#         elif self.flag == 'val':
#             seq_y = ground_truth
#             seq_x = seq_y.copy()
#             seq_x[list(self.val_set), :] = 0
#             adj = self.adj
#             seq_msk = np.zeros_like(seq_x)
#             seq_msk[list(self.val_set), :] = 1
#             seq_valid_msk = valid_msk
#             aug_idx = np.array(list(self.val_set))
#             aug_idx_sc = np.array(list(self.know_set))[aug_idx]
#         else:
#             seq_y = ground_truth
#             seq_x = seq_y.copy()
#             seq_x[list(self.unknow_set), :] = 0
#             adj = self.adj
#             seq_msk = np.zeros_like(seq_x)
#             seq_msk[list(self.unknow_set), :] = 1
#             seq_valid_msk = valid_msk
#             aug_idx = np.array(list(self.unknow_set))
#             aug_idx_sc = np.array(list(self.unknow_set))
#
#         seq_msk = seq_msk * seq_valid_msk
#         # [N, T]
#         seq_x = torch.from_numpy(seq_x.astype('float32')).to(self.device)
#         seq_y = torch.from_numpy(seq_y.astype('float32')).to(self.device)
#         seq_msk = torch.from_numpy(seq_msk.astype('float32')).to(self.device)
#         adj = torch.from_numpy(adj.astype('float32')).to(self.device)
#         # [2, E], [E,]
#         edge_index = adj.nonzero().t().contiguous()y
#         edge_weight = adj[edge_index[0], edge_index[1]]
#         adj_t = adj.t()
#         edge_index2 = adj_t.nonzero().t().contiguous()
#         edge_weight2 = adj_t[edge_index2[0], edge_index2[1]]
#
#         data = Data(x=seq_x, y=seq_y, msk=seq_msk, edge_index=edge_index, edge_weight=edge_weight,
#                     edge_index2=edge_index2, edge_weight2=edge_weight2, node_id=aug_idx_sc, idx_of_node=seq_x.shape[0],
#                         map_id=aug_idx)
#         ## topo aug
#         if self.pre_training:
#             nodes_centrality = (adj[aug_idx, :] > 0).sum(1)
#             mean_c = (adj > 0).sum(1).float().mean().int()
#             p = (nodes_centrality - mean_c) / torch.max((adj > 0).sum(1))
#             p[p < 0] = 0.
#             adj_corrupt = adj.clone().fill_diagonal_(0)
#             for i, item in enumerate(aug_idx):
#                 if p[i] == 0:
#                     continue
#                 num_smps = (nodes_centrality - mean_c)[i]
#                 smps = torch.arange(adj.shape[0], device=self.device)[adj[item, :] > 0]
#                 unif = torch.ones(smps.shape[0], device=self.device)
#                 idx = unif.multinomial(num_smps)
#                 smp_idx = torch.bernoulli(p[i].repeat(smps[idx].shape[0])).bool()
#                 adj_corrupt[item, smps[idx][smp_idx]] = 0
#             adj_corrupt = adj_corrupt.fill_diagonal_(1)
#             crpt_edge_index = adj_corrupt.nonzero().t().contiguous()
#             crpt_edge_weight = adj_corrupt[crpt_edge_index[0], crpt_edge_index[1]]
#             adj_corrupt_t = adj_corrupt.t()
#             crpt_edge_index2 = adj_corrupt_t.nonzero().t().contiguous()
#             crpt_edge_weight2 = adj_corrupt_t[crpt_edge_index2[0], crpt_edge_index2[1]]
#             data.crpt_edge_index = crpt_edge_index
#             data.crpt_edge_weight = crpt_edge_weight
#             data.crpt_edge_index2 = crpt_edge_index2
#             data.crpt_edge_weight2 = crpt_edge_weight2
#
#         # KCP模型接收的数据形状(B*N, T)
#         return data