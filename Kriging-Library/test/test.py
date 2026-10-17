import argparse
import os
import torch
from torch.utils.data import DataLoader
# from torch_geometric.loader.dataloader import DataLoader as PyG_DataLoader
from tqdm import tqdm
from data_provider.data_loader import Dataset_KITS, Dataset_IGNNK, Dataset_INCREASE, Dataset_IGNNK_self_A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Kriging')
args = parser.parse_args()
args.time_stamp = ''
args.model = 'IGNNK'
dataset = 'metr_la'
args.save_path = ""
test_name = "2024-07-27_19-44-36_%50.npz"
# dataset = Dataset_KCP(args, device,
#                       0.2, 3, True,
#                       'metr_la', 0.5, 0.1, 0.3, 0.2, 16, 'train', test_name="")
# data_loader = PyG_DataLoader(dataset, batch_size=4, shuffle=False)
# for i, data in enumerate(tqdm(data_loader, desc='Training')):
#     bsxn = data.x.size(0)
#     bs = len(data.node_id)
#     num_nodes = bsxn // bs
#     print()

# dataset = Dataset_KCP(args, device,
#                       3, 0.2, 3, True,
#                       'metr_la', 0.5, 0.1, 0.3, 0.2, 16, 'train', test_name="")
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=4,
#     shuffle=True,
#     drop_last=True)
# a = len(data_loader)
# for i, data in enumerate(tqdm(data_loader, desc='Training')):
#     a = data
#     print()

dataset = Dataset_IGNNK(args, device, 0.9, 0.5, 'metr_la', 0.5, 0.1, 0.3, 0.2, 16, 'val', test_name=test_name)
data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    collate_fn=None
    )
for i, data in enumerate(tqdm(data_loader, desc='Training')):
    a = data
    print(a['add_data'])

# import random
# import numpy as np
# fix_seed = 2024
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)
#
# for i in range(6):
#     unknow_msk = set(random.sample(range(0, 10), 5))
#     print(list(unknow_msk))

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设置随机种子以保证结果可复现
# # np.random.seed(42)
#
# # 定义日期范围
# start_date = '2023-01-01'
# end_date = '2023-01-08'
# date_range = pd.date_range(start=start_date, end=end_date, freq='H')  # 每小时的数据
#
# # 生成基础趋势
# base_trend = np.sin(2 * np.pi * date_range.hour / 24)  # 日周期
# weekly_trend = np.sin(2 * np.pi * (date_range.dayofweek + date_range.hour / 24) / 7)  # 周周期
# trend = base_trend + weekly_trend
#
# # 添加随机噪声
# noise = np.random.normal(scale=0.5, size=len(date_range))
#
# # 生成最终的时间序列数据
# traffic_data = trend + noise
#
# # 创建 DataFrame
# df = pd.DataFrame({'Traffic': traffic_data}, index=date_range)
#
# # 绘制时间序列
# plt.figure(figsize=(14, 7))
# plt.plot(df.index, df['Traffic'], label='', color='deeppink', linewidth=2.5)
# # plt.title('Simulated Traffic Data with Daily and Weekly Cycles')
# plt.xlabel('Date')
# plt.ylabel('Traffic')
# plt.legend()
# # plt.grid(True)
# plt.show()

# 如果需要保存数据到文件
# df.to_csv('simulated_traffic_data.csv')

# import logging
#
# # 配置 logger
# logging.basicConfig(level=logging.INFO,
#                     # format='%(asctime)s - %(levelname)s - %(message)s',
#                     format='%(message)s',
#                     filename='training_log.log',   # 日志文件
#                     filemode='w')                 # 覆盖旧文件
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # formatter = logging.Formatter('%(message)s')
# # console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
# config = {
#     'model_name': 'ResNet50',
#     'input_size': (224, 224, 3),
#     'num_classes': 10,
#     'batch_size': 32,
#     'num_epochs': 10,
#     'learning_rate': 0.001,
#     'optimizer': 'Adam'
# }
#
# logging.info("Model Configuration:")
# logging.info("The test_nodes file's name is %s", os.path.join('test_node_path', 'test_name'))
# import numpy as np
# x = np.random.randint(10, 50, (10, 10))
# full_set = set(range(0, 10))
# unknow_set = {0, 7, 3}
# know_set = full_set - unknow_set
# # print(x[list(know_set)])
# ave_x = x[list(know_set)]
# ave_set = set(range(0, 7))
# val_set = {1, 5, 4}
# tra_set = ave_set - val_set
# print(ave_x[list(tra_set)])
# print()
# print(x[list(know_set)][list(tra_set)])
# print()
# a = np.array(list(know_set))[list(tra_set)]
# print(x[a])

# import numpy as np
# x = np.random.randint(10, 50, (5, 5))
# x1 = x.copy()
# print(x)
# unknow_set = {0, 1, 3}
# a = x[list(unknow_set), :][:, list(unknow_set)]
# a += 1
# c = np.ix_(list(unknow_set), list(unknow_set))
# x[np.ix_(list(unknow_set), list(unknow_set))] = a
# print(x)
#
# a -= 1
# x[np.ix_(list(unknow_set), list(unknow_set))] = a
# print(x)
# print(np.array_equal(x1, x))