import random

from torch_geometric.data import Dataset, Data
from torch_geometric.loader.dataloader import DataLoader
import torch


class CustomGraphDataset(Dataset):
    def __init__(self):
        super(CustomGraphDataset, self).__init__()
        self.x = torch.randn(20, 4, 5)
        self.add = torch.randn(20, 3, 3)

    def len(self):
        return self.x.shape[0]

    def get(self, idx):
        l = [0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0]
        a = l[idx]
        if a == 0:
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            edge_weight = torch.tensor([1, 2, 3], dtype=torch.long)
        elif a == 1:
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            edge_weight = torch.tensor([1, 2], dtype=torch.long)
        x = self.x[idx]

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=x, add=self.add[idx], edge_index2=edge_index, edge_weight2=edge_weight)


dataset = CustomGraphDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=False)

data_list = []
for j, data in enumerate(loader):
    print(data.add.shape)
    print(data.x.shape)
    print(data.edge_index.shape)
    print(data.edge_index2.shape)
    print(data.edge_weight.shape)
    print(data.edge_weight2.shape)
    print("--")