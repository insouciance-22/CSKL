import math
import torch
from torch import nn
import torch.nn.functional as F
import random


class PositiveGenerator(nn.Module):
    def __init__(self, drop_self_con=False):
        super(PositiveGenerator, self).__init__()
        self.drop_self_con = drop_self_con

    def forward(self, dist, sigma):
        alpha = random.uniform(0.9, 1.1)
        adj_matrix = torch.exp(-torch.square(dist.clone().detach() / (alpha * sigma)))
        adj_matrix[adj_matrix < 0.1] = 0
        if self.drop_self_con:
            eye_matrix = torch.eye(adj_matrix.size(0)).to(adj_matrix.device)
            adj_matrix = adj_matrix * (1 - eye_matrix)

        return adj_matrix


class NegativeGenerator(nn.Module):
    def __init__(self, drop_self_con=False):
        super(NegativeGenerator, self).__init__()
        self.drop_self_con = drop_self_con

    def forward(self, dist, sigma):
        if random.choice([True, False]):
            alpha = 1 + random.uniform(0.6, 1.)
        else:
            alpha = 1 - random.uniform(0.6, 1.)
        adj_matrix = torch.exp(-torch.square(dist.clone().detach() / (alpha * sigma)))
        adj_matrix[adj_matrix < 0.1] = 0
        if self.drop_self_con:
            eye_matrix = torch.eye(adj_matrix.size(0)).to(adj_matrix.device)
            adj_matrix = adj_matrix * (1 - eye_matrix)

        return adj_matrix


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_p, h_n, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_p)

        sc_1 = torch.squeeze(self.f_k(h_p, c), 2) # (B, N, 1)
        sc_2 = torch.squeeze(self.f_k(h_n, c), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)  #(B, 2N)

        return logits