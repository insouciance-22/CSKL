from torch import nn
from layers.DGCN_Block import D_GCN


class Model(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.time_dimension = args.seq_len
        self.hidden_dimension = args.d
        self.order = args.k_DGCN

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order, activation='relu')

        self.time = 1

    def forward(self, x, add_data):
        """
        :param x: Input data of shape (batch_size, num_timesteps, num_nodes)/(B, T, N)
        :A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)/(B, N, N)
        :A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        A_q = add_data['A_q']
        A_h = add_data['A_h']
        X_S = x.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)

        # (B, N, T)
        X_s3 = 1
        X_res = X_s3.permute(0, 2, 1)

        return X_res
