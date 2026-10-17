from torch import nn
from layers.GraphAttention_Block import GAT


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
        self.nfeat = args.seq_len
        self.nhid = args.d
        self.nout = args.seq_len
        self.dropout = args.dropout
        self.alpha = args.alpha_gat
        self.nheads = args.heads_gat

        self.GNN1 = GAT(self.nfeat, self.nhid, self.nout, self.dropout, self.alpha, self.nheads)
        self.GNN2 = GAT(self.nfeat, self.nhid, self.nout, self.dropout, self.alpha, self.nheads)
        self.GNN3 = GAT(self.nfeat, self.nhid, self.nout, self.dropout, self.alpha, self.nheads)

    def forward(self, x, add_data):
        """
        :param x: Input data of shape (batch_size, num_timesteps, num_nodes)/(B, T, N)
        :A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)/(B, N, N)
        :A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        A = add_data['A']
        X_S = x.permute(0, 2, 1)  # to correct the input dims

        X_s1 = self.GNN1(X_S, A)
        # X_s2 = self.GNN2(X_s1, A)
        # X_s2 = self.GNN2(X_s1, A) + X_s1  # num_nodes, rank
        # X_s3 = self.GNN3(X_s2, A)

        X_res = X_s1.permute(0, 2, 1)

        return X_res
