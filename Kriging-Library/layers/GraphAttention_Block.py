import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
        Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # (B, N, T)
    def forward(self, h, adj):
        Wh = torch.einsum('bni,io->bno', h, self.W)
        # Wh = torch.matmul(h, self.W)
        E = []
        for i in range(Wh.shape[0]):
            e = self._prepare_attentional_mechanism_input(Wh[i])
            E.append(e)
        E = torch.stack(E, dim=0)
        zero_vec = -9e15 * torch.ones_like(E)
        attention = torch.where(adj > 0, E, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('bno,bjn->bjo', Wh, attention)
        # h_prime = torch.bmm(attention, Wh)
        # h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class Node_Attention(nn.Module):
    def __init__(self, in_features, hidden_dim, use_batch_mean=False):
        super(Node_Attention, self).__init__()
        self.query_proj = nn.Linear(in_features, hidden_dim)
        self.key_proj = nn.Linear(in_features, hidden_dim)
        self.use_batch_mean = use_batch_mean

    def forward(self, X):
        Q = self.query_proj(X)  # (B, N, T, hidden_dim) / (B, N, hidden_dim)
        K = self.key_proj(X)
        attention_scores = torch.einsum('bnt,bqt->bnq', Q, K) / (Q.shape[-1] ** 0.5) # (B, N, N)
        if self.use_batch_mean:
            attention_scores = torch.mean(attention_scores, dim=0)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, N) / (N, N)
        return attention_weights


class Node_Attention_2(nn.Module):
    def __init__(self, in_features, hidden_dim, use_batch_mean=False, n_heads=1):
        super(Node_Attention_2, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % n_heads == 0
        self.head_dim = hidden_dim // n_heads
        self.query_projection = nn.Linear(in_features, hidden_dim)
        self.key_projection = nn.Linear(in_features, hidden_dim)
        self.use_batch_mean = use_batch_mean

    def forward(self, X, dist=None):
        batch_size, num_nodes, _ = X.shape
        Q = self.query_proj(X)
        K = self.key_proj(X)
        Q = Q.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        # attention_scores = torch.einsum('bnt,bqt->bnq', Q, K) / (Q.shape[-1] ** 0.5) # (B, N, N)
        attention_scores = torch.einsum('bnqd,bnkd->bnqk', Q, K) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attention_scores = attention_scores.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        if self.use_batch_mean:
            attention_scores = torch.mean(attention_scores, dim=0)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, N) / (N, N)
        # λ缩放

        return attention_weights




