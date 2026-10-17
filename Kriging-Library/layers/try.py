import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, in_f, hid_f, out_f, dropout, alpha, nhead):
        super().__init__()
        self.attentions = nn.ModuleList([GraphAttentionLayer(in_f, hid_f, True, dropout, alpha) for _ in range(nhead)])
        self.dropout = dropout
        self.out_attn = GraphAttentionLayer(hid_f*nhead, out_f, False, dropout, alpha)

    def forward(self, x, adj):
        h = F.dropout(x, self.dropout, training=self.training)
        h = torch.cat([attn(h, adj) for attn in self.attentions], dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.out_attn(h, adj)
        return h


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=False, dropout=0.0, alpha=0.01):
        super().__init__()
        self.leakeyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.out_f = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.concat = concat
        with torch.no_grad:
            nn.init.xavier_uniform_(self.W, gain=1.414)
            nn.init.xavier_uniform_(self.a, gain=1.414)

    def additive_attn(self, h):
        e1 = torch.matmul(h, self.a[:self.out_f])
        e2 = torch.matmul(h, self.a[self.out_f:])
        e = e1 + e2.T
        return self.leakeyrelu(e)

    def forward(self, x, adj):
        if len(adj.shape) != 3:
            adj = adj.unsqueeze(0).repeat(x.shape[0], 1, 1)
        h = torch.einsum('bnf,fo->bno', x, self.W)
        E = []
        for i in range(h.shape[0]):
            E.append(self.additive_attn(h))
        E = torch.stack(E)
        zero_vec = -9e15 * torch.ones_like(E)
        attention = torch.where(adj>0, E, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = torch.einsum('bno,bmn->bmo', h, attention)
        if self.concat:
            out = F.elu(out)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, in_f, hid_f, out_f, nhead):
        super().__init__()
        self.head_dim = hid_f // nhead
        self.nhead = nhead
        self.lin_q = nn.Linear(in_f, hid_f)
        self.lin_k = nn.Linear(in_f, hid_f)
        self.lin_v = nn.Linear(in_f, hid_f)
        self.out_projection = nn.Linear(hid_f, out_f)
        self.scale = self.head_dim ** 0.5

    def forward(self, x, mask):
        B, T, N = x.shape
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)

        q = q.reshape(B, T, self.nhead, self.head_dim)
        k = k.reshape(B, T, self.nhead, self.head_dim)
        v = v.reshape(B, T, self.nhead, self.head_dim)

        score = torch.einsum('bthf,bshf->bhts', q, k) / self.scale
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1)==0, -1e9)
        attn_weight = F.softmax(score, dim=-1)
        attn_out = torch.einsum('bhts,bshd->bthd', attn_weight, v)
        attn_out = attn_out.reshape(B, T, -1)
        out = self.out_projection(attn_out)
        return out


