import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        # 多头线性变换
        self.W = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.Tensor(heads, out_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(heads, out_dim, 1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(out_dim * heads, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h: [batch, N, F]
        # adj: [N, N] 或 [batch, N, N]
        h = h.transpose(1, 2)
        B, N, _ = h.shape
        h_prime = []

        for head in range(self.heads):
            Wh = torch.matmul(h, self.W[head])  # [B, N, out_dim]

            # 计算注意力分数
            f_src = torch.matmul(Wh, self.a_src[head])  # [B, N, 1]
            f_dst = torch.matmul(Wh, self.a_dst[head])  # [B, N, 1]

            e = f_src + f_dst.transpose(1, 2)  # [B, N, N]
            e = self.leaky_relu(e)

            if adj.dim() == 2:
                e = e.masked_fill(adj.unsqueeze(0) == 0, -1e9)
            else:
                e = e.masked_fill(adj == 0, -1e9)

            alpha = F.softmax(e, dim=-1)  # [B, N, N]
            alpha = self.dropout(alpha)

            # 聚合邻居
            h_out = torch.matmul(alpha, Wh)  # [B, N, out_dim]
            h_prime.append(h_out)

        h_prime = torch.cat(h_prime, dim=-1) if self.concat else torch.mean(torch.stack(h_prime, dim=0), dim=0)

        out = self.lin(h_prime)
        out = out.transpose(1, 2)
        return out
