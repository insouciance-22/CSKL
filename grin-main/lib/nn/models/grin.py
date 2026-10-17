import torch
from einops import rearrange
from torch import nn
import numpy as np
import torch.nn.functional as F

from ..layers import (
    Kernel_Layer,
    Kernel_Layer_th,
    Kernel_Layer_ada,
    Kernel_Layer_22,
)
from ..layers import BiGRIL
# from ..layers import BiGRIL_GAT
from ...utils.parser_utils import str_to_bool
from ...utils.kernel_func import rfb_kernel


class GRINet(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True,
                 dist=None,
                 num_nodes=0,
                 kernel_type='Kernel_Layer_th',
                 kernel_alpha=1.0,
                 kernel_theta=0.0,
                 kernel_use_alpha=True,
                 kernel_no_self_loop=True):
        super(GRINet, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        # self.d_u = int(d_u) if d_u is not None else 0
        self.d_u = 0
        self.d_emb = int(d_emb) if d_emb is not None else 0

        self.kernel_type = kernel_type
        self.kernel_alpha = float(kernel_alpha)
        self.kernel_theta = float(kernel_theta)
        self.kernel_use_alpha = bool(kernel_use_alpha)
        self.kernel_no_self_loop = bool(kernel_no_self_loop)

        kernel_classes = {
            'Kernel_Layer': Kernel_Layer,
            'Kernel_Layer_th': Kernel_Layer_th,
            'Kernel_Layer_ada': Kernel_Layer_ada,
            'Kernel_Layer_22': Kernel_Layer_22,
        }
        if self.kernel_type not in kernel_classes:
            raise ValueError(
                f'Unknown kernel_type: {self.kernel_type}. '
                f'Available kernels: {list(kernel_classes.keys())}'
            )

        _, self.sigma = rfb_kernel(dist, l=1.)
        max_to_inf = np.max(dist[np.isfinite(dist)]) * 10
        dist[np.isinf(dist)] = max_to_inf
        self.register_buffer('dist', torch.tensor(dist).float())
        kernel_cls = kernel_classes[self.kernel_type]
        self.kernel_layer = kernel_cls(
            num_nodes=num_nodes,
            use_alpha=self.kernel_use_alpha,
            kernel_alpha=self.kernel_alpha,
            theta=self.kernel_theta,
            no_self_loop=self.kernel_no_self_loop
        )

        self.register_buffer('adj', torch.tensor(adj).float())
        self.impute_only_holes = impute_only_holes

        self.bigrill = BiGRIL(input_size=self.d_in,
                              ff_size=d_ff,
                              ff_dropout=ff_dropout,
                              hidden_size=self.d_hidden,
                              embedding_size=self.d_emb,
                              n_nodes=self.adj.shape[0],
                              n_layers=n_layers,
                              kernel_size=kernel_size,
                              decoder_order=decoder_order,
                              global_att=global_att,
                              u_size=self.d_u,
                              layer_norm=layer_norm,
                              merge=merge)

        # self.adj_module = AdaptiveAdj1(num_nodes, 64)
        # self.adj_module = SAAdjacency(num_nodes,d_in)


    def forward(self, x, mask=None, u=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        u = None
        # adj = torch.ones_like(self.dist)
        # adj.fill_diagonal_(0)

        adj = self.kernel_layer(self.dist, self.sigma)
        # adj = self.adj_module(self.adj)
        # adj = self.adj
        # adj = self.adj_module()

        x = rearrange(x, 'b s n c -> b c n s')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')

        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')

        # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        imputation, prediction = self.bigrill(x, adj, mask=mask, u=u, cached_support=False)
        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)
        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = torch.transpose(imputation, -3, -1)
        prediction = torch.transpose(prediction, -3, -1)
        if self.training:
            return imputation, prediction
        return imputation


class AdaptiveAdj(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, static_W):
        A_adp = F.relu(self.E1 @ self.E2.T)
        A_adp = F.softmax(A_adp, dim=1)
        return self.alpha * static_W + (1 - self.alpha) * A_adp


class AdaptiveAdj1(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self):
        A_adp = F.relu(self.E1 @ self.E2.T)
        A_adp = F.softmax(A_adp, dim=1)
        return A_adp


# class SAAdjacency(nn.Module):
#     def __init__(self, feat_dim, d_k=16):
#         super().__init__()
#         self.Wq = nn.Linear(feat_dim, d_k)
#         self.Wk = nn.Linear(feat_dim, d_k)
#
#     def forward(self, X_seq):
#         B, T, N, C = X_seq.shape
#         X_seq = X_seq.permute(0, 2, 1, 3).reshape(B, N, T * C)
#         # X_seq: [B, N, F_seq], F_seq = T*F
#         Q = self.Wq(X_seq)  # [B, N, d_k]
#         K = self.Wk(X_seq)  # [B, N, d_k]
#         attn = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # [B, N, N]
#         A_adapt = F.softmax(attn, dim=-1)  # [B, N, N]
#         return A_adapt
class SAAdjacency(nn.Module):
    def __init__(self, num_nodes, feat_dim, d_k=16, use_node_emb=True):
        super().__init__()
        self.Wq = nn.Linear(feat_dim, d_k)
        self.Wk = nn.Linear(feat_dim, d_k)
        self.use_node_emb = use_node_emb
        if use_node_emb:
            self.node_emb = nn.Parameter(torch.randn(num_nodes, feat_dim))

    def forward(self, X):
        # X: [B, T, N, F]
        B, T, N, C = X.shape
        A_all = []
        for t in range(T):
            x_t = X[:, t]  # [B, N, F]
            if self.use_node_emb:
                x_t = x_t + self.node_emb.unsqueeze(0)  # 加节点静态嵌入
            Q = self.Wq(x_t)
            K = self.Wk(x_t)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
            A_t = F.softmax(attn, dim=-1)
            A_all.append(A_t)
        A_all = torch.stack(A_all, dim=1)  # [B, T, N, N]
        A_mean = A_all.mean(dim=1)
        return A_mean


    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--decoder-order', type=int, default=1)
        parser.add_argument('--d-u', type=int, default=0)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--global-att', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--merge', type=str, default='mlp')
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument('--kernel-type', type=str,
                            choices=['Kernel_Layer', 'Kernel_Layer_th',
                                     'Kernel_Layer_ada', 'Kernel_Layer_22'],
                            default='Kernel_Layer_th')
        parser.add_argument('--kernel-alpha', type=float, default=1.0)
        parser.add_argument('--kernel-theta', type=float, default=0.0)
        parser.add_argument('--kernel-use-alpha', type=str_to_bool,
                            nargs='?', const=True, default=True)
        parser.add_argument('--kernel-no-self-loop', type=str_to_bool,
                            nargs='?', const=True, default=True)
        return parser
