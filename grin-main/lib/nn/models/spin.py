from tsl.typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
import numpy as np
# from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor
from ..layers import Kernel_Layer
from ...utils.kernel_func import rfb_kernel

from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers.base import NodeEmbedding, PositionalEncoding
from tsl.nn.layers.graph_convs import (HierarchicalSpatiotemporalCrossAttention, SpatiotemporalCrossAttention)
from tsl.nn.models.base_model import BaseModel


class SPINPositionalEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None):
        super(SPINPositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels,
                       out_channels,
                       out_channels,
                       n_layers=n_layers,
                       activation='relu')
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = NodeEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)

    def forward(self,
                x: Tensor,
                node_emb: OptTensor = None,
                node_index: OptTensor = None) -> Tensor:
        if node_emb is None:
            node_emb = self.node_emb(node_index=node_index)
        # x: [b t (n) f], node_emb: [n f] -> [b t n f] (broadcasting)
        x = self.lin(x)
        if x.ndim == 3:
            x = x.unsqueeze(-2)  # x: [b t f] -> [b t 1 f]
        x = self.activation(x + node_emb)
        out = self.mlp(x)
        out = self.positional(out)
        return out


class SPINModel(BaseModel):
    r"""The Spatiotemporal Point Inference Network (SPIN) from the paper
    `"Learning to Reconstruct Missing Data from Spatiotemporal Graphs with
    Sparse Observations" <https://arxiv.org/abs/2205.13479>`_ (Marisca et al.,
    NeurIPS 2022).
    """

    # return_type = tuple

    def __init__(self,
                 adj,
                 dist,
                 d_in: int,
                 d_hidden: int,
                 num_nodes: int,
                 d_u: Optional[int] = None,
                 output_size: Optional[int] = None,
                 temporal_self_attention: bool = True,
                 reweigh: Optional[str] = 'softmax',
                 n_layers: int = 4,
                 eta: int = 3,
                 message_layers: int = 1):
        super(SPINModel, self).__init__()

        exog_size = d_u or d_in
        output_size = output_size or d_in
        self.n_nodes = num_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        _, self.sigma = rfb_kernel(dist, l=1.)
        max_to_inf = np.max(dist[np.isfinite(dist)]) * 10
        dist[np.isinf(dist)] = max_to_inf
        self.register_buffer('dist', torch.tensor(dist).float())
        self.kernel_layer = Kernel_Layer(num_nodes=num_nodes, use_alpha=1, kernel_alpha=1., no_self_loop=True)

        self.register_buffer('adj', torch.tensor(adj).float())

        self.u_enc = SPINPositionalEncoder(in_channels=exog_size,
                                           out_channels=d_hidden,
                                           n_layers=2,
                                           n_nodes=num_nodes)

        self.h_enc = MLP(d_in, d_hidden, n_layers=2)
        self.h_norm = LayerNorm(d_hidden)

        self.valid_emb = NodeEmbedding(num_nodes, d_hidden)
        self.mask_emb = NodeEmbedding(num_nodes, d_hidden)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for layer in range(n_layers):
            x_skip = nn.Linear(d_in, d_hidden)
            encoder = SpatiotemporalCrossAttention(
                input_size=d_hidden,
                output_size=d_hidden,
                msg_size=d_hidden,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweigh=reweigh,
                mask_temporal=True,
                mask_spatial=layer < eta,
                norm=True,
                root_weight=True,
                dropout=0.0)
            readout = MLP(d_hidden, d_hidden, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def adj_to_edx(self, adj):
        edge_index = adj.nonzero(as_tuple = False).T
        edge_weight = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_weight

    def forward(self,
                x: Tensor,
                u: Tensor,
                mask: Tensor,
                # edge_index: Tensor,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        """"""
        u = u.squeeze(-1)
        adj = self.kernel_layer(self.dist, self.sigma)
        edge_index, edge_weight = self.adj_to_edx(adj)

        if target_nodes is None:
            target_nodes = slice(None)

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for layer in range(self.n_layers):
            if layer == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(node_index=node_index)
                masked = self.mask_emb(node_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[layer](x) * mask  # skip connection for valid x
            h = self.encoder[layer](h, edge_index, mask=mask, edge_weight=edge_weight)
            # Read from H to get imputations
            target_readout = self.readout[layer](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)
        if self.training:
            return x_hat, imputations
        return x_hat

    # def predict(self,
    #             x: Tensor,
    #             u: Tensor,
    #             mask: Tensor,
    #             edge_index: Tensor,
    #             node_index: OptTensor = None,
    #             target_nodes: OptTensor = None) -> Tensor:
    #     """"""
    #     imputation = self.forward(x=x,
    #                               u=u,
    #                               mask=mask,
    #                               edge_index=edge_index,
    #                               node_index=node_index,
    #                               target_nodes=target_nodes)[0]
    #     return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--hidden_size', type=int, default=32)
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--exog_size', type=int, default=None)
        parser.add_argument('--output_size', type=int, default=None)
        parser.add_argument('--temporal_self_attention', type=bool, default=True)
        parser.add_argument('--reweigh', type=str, default='softmax')
        parser.add_argument('--n_layers', type=int, default=4)
        parser.add_argument('--eta', type=int, default=3)
        parser.add_argument('--message_layers', type=int, default=1)
        return parser