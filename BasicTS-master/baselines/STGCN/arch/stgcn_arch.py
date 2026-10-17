import torch
import torch.nn as nn

from .stgcn_layers import STConvBlock, OutputBlock
from basicts.utils.add_layers import Kernel_Layer_ty
from basicts.utils.adj_mx_tensor_norm import calculate_adj


class STGCNChebGraphConv(nn.Module):
    """
    Paper: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Trafﬁc Forecasting
    Official Code: https://github.com/VeritasYin/STGCN_IJCAI-18 (tensorflow)
    Ref Code: https://github.com/hazdzz/STGCN
    Venue: IJCAI 2018
    Task: Spatial-Temporal Forecasting
    Note:  
        https://github.com/hazdzz/STGCN/issues/9
    Link: https://arxiv.org/abs/1709.04875
    """

    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, act_func, graph_conv_type, gso, bias, droprate, num_nodes):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, bias, droprate))
        self.st_blocks = nn.ModuleList(modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."
        self.output = OutputBlock(
            Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, bias, droprate)
        self.num_nodes = num_nodes
        self.use_alpha = 1
        self.kernel_layer = Kernel_Layer_ty(num_nodes=num_nodes, use_alpha=self.use_alpha, kernel_alpha=1.)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                dist: torch.Tensor = None, sigma: torch.Tensor = None, adj_sc: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """feedforward function of STGCN.

        Args:
            history_data (torch.Tensor): historical data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """


        adj_matrix = self.kernel_layer(dist[0], sigma[0])
        adjs, _ = calculate_adj(adj_matrix, "normlap")

        x = history_data.permute(0, 3, 1, 2).contiguous()
        for block in self.st_blocks:
            x = block(x, adjs[0])
        x = self.output(x)

        x = x.transpose(2, 3)
        return x
