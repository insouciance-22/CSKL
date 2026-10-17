import torch
from torch import nn
import numpy as np
from basicts.utils.add_layers import Kernel_Layer_ty
from basicts.utils.adj_mx_tensor_norm import calculate_adj

from .dcrnn_cell import DCGRUCell


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch.nn.functional as F
from basicts.utils.parser_utils import str_to_bool

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
        A_time = A_all.mean(dim=1)
        A_mean = A_time.mean(dim=0)
        return A_mean


    # @staticmethod
    # def add_model_specific_args(parser):
    #     parser.add_argument('--d-hidden', type=int, default=64)
    #     parser.add_argument('--d-ff', type=int, default=64)
    #     parser.add_argument('--ff-dropout', type=int, default=0.)
    #     parser.add_argument('--n-layers', type=int, default=1)
    #     parser.add_argument('--kernel-size', type=int, default=2)
    #     parser.add_argument('--decoder-order', type=int, default=1)
    #     parser.add_argument('--d-u', type=int, default=0)
    #     parser.add_argument('--d-emb', type=int, default=8)
    #     parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
    #     parser.add_argument('--global-att', type=str_to_bool, nargs='?', const=True, default=False)
    #     parser.add_argument('--merge', type=str, default='mlp')
    #     parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
    #     return parser


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(
            model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adjs, hidden_state=None):
        batch_size, _ = inputs.size()

        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)


        hidden_states = []

        output = inputs

        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adjs)

            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        stacked_hidden_states = torch.stack(hidden_states)

        # runs in O(num_layers) so not too slow
        return output, stacked_hidden_states


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adjs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adjs)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, Seq2SeqAttrs):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Link: https://arxiv.org/abs/1707.01926
    Official Code:
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py,
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
    Venue: ICLR 2018
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, adj_mx, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 2000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False))
        self.num_nodes = model_kwargs.get("num_nodes")
        self.use_alpha = 1
        self.kernel_layer = Kernel_Layer_ty(num_nodes=model_kwargs.get("num_nodes"), use_alpha=self.use_alpha, kernel_alpha=1.)

        # self.adj_module = SAAdjacency(self.num_nodes, model_kwargs.get("input_dim"))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adjs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], adjs, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adjs, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, adjs, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                dist: torch.Tensor = None, sigma: torch.Tensor = None, adj_sc: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [L, B, N*C]
            future_data (torch.Tensor, optional): future data with shape [L, B, N*C_out]
            batch_seen (int, optional): batches seen till now, used for curriculum learning. Defaults to None.

        Returns:
            torch.Tensor: prediction with shape [L, B, N*C_out]
        """

        # print("\n========== DCRNN.forward ==========")
        # print("history_data:", history_data.shape)
        if future_data is not None:
            print("future_data:", future_data.shape)
        # print("dist[0]:", dist[0].shape)
        # print("sigma[0]:", sigma[0].shape)

        adj_matrix = self.kernel_layer(dist[0], sigma[0])
        # adj_matrix = self.adj_module(history_data)
        # print("history_data for SA:", history_data.shape)
        # print("attention adj:", adj_matrix.shape)
        # print(adj_matrix.min())
        # print(adj_matrix.max())
        # print(adj_matrix.sum(dim=1))
        # adjs = []
        # adjs.append(adj_matrix)
        # adjs.append(adj_matrix.T)
        adjs, _ = calculate_adj(adj_matrix, "doubletransition")

        # reshape data
        batch_size, length, num_nodes, channels = history_data.shape

        # print("\n--- history reshape ---")
        # print("before reshape:", history_data.shape)

        history_data = history_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]

        # print("after reshape:", history_data.shape)

        history_data = history_data.transpose(0, 1)         # [L, B, N*C]

        # print("after transpose:", history_data.shape)

        if future_data is not None:
            future_data = future_data[..., [0]]     # teacher forcing only use the first dimension.
            batch_size, length, num_nodes, channels = future_data.shape
            future_data = future_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
            future_data = future_data.transpose(0, 1)         # [L, B, N*C]

        # DCRNN
        # print("\n--- before encoder ---")
        # print("history_data passed to encoder:", history_data.shape)

        encoder_hidden_state = self.encoder(history_data, adjs)

        # print("encoder_hidden_state:", encoder_hidden_state.shape)

        outputs = self.decoder(encoder_hidden_state, adjs, future_data,
                               batches_seen=batch_seen)      # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(B, L, self.num_nodes,
                               self.decoder_model.output_dim)

        if batch_seen == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
            print("Parameter Number: ".format(count_parameters(self)))
            print(count_parameters(self))
        return outputs
