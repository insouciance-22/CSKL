# from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import torch
from layers.Kernel_Block import Kernel_Layer


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.in_dim = 1
        self.hidden_size = args.d_lsjstn
        self.tanhalpha = 2
        dropout = 0.2
        list_weight = [0.05, 0.95, 0.95]

        self.kernel_layer = Kernel_Layer(args.use_alpha, args.kernel_alpha)
        # Section: TGN for short-term learning
        self.TGNs = nn.ModuleList()
        TGN_param = {
            'layer1': {'dims_TGN': [-1, 8], 'depth_TGN': 3},
            'layer2': {'dims_TGN': [8, 16], 'depth_TGN': 3},
            'layer3': {'dims_TGN': [16, 32], 'depth_TGN': 3},
        }
        for layer_name, layer_param in TGN_param.items():
            if layer_param['dims_TGN'][0] == -1:
                layer_param['dims_TGN'][0] = self.in_dim
            TGN_block = nn.ModuleDict({
                'b_tga': TAttn(layer_param['dims_TGN'][0]),
                'b_tgn1': GCN(layer_param['dims_TGN'], layer_param['depth_TGN'], dropout, *list_weight, 'TGN'),
            })
            self.TGNs.append(TGN_block)
        self.TGN_layers = len(self.TGNs)

        # Section: Hyper Adaptive graph generation
        # Subsection: define hyperparameter for adaptive graph generation
        hyperGCN_param = {'dims_hyper': [-1, 16], 'depth_GCN': 1}
        dims_hyper = hyperGCN_param['dims_hyper']
        dims_hyper[0] = self.hidden_size
        gcn_depth = hyperGCN_param['depth_GCN']

        # Subsection: GCN and node embedding for adaptive graph generation
        self.GCN_agg1 = GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN_agg2 = GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.source_nodeemb = nn.Linear(self.in_dim, dims_hyper[-1])
        self.target_nodeemb = nn.Linear(self.in_dim, dims_hyper[-1])

        # Section: Long-term recurrent graph GRU learning
        dims = [self.in_dim + self.hidden_size, self.hidden_size]
        self.gz1_de = GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')

        # Section: Final output linear transformation
        self.fc_final_short = nn.Linear(layer_param['dims_TGN'][-1], self.in_dim)
        self.fc_final_long = nn.Linear(self.hidden_size, self.in_dim)
        self.fc_final_mix = nn.Linear(layer_param['dims_TGN'][-1] + self.hidden_size, self.in_dim)

        # gaussian output

    def calculate_normalized_laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Calculate the symmetric normalized Laplacian using PyTorch tensors for dense matrices.
        The symmetric normalized Laplacian is computed as:
        L^{Sym} = I - D^{-1/2} A D^{-1/2},
        where A is the adjacency matrix, D is the degree matrix.

        Args:
            adj (torch.Tensor): Adjacency matrix A (n x n).

        Returns:
            torch.Tensor: Symmetric normalized Laplacian L^{Sym} (n x n).
        """
        # 计算度数向量 D (行和)
        degree = adj.sum(dim=1)  # 每行的和，即节点的度数

        # 计算 D^{-1/2}
        degree_inv_sqrt = degree.pow(-0.5)  # 度数倒数平方根
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 处理 degree 为零的情况，防止除零错误

        # 创建 D^{-1/2} 对角矩阵
        degree_inv_sqrt_mat = torch.diag(degree_inv_sqrt)

        # 计算对称归一化拉普拉斯矩阵 L^{Sym} = I - D^{-1/2} A D^{-1/2}
        identity = torch.eye(adj.shape[0], device=adj.device)  # 单位矩阵 I
        laplacian = identity - degree_inv_sqrt_mat @ adj @ degree_inv_sqrt_mat  # 矩阵乘法

        return laplacian

    def forward(self, x, add_data):
        """
        Kriging for one iteration
        :param sample: graphs [batch, num_timesteps, num_nodes, num_features]
        :param predefined_A: list, len(2)
        :return: completed graph
        """
        dist = add_data['D_S']
        sigma = add_data['sigma']
        adj_matrix = self.kernel_layer(dist, sigma)
        sample = x.unsqueeze(-1)
        predefined_A = (self.calculate_normalized_laplacian(adj_matrix) + self.calculate_normalized_laplacian(adj_matrix.transpose(0, 1))) / 2
        batch_size, num_nodes, num_t = sample.shape[0], sample.shape[2], sample.shape[1]
        hidden_state, _ = self.initHidden(batch_size * num_nodes, self.hidden_size, device=sample.device)

        # Section: Long-term graph GRU encoding
        # GRU学习长期自适应邻接矩阵A，得到hidden_state，仅使用前四个时间步
        for current_t in range(0, num_t-4, 4):
            current_graph = sample[:, current_t]
            hidden_state = self.gru_step(current_graph, hidden_state, predefined_A)

        # Section: Short-term learning (joint spatiotemporal attention)
        # 根据注意力计算短期（最近四个时间步）注意力邻接矩阵并用于卷积，得到gat_result
        atten_scores = []
        b_src_filter = sample[:, -4:]
        tar_filer = sample[:, -1]
        for i in range(self.TGN_layers):
            # beforehand temporal graph attention (include target graph)
            b_attn_scores = self.TGNs[i]['b_tga'](tar_filer, b_src_filter)
            atten_scores.append(b_attn_scores)
            b_src_filter = b_src_filter.reshape([-1, num_nodes, b_src_filter.shape[-1]])
            # merge attn scores into pre_defined adjacent matrix
            b_A_tgn = [predefined_A.unsqueeze(0) + b_attn_scores.reshape([-1, num_nodes, num_nodes])]
            b_src_filter = (
                    self.TGNs[i]['b_tgn1'](b_src_filter, b_A_tgn[0])
                    ).reshape([batch_size, b_src_filter.shape[0]//batch_size, num_nodes, -1])
            b_src_filter = torch.relu(b_src_filter)
            tar_filer = torch.sum(b_src_filter, dim=1)
        gat_result = self.fc_final_short(tar_filer).reshape([batch_size, num_nodes, -1])

        # Section: long results
        # 利用上一步的短期结果更新hidden_state，得到长期输出，长短期拼接得到最终输出
        hidden_state = self.gru_step(gat_result, hidden_state, predefined_A)
        hidden_state = hidden_state.reshape([batch_size, num_nodes, -1])
        gru_result = self.fc_final_long(hidden_state)

        final_result = self.fc_final_mix(torch.cat([tar_filer, hidden_state], dim=2))

        return gat_result, gru_result, final_result, atten_scores[0]

    def gru_step(self, current_graph, hidden_state, predefined_A):
        """
        Kriging one time step (reference graph)
        :param: current_graph: current input for graph GRU [batch, num_nodes, num_features]
        :param: hidden_state:  [batch, num_nodes, hidden_size]
        :param: predefined_A: predefined adjacent matrix, static per iteration, no need batch [num_nodes, num_nodes]
        :return: kriging results of current reference graph
        """
        batch_size, num_nodes = current_graph.shape[0], current_graph.shape[1]
        hidden_state = hidden_state.view(-1, num_nodes, self.hidden_size)

        # Section: Generate graph for graph learning
        graph_source = self.GCN_agg1(hidden_state, predefined_A)
        graph_target = self.GCN_agg2(hidden_state, predefined_A)

        nodevec_source = torch.tanh(self.tanhalpha * torch.mul(
            self.source_nodeemb(current_graph), graph_source))
        nodevec_target = torch.tanh(self.tanhalpha * torch.mul(
            self.target_nodeemb(current_graph), graph_target))

        a = torch.matmul(nodevec_source, nodevec_target.transpose(2, 1)) \
            - torch.matmul(nodevec_target, nodevec_source.transpose(2, 1))

        adp_adj = torch.relu(torch.tanh(self.tanhalpha * a))

        adp = self.adj_processing(adp_adj, num_nodes, predefined_A, current_graph.device)

        # Section: Long_term Learning
        combined = torch.cat((current_graph, hidden_state), -1)
        z = torch.sigmoid(self.gz1_de(combined, adp))
        r = torch.sigmoid(self.gr1_de(combined, adp))
        temp = torch.cat((current_graph, torch.mul(r, hidden_state)), dim=-1)
        cell_state = torch.tanh(self.gc1_de(temp, adp))
        hidden_state = torch.mul(z, hidden_state) + torch.mul(1 - z, cell_state)
        hidden_state = hidden_state.reshape([-1, self.hidden_size])

        return hidden_state

    def adj_processing(self, adp_adj, num_nodes, predefined_A, device):
        adp_adj = adp_adj + torch.eye(num_nodes).to(device)
        adp_adj = adp_adj / torch.unsqueeze(adp_adj.sum(-1), -1)
        return [adp_adj, predefined_A]

    def initHidden(self, batch_size, hidden_size, device):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(device))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size).to(device))
            # nn.init.orthogonal_(Hidden_State)
            # nn.init.orthogonal_(Cell_State)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def get_source_index(self, num_t, current_t):

        b_index = [i for i in range(max(0, current_t-2), current_t+1)]  # must has at least oen index
        a_index = [i for i in range(current_t+1, min(current_t+3, num_t))]  # might be empty
        return b_index, a_index


class GCN(nn.Module):
    def __init__(self, dims, gdep, dropout, tanhalpha, beta, gamma, type=None):
        super(GCN, self).__init__()
        if type == 'RNN':
            self.gconv = GconvAdp()
            self.gconv_preA = GconvPre()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'hyper':
            self.gconv_preA = GconvPre()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'TGN':
            self.gconv = GconvAdp()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
        else:
            raise NotImplementedError('GCN type is not implemented!')

        if dropout:
            self.dropout_ = nn.Dropout(p=dropout)

        self.dropout = dropout
        self.gdep = gdep
        self.tanhalpha = tanhalpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x, adj):
        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.tanhalpha * x + self.beta * self.gconv(
                    h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
                out.append(h)
        elif self.type_GNN == 'hyper':
            for _ in range(self.gdep):
                h = self.tanhalpha * x + self.gamma * self.gconv_preA(h, adj)
                out.append(h)
        elif self.type_GNN == 'TGN':
            for _ in range(self.gdep):
                h = self.tanhalpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)

        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)
        if self.dropout:
            ho = self.dropout_(ho)

        return ho


class GconvAdp(nn.Module):
    def __init__(self):
        super(GconvAdp, self).__init__()

    def forward(self, x, A):
        if x.shape[0] != A.shape[0]:
            A = A.unsqueeze(1).repeat(1, x.shape[0]//A.shape[0], 1, 1).reshape([-1, x.shape[1], x.shape[1]])
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class GconvPre(nn.Module):
    def __init__(self):
        super(GconvPre, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class MultiAtten(nn.Module):
    def __init__(self):
        super(MultiAtten, self).__init__()


class TAttn(nn.Module):

    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in * 4
        self.w_query = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.w_key = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.trans = nn.Parameter(torch.FloatTensor(dim_out, 1))

    def forward(self, query, keys):
        """
        :param query: current kriging graph [batch, num_node, num_features_2]
        :param keys: graphs in the temporal direction [batch, num_time, num_node, num_features_1]
        :return: temporal attention scores (a.k.a temporal attention adjacent matrix) [batch, num_time, num_node, num_node]
        """
        query = torch.matmul(query, self.w_query).unsqueeze(1).unsqueeze(3)
        keys = torch.matmul(keys, self.w_key).unsqueeze(2)
        attn_scores = torch.matmul(torch.tanh(query + keys + self.bias), self.trans).squeeze(-1)
        # multi_dimensional Softmax
        attn_scores = torch.exp(attn_scores) / torch.sum(torch.exp(attn_scores), dim=-1, keepdim=True)

        return attn_scores