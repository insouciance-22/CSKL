import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, units, activations, use_bias=True):
        super(FC, self).__init__()
        self.layers = nn.ModuleList()
        for unit, activation in zip(units, activations):
            self.layers.append(nn.Linear(unit[0], unit[1], bias=use_bias))
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'softplus':
                self.layers.append(nn.Softplus())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.backward = args.use_backward
        self.d = args.d_linear
        self.h = args.seq_len
        self.K = args.k_KNN
        self.num_interval_per_day = int(24 * 60 / args.interval)
        if args.use_alpha:
            self.alpha = nn.Parameter(torch.full((1,), args.kernel_alpha))
        else:
            self.alpha = args.kernel_alpha

        self.fc_x_gp = FC([(1, self.d)], ['None'])
        self.fc_s_gp1 = FC([(self.d, self.d)], ['relu'])
        self.fc_var_gp = FC([(self.d, self.d)], ['tanh'])
        self.fc_s_gp2 = nn.Linear(self.d, self.d, bias=False)
        if self.backward: # 反向
            self.fc_x_gp_bw = FC([(1, self.d), (self.d, self.d)], ['relu', None])
            self.fc_s_gp1_bw = FC([(self.d, self.d)], ['relu'])
            self.fc_var_gp_bw = FC([(self.d, self.d), (self.d, self.d)], ['relu', 'tanh'])
            self.fc_s_gp2_bw = nn.Linear(self.d, self.d, bias=False)
        self.fc_TE = FC([(self.num_interval_per_day, self.d), (self.d, self.d)], ['relu', None])
        if self.backward:
            self.fc_s = FC([(2 * self.d, self.d)], ['relu'])
            self.fc_var = FC([(2 * self.d, self.d)], ['relu'])
        else:
            self.fc_s = nn.Identity()
            self.fc_var = nn.Identity()
        self.fc_g1 = FC([(self.d, self.d), (self.d, self.d)], ['relu', 'relu'])
        self.gru = nn.GRUCell(input_size=self.d, hidden_size=self.d)
        self.fc_g2 = FC([(self.d, self.d)], ['relu'])
        self.fc_out = FC([(self.d, self.d), (self.d, 1)], ['relu', None])

    def heterogeneous_relations(self, relation_sc, known_node):
        if self.training:
            relation = relation_sc
        else:
            relation = torch.zeros_like(relation_sc) - 1
            relation[:, known_node] = relation_sc[:, known_node]
        _, Neighbor = torch.topk(relation, self.K, dim=1, largest=True, sorted=True)
        relation = torch.gather(relation, 1, Neighbor)
        relation = relation[:, :self.K]
        msk = relation < 0
        relation[msk] = 0
        row_sum = relation.sum(dim=1, keepdim=True)
        relation = relation / (1e-10 + row_sum)
        return relation, Neighbor

    def preprocess(self, x, relation_dist, sigma, target_set, known_node):
        relation = torch.exp(-torch.square(relation_dist / (self.alpha * sigma)))
        msk = relation > 0.1
        relation = relation * msk
        relation.fill_diagonal_(-1)
        relation, neighbor = self.heterogeneous_relations(relation, known_node)
        # relation2, neighbor2 = self.heterogeneous_relations(relation)
        relation = relation[target_set, None, None].unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        x = x[:, :, neighbor[target_set]]
        x = x.unsqueeze(-1)
        return x, relation

    def forward(self, x, add_data):
        # x_gp, relation_gp, TE, x_gp_bw=None, relation_gp_bw=None
        # gp stands Graph Spatial-Proximity.
        relation_dist = add_data['relation_dist'][0]
        sigma = add_data['sigma'][0]
        target_set = add_data['target_set'][0]
        known_node = add_data['known_node'][0]
        x_gp, relation_gp = self.preprocess(x, relation_dist, sigma, target_set, known_node)
        # Trans-Probability and Func-Similarity are not used here, because the datasets don't have this type of data.
        TE = add_data['TE']
        # (B, h, N, K, 1)->(B, N, h, K, 1)
        x_gp = torch.transpose(x_gp, 1, 2)
        N = x_gp.size(1)
        B = x_gp.size(0)
        # (B, N, 1, 1, K)->(B, N, h, 1, K)
        relation_gp = relation_gp.repeat(1, 1, self.h, 1, 1)
        # input
        # (B, N, h, K, 1)->(B, N, h, K, d) embedding
        h_gp = self.fc_x_gp(x_gp)

        # spatial aggregation
        # (B, N, h, 1, K)×(B, N, h, K, d)->(B, N, h, 1, d) 每个点s等于邻居值h乘以边权重r
        # s_gp = torch.matmul(relation_gp, h_gp)
        s_gp = torch.einsum('bnhik,bnhkd->bnhid', relation_gp, h_gp)
        # (B, N, h, 1, d)->(B, N, h, 1, d)
        s_gp = self.fc_s_gp1(s_gp)
        # (B, N, h, 1, d)-(B, N, h, K, d)->(B, N, h, K, d) 每个点与k个近邻的差异性var
        var_gp = torch.abs(s_gp - h_gp)
        # (B, N, h, 1, K)×(B, N, h, K, d)->(B, N, h, 1, d)
        # var_gp = torch.matmul(relation_gp, var_gp)
        var_gp = torch.einsum('bnhik,bnhkd->bnhid', relation_gp, var_gp)
        # (B, N, h, 1, d)->(B, N, h, 1, d)
        var_gp = self.fc_var_gp(var_gp)
        # (B, N, h, 1, d)->(B, N, h, 1, d)
        s_gp = F.relu(self.fc_s_gp2(s_gp) + var_gp)
        if self.backward:
            x_gp_bw = add_data['x2']
            relation_gp_bw = add_data['relation2']
            x_gp_bw = torch.transpose(x_gp_bw, 1, 2)
            relation_gp_bw = relation_gp_bw.repeat(1, 1, self.h, 1, 1)
            h_gp_bw = self.fc_x_gp_bw(x_gp_bw)
            s_gp_bw = torch.matmul(relation_gp_bw, h_gp_bw)
            s_gp_bw = self.fc_s_gp1_bw(s_gp_bw)
            var_gp_bw = torch.abs(s_gp_bw - h_gp_bw)
            var_gp_bw = torch.matmul(relation_gp_bw, var_gp_bw)
            var_gp_bw = self.fc_var_gp_bw(var_gp_bw)
            s_gp_bw = F.relu(self.fc_s_gp2_bw(s_gp_bw) + var_gp_bw)
            s_gp = torch.cat((s_gp, s_gp_bw), dim=-1)
            var_gp = torch.cat((var_gp, var_gp_bw), dim=-1)
        # (B, N, h, 1, 2d)->(B, N, h, 2d)
        s = torch.squeeze(s_gp, dim=3)
        # (B, N, h, 2d)->(B, N, h, d)
        s = self.fc_s(s)
        # (B, N, h, 1, d)->(B, N, h, d)
        var = torch.squeeze(var_gp, dim=3)
        # (B, N, h, d/2d)->(B, N, h, d)
        var = self.fc_var(var)
        # (B, N, h, d)->(B, N, h, d) 偏差大小决定了输入门的重要性，自适应控制输入门
        beta = self.fc_g1(var)
        beta = 1 / torch.exp(beta)
        # (B, N, h, d)*(B, N, h, d)->(B, N, h, d) normalize y based on x
        s = beta * s
        # temporal modeling
        # (B, 1, h)->(B, 1, h, num_interval_per_day)
        TE = F.one_hot(TE.to(torch.int64), num_classes=self.num_interval_per_day)
        TE = TE.to(torch.float32)
        # (B, 1, h, num_interval_per_day)->(B, 1, h, d)
        TE = self.fc_TE(TE)
        # (B, 1, h, d)->(B, N, h, d)
        TE = TE.repeat(1, N, 1, 1)
        TE = TE.float()
        # (B, N, h, d)||(B, N, h, d)->(B, N, h, 2d)
        # s = torch.cat((s, TE), dim=-1)
        pred = []
        # (B, N, d)
        state = torch.zeros(s.shape[0], N, self.d).to("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.h):
            if i==0:
                gamma = self.fc_g2(state)
                gamma = 1 / torch.exp(gamma)
                # (B, N, d)*(B, N, d) -> (B, N, d)
                state = gamma * state
                # (B*N, d)
                state = state.reshape(-1, self.d)
                s_in = s[:, :, i].reshape(-1, s.shape[-1])
                state = self.gru(s_in, state)
                # (B, N, d)
                state = state.reshape(B, N, self.d)
                # (B, N, 1, d)
                pred.append(state.unsqueeze(-2))
            else:
                gamma = self.fc_g2(var[:, :, i - 1]) # 自适应控制遗忘门
                gamma = 1 / torch.exp(gamma)
                state = gamma * state
                state = state.reshape(-1, self.d)
                s_in = s[:, :, i].reshape(-1, s.shape[-1])
                state = self.gru(s_in, state)
                state = state.reshape(B, N, self.d)
                pred.append(state.unsqueeze(-2))
        # h*(B, N, d)->h*(B*N, d)->h*(B, N, 1, d)->(B, N, h, d)
        pred = torch.cat(pred, dim=-2)
        # (B, N, h, d)->(B, N, h, 1)->(B, N, h)->(B, h, N)
        pred = self.fc_out(pred)
        pred = torch.transpose(torch.squeeze(pred), 1, 2)

        return pred