import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()
        self.d = args.d_linear
        self.h = args.seq_len
        self.K = args.k_KNN

        self.T = int(24 * 60 / args.interval)

        self.input_fc = nn.Linear(1, self.d)
        self.input_fc_bw = nn.Linear(1, self.d)
        # 定义全连接层
        self.fc1 = nn.Linear(self.d, self.d)
        self.fc2 = nn.Linear(self.d, self.d)
        self.fc3 = nn.Linear(self.d, self.d)
        self.fc4 = nn.Linear(self.d, self.d)
        self.fc5 = nn.Linear(self.d, self.d)
        self.fc1_bw = nn.Linear(self.d, self.d)
        self.fc2_bw = nn.Linear(self.d, self.d)
        self.fc3_bw = nn.Linear(self.d, self.d)
        self.fc4_bw = nn.Linear(self.d, self.d)
        self.fc5_bw = nn.Linear(self.d, self.d)

        self.fc6 = nn.Linear(2*self.d, self.d)
        self.fc7 = nn.Linear(2*self.d, self.d)
        self.fc8 = nn.Linear(self.d, self.d)
        self.fc9 = nn.Linear(self.d, self.d)
        self.gru = nn.GRUCell(input_size=2*self.d, hidden_size=self.d)

        self.output_fc = nn.Linear(self.d, 1)  # 最终输出
        self.te_fc = nn.Linear(self.T, self.d)

    def forward(self, x, add_data):
        x_gp_fw = x #(B, H, N, K, 1)
        x_gp_fw = x_gp_fw.permute(0, 2, 1, 3, 4).squeeze(dim=0)
        x_gp_bw = add_data['x2']
        x_gp_bw = x_gp_bw.permute(0, 2, 1, 3, 4).squeeze(dim=0)
        gp_fw = add_data['relation'] #(B, N, 1, 1, K)
        gp_fw = gp_fw.squeeze(dim=0)
        gp_bw = add_data['relation2']
        gp_bw = gp_bw.squeeze(dim=0)
        TE = add_data['TE'] #(B, 1, H)
        TE = TE.squeeze(dim=0)

        # 输入数据归一化
        # x_gp_fw = (x_gp_fw - mean) / std
        # x_gp_bw = (x_gp_bw - mean) / std

        # 全连接层处理
        x_gp_fw = F.relu(self.input_fc(x_gp_fw))
        x_gp_bw = F.relu(self.input_fc_bw(x_gp_bw))

        # 空间聚合
        gp_fw = gp_fw.repeat(1, self.h, 1, 1)
        gp_bw = gp_bw.repeat(1, self.h, 1, 1)

        y_gp_fw = torch.matmul(gp_fw, x_gp_fw)
        y_gp_bw = torch.matmul(gp_bw, x_gp_bw)

        y_gp_fw = F.relu(self.fc1(y_gp_fw))
        y_gp_bw = F.relu(self.fc1_bw(y_gp_bw))

        x_gp_fw = F.relu(self.fc2(x_gp_fw))
        x_gp_bw = F.relu(self.fc2_bw(x_gp_bw))

        x_gp_fw = torch.abs(y_gp_fw - x_gp_fw)
        x_gp_bw = torch.abs(y_gp_bw - x_gp_bw)

        x_gp_fw = torch.matmul(gp_fw, x_gp_fw)
        x_gp_bw = torch.matmul(gp_bw, x_gp_bw)

        x_gp_fw = F.relu(self.fc3(x_gp_fw))
        x_gp_bw = F.relu(self.fc3_bw(x_gp_bw))

        y_gp_fw = self.fc4(y_gp_fw)
        y_gp_bw = self.fc4_bw(y_gp_bw)

        y_gp_fw = x_gp_fw + y_gp_fw
        y_gp_bw = x_gp_bw + y_gp_bw

        y_gp_fw = F.relu(self.fc5(y_gp_fw))
        y_gp_bw = F.relu(self.fc5_bw(y_gp_bw))

        # 时间建模
        TE = F.one_hot(TE, num_classes=self.T).float()
        TE = F.relu(self.te_fc(TE))
        TE = TE.repeat(x_gp_fw.size(0), 1, 1)

        y = torch.cat((y_gp_fw, y_gp_bw), dim=-1)
        y = y.squeeze(dim=2)
        y = F.relu(self.fc6(y))

        x = torch.cat((x_gp_fw, x_gp_bw), dim=-1)
        x = x.squeeze(dim=2)
        x = F.relu(self.fc7(x))

        g1 = F.relu(self.fc8(x))
        g1 = 1 / torch.exp(g1)
        y = g1 * y

        y = torch.cat((y, TE), dim=-1)

        pred = []
        state = torch.zeros(x_gp_fw.size(0), self.d).to(x_gp_fw.device)

        for i in range(self.h):
            if i == 0:
                g2 = F.relu(self.fc9(state))
                g2 = 1 / torch.exp(g2)
                state = g2 * state
                state = self.gru(y[:, i], state)
                pred.append(state.unsqueeze(1))
            else:
                g2 = F.relu(self.fc9(x[:, i-1]))
                g2 = 1 / torch.exp(g2)
                state = g2 * state
                state = self.gru(y[:, i], state)
                pred.append(state.unsqueeze(1))

        pred = torch.cat(pred, dim=1)

        # 输出层
        pred = self.output_fc(pred).squeeze().unsqueeze(0).permute(0, 2, 1)

        return pred


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
        self.d = args.d_linear
        d = self.d
        self.h = args.seq_len
        self.K = args.k_KNN
        self.num_interval_per_day = int(24 * 60 / args.interval)
        self.fc_x_gp1 = FC([(1, d), (d, d)], ['relu', None])
        self.fc_x_gp1_bw = FC([(1, d), (d, d)], ['relu', None])
        self.fc_y_gp1 = FC([(d, d), (d, d)], ['relu', None])
        self.fc_y_gp1_bw = FC([(d, d), (d, d)], ['relu', None])
        self.fc_x_gp2 = FC([(d, d), (d, d)], ['relu', None])
        self.fc_x_gp2_bw = FC([(d, d), (d, d)], ['relu', None])
        self.fc_x_gp3 = FC([(d, d), (d, d)], ['relu', 'tanh'])
        self.fc_x_gp3_bw = FC([(d, d), (d, d)], ['relu', 'tanh'])
        self.fc_y_gp2 = FC([(d, d)], [None])
        self.fc_y_gp2_bw = FC([(d, d)], [None])
        self.fc_y_gp3 = FC([(d, d), (d, d)], ['relu', None])
        self.fc_y_gp3_bw = FC([(d, d), (d, d)], ['relu', None])
        self.fc_TE = FC([(self.num_interval_per_day, d), (d, d)], ['relu', None])
        self.fc_y = FC([(2*d, d), (d, d)], ['relu', None])
        self.fc_x = FC([(2*d, d), (d, d)], ['relu', None])
        self.fc_g1 = FC([(d, d), (d, d)], ['relu', 'relu'])
        self.gru = nn.GRUCell(input_size=2*d, hidden_size=d)
        self.fc_g2 = FC([(d, d)], ['relu'])
        self.fc_out = FC([(d, d), (d, d), (d, 1)], ['relu', 'relu', None])

    def forward(self, x, add_data):
        x_gp = x  # (B, H, N, K, 1)
        x_gp = x_gp.permute(0, 2, 1, 3, 4).squeeze(dim=0)
        x_gp_bw = add_data['x2']
        x_gp_bw = x_gp_bw.permute(0, 2, 1, 3, 4).squeeze(dim=0)
        relation_gp = add_data['relation']  # (B, N, 1, 1, K)
        relation_gp = relation_gp.squeeze(dim=0)
        relation_gp_bw = add_data['relation2']
        relation_gp_bw = relation_gp_bw.squeeze(dim=0)
        TE = add_data['TE']  # (B, 1, H)
        TE = TE.squeeze(dim=0)
        N = x_gp.size(0)
        # x_gp = (x_gp - self.mean) / self.std
        # x_gp_bw = (x_gp_bw - self.mean) / self.std
        # (N, 1, 1, K)->(N, h, 1, K)
        relation_gp = relation_gp.repeat(1, self.h, 1, 1)
        relation_gp_bw = relation_gp_bw.repeat(1, self.h, 1, 1)
        # input
        # (N, h, K, 1)->(N, h, K, d)
        x_gp = self.fc_x_gp1(x_gp)
        x_gp_bw = self.fc_x_gp1_bw(x_gp_bw)
        # spatial aggregation
        # (N, h, 1, K)×(N, h, K, d)->(N, h, 1, d)
        y_gp = torch.matmul(relation_gp, x_gp)
        y_gp_bw = torch.matmul(relation_gp_bw, x_gp_bw)
        # (N, h, 1, d)->(N, h, 1, d)
        y_gp = self.fc_y_gp1(y_gp)
        y_gp_bw = self.fc_y_gp1_bw(y_gp_bw)
        # (N, h, K, d)->(N, h, K, d)
        x_gp = self.fc_x_gp2(x_gp)
        x_gp_bw = self.fc_x_gp2_bw(x_gp_bw)
        # (N, h, 1, d)-(N, h, K, d)->(N, h, K, d)
        x_gp = torch.abs(y_gp - x_gp)
        x_gp_bw = torch.abs(y_gp_bw - x_gp_bw)
        # (N, h, 1, K)×(N, h, K, d)->(N, h, 1, d)
        x_gp = torch.matmul(relation_gp, x_gp)
        x_gp_bw = torch.matmul(relation_gp_bw, x_gp_bw)
        # (N, h, 1, d)->(N, h, 1, d)
        x_gp = self.fc_x_gp3(x_gp)
        x_gp_bw = self.fc_x_gp3_bw(x_gp_bw)
        # (N, h, 1, d)->(N, h, 1, d)
        y_gp = self.fc_y_gp2(y_gp)
        y_gp_bw = self.fc_y_gp2_bw(y_gp_bw)
        # (N, h, 1, d)+(N, h, 1, d)->(N, h, 1, d)
        y_gp = y_gp + x_gp
        y_gp_bw = y_gp_bw + x_gp_bw
        # (N, h, 1, d)->(N, h, 1, d)
        y_gp = self.fc_y_gp3(y_gp)
        y_gp_bw = self.fc_y_gp3_bw(y_gp_bw)
        # temporal modeling
        # (1, h)->(1, h, num_interval_per_day)
        TE = F.one_hot(TE.to(torch.int64), num_classes=self.num_interval_per_day)
        TE = TE.to(torch.float32)
        # (1, h, num_interval_per_day)->(1, h, d)
        TE = self.fc_TE(TE)
        # (1, h, d)->(N, h, d)
        TE = TE.repeat(N, 1, 1)
        # (N, h, 1, d)->(N, h, d)
        y = torch.cat((y_gp, y_gp_bw), dim=-1)
        y = torch.squeeze(y, dim=2)
        # (N, h, d)->(N, h, d)
        y = self.fc_y(y)
        # 2(N, h, 1, d)->(N, h, 2d)
        x = torch.cat((x_gp, x_gp_bw), dim=-1)
        x = torch.squeeze(x, dim=2)
        # (N, h, d)->(N, h, d)
        x = self.fc_x(x)
        # (N, h, d)->(N, h, d)
        g1 = self.fc_g1(x)
        g1 = 1 / torch.exp(g1)
        # (N, h, d)*(N, h, d)->(N, h, d)
        y = g1 * y
        TE = TE.float()
        # (N, h, d)||(N, h, d)->(N, h, 2d)
        y = torch.cat((y, TE), dim=-1)
        pred = []
        # (N, d)
        state = torch.zeros(N, self.d).to("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.h):
            if i==0:
                g2 = self.fc_g2(state)
                g2 = 1 / torch.exp(g2)
                state = g2 * state
                state = self.gru(y[:, i], state)
                pred.append(state.unsqueeze(1))
            else:
                g2 = self.fc_g2(x[:, i - 1])
                g2 = 1 / torch.exp(g2)
                state = g2 * state
                state = self.gru(y[:, i], state)
                pred.append(state.unsqueeze(1))
        # h*(N,d)->h*(N, 1, d)->(N, h, d)
        pred = torch.cat(pred, dim=1)
        # (N, h, d)->(N, h, 1)
        pred = self.fc_out(pred).squeeze().unsqueeze(0).permute(0, 2, 1)
        # return pred * self.std + self.mean
        return pred
