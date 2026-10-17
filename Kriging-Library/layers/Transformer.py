import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)] #(B, N, T, F)


class TemporalEmbedding(nn.Module):
    def __init__(self, num_classes, d_model):
        super(TemporalEmbedding, self).__init__()
        self.num_classes = num_classes
        self.timestamp_embedding = nn.Linear(num_classes, d_model)

    def forward(self, TE): #(B, T)
        timestamp_encoded = F.one_hot(TE, self.num_classes).float() #(B, T, num_classes)
        timestamp_encoded = self.timestamp_embedding(timestamp_encoded) #(B, T, D)
        return timestamp_encoded


class TemporalSelfAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes=0):
        super(TemporalSelfAttention, self).__init__()
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)

        self.positional_encoding = PositionalEncoding(feature_dim)
        self.timestamp_embedding = TemporalEmbedding(num_classes, feature_dim) if num_classes > 0 else None

        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, X, TE):
        X = X.unsqueeze(-1) # (B, N, T, 1)
        # 是否需要对x进行embed
        B, N, T, _ = X.shape
        X = self.positional_encoding(X)
        # 添加时间戳编码（如果提供）
        if TE is not None and self.timestamp_embedding is not None:
            timestamp_encoded = self.timestamp_embedding(TE).unsqueeze(1)  # (B, 1, T, D)
            X = X + timestamp_encoded

        # 查询、键、值计算
        Q = self.query_proj(X)  # (B, N, T, hidden_dim)
        K = self.key_proj(X)  # (B, N, T, hidden_dim)
        V = self.value_proj(X)  # (B, N, T, hidden_dim)

        # 计算注意力分数
        attention_scores = torch.einsum('bntd,bnqd->bntq', Q, K) / (Q.shape[-1] ** 0.5)  # (B, N, T, T)

        # 应用稀疏掩码
        # if mask is not None:
        #     mask = mask.unsqueeze(-1)  # (B, N, T, 1)
        #     attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, N, T, T)

        # 注意力加权求和
        attention_output = torch.einsum('bntq,bnqd->bntd', attention_weights, V)  # (B, N, T, hidden_dim)

        # 输出投影
        output = self.output_proj(attention_output)  # (B, N, T, F)

        output = output.squeeze(-1)

        output = F.relu(output)

        return output

