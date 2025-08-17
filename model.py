import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.edge_query = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.edge_key = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, edge_inital, G, Pair=False, Group=False):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        edge_query = self.edge_query(edge_inital)  # batch_size 4 seq_len d_model
        edge_key = self.edge_key(edge_inital)      # batch_size 4 seq_len d_model
        if Pair:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        if Group:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            div = torch.sum(G, dim=1)[:, None, :, None]
            Gquery = query + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_query) / div  # q [batch, num_agent, heads, 64/heads]
            Gkey = key + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_key) / div
            attention = torch.matmul(Gquery, Gkey.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        attention = self.softmax(attention / self.scaled_factor)
        return attention


class Edge_inital(nn.Module):
    def __init__(self, in_dims=2, d_model=64):
        super(Edge_inital, self).__init__()
        self.x_embedding = nn.Linear(in_dims, d_model//4)
        self.edge_embedding = nn.Linear(d_model//4, d_model//4)
    def forward(self, x, G):
        assert len(x.shape) == 3
        embeddings = self.x_embedding(x)  # batch_size seq_len d_model
        div = torch.sum(G, dim=-1)[:, :, None]
        edge_init = self.edge_embedding(torch.matmul(G, embeddings) / div)  # T N d_model
        edge_init = edge_init.unsqueeze(1).repeat(1, 4, 1, 1)
        return edge_init


class GateFusion(nn.Module):
    def __init__(self, obs_len=8):
        super(GateFusion, self).__init__()
        self.avg_gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(obs_len, obs_len//4, 1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(obs_len//4, obs_len, 1, padding=0, bias=False),
            nn.Sigmoid())
        self.conv = nn.Conv2d(obs_len, obs_len, 1)
    def forward(self, x):
        x_gap = self.avg_gap(x)
        cattn = self.ca(x_gap)
        x = cattn * self.conv(x) + x
        return x.squeeze()


class SparseGate(nn.Module):
    def __init__(self):
        super(SparseGate, self).__init__()
        self.avg_gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(4, 4, 1),
            nn.Sigmoid())
    def forward(self, x):
        x_avg = self.avg_gap(x)
        threshold = self.conv(x_avg)
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x


class SparseWeightedAdjacency(nn.Module):
    def __init__(self, spa_in_dims=2, embedding_dims=64, dropout=0,):
        super(SparseWeightedAdjacency, self).__init__()
        # edge_inital
        self.edge_inital = Edge_inital(spa_in_dims, embedding_dims)

        # dense interaction
        self.pair_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.group_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.dropout = dropout

        # attention fusion
        self.pair_fusion = GateFusion(obs_len=8)
        self.group_fusion = GateFusion(obs_len=8)

        self.pair_output = nn.Sigmoid()
        self.group_output = nn.Sigmoid()

        self.pair_sparsegate = SparseGate()
        self.group_sparsegate = SparseGate()
    def forward(self, graph, identity, obs_traj):
        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, 1:]  # (T N 2)

        ######### Group definition rules ############
        # ## Distance ##
        dis_graph = obs_traj.unsqueeze(0) - obs_traj.unsqueeze(1)  # (N, N, 2, T)
        dis_graph = dis_graph.permute(3, 0, 1, 2)
        distance = torch.norm(dis_graph, dim=3)
        distance_norm = distance.clamp(min=1e-8)  # 防止模为零
        spatial_distance = torch.where(distance_norm < 2, 1, 0)  # spatial distance threshold = 5
        Distance_G = spatial_distance.to(device)
        Distance_G = Distance_G.float()  # [T N N]

        # ## Speed ##
        speed_diff = spatial_graph.unsqueeze(2) - spatial_graph.unsqueeze(1)  # 计算两个行人之间的速度差 (T, N, N, 2)
        speed_norm = speed_diff.norm(dim=3)  # 计算速度差的模 (T, N, N)
        speed_norm = speed_norm.clamp(min=1e-8)  # 防止模为零
        Speed_G = torch.where(speed_norm < 0.3, 1, 0)
        Speed_G = Speed_G.to(device)
        Speed_G = Speed_G.float()  # [T N N]

        # ## Direction ##
        norms = torch.norm(spatial_graph, dim=-1)
        v_i = spatial_graph.unsqueeze(2)  # [T, N, 1, 2]
        v_j = spatial_graph.unsqueeze(1)  # [T, 1, N, 2]
        dot_products = (v_i * v_j).sum(dim=-1)
        norm_products = norms.unsqueeze(2) * norms.unsqueeze(1)
        cosine_sim = dot_products.clamp(min=1e-8) / norm_products.clamp(min=1e-8)
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        angle_radians = torch.acos(cosine_sim)
        angle_degrees = angle_radians * 180.0 / math.pi
        Direction_G = torch.where(angle_degrees < 30, 1, 0)
        Direction_G = Direction_G.to(device)
        Direction_G = Direction_G.float()  # [T N N]

        G = Distance_G * Speed_G * Direction_G
        ##########################################

        edge_inital = self.edge_inital(spatial_graph, G)  # (T 4 N 16)

        # (T num_heads N N)
        pair_spatial_interaction = self.pair_attention(spatial_graph, edge_inital, G, Pair=True, Group=False)
        group_spatial_interaction = self.group_attention(spatial_graph, edge_inital, G, Pair=False, Group=True)

        pair_interaction = self.pair_fusion(pair_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        group_interaction = self.group_fusion(group_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        pair_interaction = self.pair_output(pair_interaction)
        group_interaction = self.group_output(group_interaction)

        pair_interaction = self.pair_sparsegate(pair_interaction)
        group_interaction = self.group_sparsegate(group_interaction)

        pair_interaction = pair_interaction + identity[0].unsqueeze(1)
        group_interaction = group_interaction + identity[0].unsqueeze(1)

        return pair_interaction, group_interaction, G, edge_inital


class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.edge_value = nn.Linear(embedding_dims, in_dims)
        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, adjacency, G, edge_inital, Pair=False, Group=False):
        # graph=[T, 1, N, 2](seq_len 1 num_p 2)
        if Pair:
            gcn_features = self.embedding(torch.matmul(adjacency, graph))
        if Group:
            div = torch.sum(G, dim=1)[:, None, :, None]
            edge = self.edge_value(edge_inital)
            value = graph + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge) / div
            gcn_features = self.embedding(torch.matmul(adjacency, value))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]



class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout
        self.pair_gcn = nn.ModuleList()
        self.pair_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.group_gcn = nn.ModuleList()
        self.group_gcn.append(GraphConvolution(in_dims, embedding_dims))
    def forward(self, graph, pair_interaction, group_interaction, G, edge_inital):
        # graph [1 seq_len num_pedestrians  3]
        # pair_interaction, group_interaction [batch num_heads seq_len seq_len]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)

        # pair_features [T, heads, N, 16]
        pair_features = self.pair_gcn[0](spa_graph, pair_interaction, G, edge_inital, Pair=True, Group=False)
        pair_features = pair_features.permute(2, 1, 0, 3)

        # group_features [T, heads, N, 16]
        group_features = self.group_gcn[0](spa_graph, group_interaction, G, edge_inital, Pair=False, Group=True)
        group_features = group_features.permute(2, 1, 0, 3)

        return pair_features + group_features


class TrajectoryModel(nn.Module):
    def __init__(self,embedding_dims=64, number_gcn_layers=1, dropout=0,obs_len=8, pred_len=12, n_tcn=5, num_heads=4):
        super(TrajectoryModel, self).__init__()
        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()

        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()))
        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims // num_heads, 2)
        self.multi_output = nn.Sequential(nn.Conv2d(num_heads, 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(16, 20, 1, padding=0),)

    def forward(self, graph, identity, obs_traj):
        # graph 1 obs_len N 3      # obs_traj 1 obs_len N 2
        pair_interaction, group_interaction, G, edge_inital = self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity, obs_traj.squeeze())

        gcn_representation = self.stsgcn(graph, pair_interaction, group_interaction, G, edge_inital)

        gcn_representation = gcn_representation.permute(0, 2, 1, 3)

        features = self.tcns[0](gcn_representation)
        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        prediction = self.output(features)   # prediction=[N, Tpred, nums, 2]
        prediction = self.multi_output(prediction.permute(0, 2, 1, 3))   # prediction=[N, 20, Tpred, 2]

        return prediction.permute(1, 2, 0, 3).contiguous()
