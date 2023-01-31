import torch.nn as nn
import torch


class HypergraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, hypergraph):
        X = self.theta(X)
        Y = hypergraph.v2e(X, aggr="mean")
        X_ = hypergraph.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_

class CascadeConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True, drop_rate = 0.5):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, hypergraph):
        X = self.theta(X)
        Y = hypergraph.v2e(X, aggr="mean")
        X_ = hypergraph.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_


class ConHypergraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5, hg_group=3):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)
        self.group_weight = nn.Parameter(torch.FloatTensor(hg_group))

    def forward(self, X, hypergraph):
        edge_num = []
        for name in hypergraph.group_names:
            edge_num.append(hypergraph.num_e_of_group(name))
        weight = torch.cat([self.group_weight[i].expand(edge_num[i]) for i in range(len(edge_num))], dim=0)

        X = self.theta(X)
        X_ = hypergraph.v2v(X, aggr="mean")
        X_ = self.act(X_)
        X_ = self.drop(X_)

        return X_