import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINConv,
    MessagePassing,
    global_add_pool,
    global_mean_pool,
)


class GINConv(MessagePassing):
    def __init__(self, input_dim, hidden_dim, epsilon=0.0, train_epsilon=False):
        # 和集約を選択
        super(GINConv, self).__init__(aggr="add")

        if train_epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([epsilon]))
        else:
            self.register_buffer("epsilon", torch.Tensor([epsilon]))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x, edge_idx):

        out = self.propagate(edge_idx, x)

        out = (1 + self.epsilon) * x + out

        out = self.mlp(out)

        return out

    def message(self, x_j):
        return x_j


class GIN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        epsilon=0.0,
        train_epsilon=False,
        pooling="sum",
    ):
        super(GIN, self).__init__()

        # num of GIN layers
        self.num_layers = num_layers
        # final pooling option
        self.pooling = pooling

        self.gin_convs = nn.ModuleList()

        # init of convolutoin layers in GIN
        self.gin_convs.append(GINConv(input_dim, hidden_dim, epsilon, train_epsilon))
        for _ in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(hidden_dim, hidden_dim, epsilon, train_epsilon)
            )

        # clasifier of model
        self.predector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for gin_conv in self.gin_convs:
            x = gin_conv(x, edge_index)

        if self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        x = self.predector(x)

        return x
