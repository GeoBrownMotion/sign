"""
Model definitions for the utility-verification trade-off study.

Models:
  - MLP: global_mean_pool → [Linear+tanh] × num_layers  (no message passing)
  - GCN: [GCNConv+tanh] → [Linear+tanh] per node → global_mean_pool

MLP uses pool-first to completely ignore graph structure (standard baseline).
GCN matches the reference paper export format (Ladner et al., 2025).
Activation is always tanh so that the CORA polynomial-zonotope verifier can
handle non-linearities exactly.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class MLP(nn.Module):
    """
    Graph-level MLP baseline (no message passing).

    Pipeline: global_mean_pool → [Linear+tanh] x num_hidden_layers → Linear

    Completely ignores graph structure; operates only on mean-pooled node features.

    Args:
        in_channels:       Number of input node features.
        hidden_channels:   Width of hidden layers.
        out_channels:      Number of output classes.
        num_hidden_layers: Number of hidden Linear+tanh layers (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_hidden_layers: int = 3,
    ):
        super().__init__()
        layers = []
        layers += [nn.Linear(in_channels, hidden_channels), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_channels, hidden_channels), nn.Tanh()]
        layers += [nn.Linear(hidden_channels, out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index, batch):
        x = global_mean_pool(x, batch)   # pool first, ignore graph structure
        return self.net(x)


class GCN(nn.Module):
    """
    Graph-level GCN with node features only.

    Architecture matches the reference paper (Ladner et al., 2025) for Enzymes:
        [GCNConv → tanh] x num_conv_layers  →  global_mean_pool  →  [Linear → tanh] x (num_lin_layers-1)  →  Linear

    Args:
        in_channels:     Number of input node features.
        hidden_channels: Width of every hidden layer.
        out_channels:    Number of output classes.
        num_conv_layers: Number of GCN message-passing layers (default 3).
        num_lin_layers:  Number of linear layers after pooling, counting the
                         final output layer (default 1 → just the output layer).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int = 3,
        num_lin_layers: int = 1,
    ):
        super().__init__()
        # GCN layers
        conv_list = []
        for i in range(num_conv_layers):
            c_in = in_channels if i == 0 else hidden_channels
            conv_list.append(GCNConv(c_in, hidden_channels))
        self.convs = nn.ModuleList(conv_list)

        # Linear layers applied per-node (before pooling), all with tanh
        lin_list = []
        for i in range(num_lin_layers - 1):
            lin_list += [nn.Linear(hidden_channels, hidden_channels), nn.Tanh()]
        lin_list += [nn.Linear(hidden_channels, out_channels), nn.Tanh()]
        self.lins = nn.Sequential(*lin_list)

        self.act = nn.Tanh()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        x = self.lins(x)                 # per-node linear+tanh
        return global_mean_pool(x, batch)  # pool after linear
