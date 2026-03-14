"""
Model definitions for the utility-verification trade-off study.

Models:
  - MLP: global mean-pool node features, then stack of Linear+tanh layers.
  - GCN: stack of GCNConv+tanh layers, then global mean-pool, then linear head.

Activation is always tanh so that the polynomial-zonotope verifier (CORA) can
handle the non-linearities exactly as in the reference paper.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class MLP(nn.Module):
    """
    Graph-level MLP baseline.

    Inference pipeline (no message passing):
        global_mean_pool(node features)  →  [Linear → tanh] x num_hidden_layers  →  Linear

    For verification: the ε-ball perturbation on all node features propagates
    linearly through mean-pooling before entering the MLP, so the verifier sees
    a plain feedforward network with a (possibly scaled) input set.

    Args:
        in_channels:       Number of input node features.
        hidden_channels:   Width of every hidden layer.
        out_channels:      Number of output classes.
        num_hidden_layers: Number of hidden Linear+tanh layers (default 3,
                           matching the 3 GCN layers in the GNN baseline).
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
        # first hidden layer: in_channels → hidden_channels
        layers += [nn.Linear(in_channels, hidden_channels), nn.Tanh()]
        # remaining hidden layers
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_channels, hidden_channels), nn.Tanh()]
        # output layer (no activation — cross-entropy loss is applied outside)
        layers += [nn.Linear(hidden_channels, out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index, batch):
        # pool first, then apply MLP
        x = global_mean_pool(x, batch)   # (num_graphs, in_channels)
        return self.net(x)               # (num_graphs, out_channels)


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

        # Linear layers after pooling
        lin_list = []
        for i in range(num_lin_layers - 1):
            lin_list += [nn.Linear(hidden_channels, hidden_channels), nn.Tanh()]
        # final output layer
        lin_list.append(nn.Linear(hidden_channels, out_channels))
        self.lins = nn.Sequential(*lin_list)

        self.act = nn.Tanh()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        x = global_mean_pool(x, batch)   # (num_graphs, hidden_channels)
        return self.lins(x)              # (num_graphs, out_channels)
