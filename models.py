"""
Model definitions for the utility-verification trade-off study.

Graph-level models (ENZYMES):
  - MLP:        global_mean_pool → [Linear+act] × num_layers  (no message passing, pool-first)
  - MLPPerNode: [Linear+act] × num_layers per node → global_mean_pool  (no message passing, pool-last)
  - GCN:        [GCNConv+act] → [Linear+act] per node → global_mean_pool

Node-level models (CiteSeer):
  - NodeMLP:    [Linear+act] × num_layers per node  (no message passing, no pooling)
  - NodeGCN:    [GCNConv+act] × num_conv_layers → [Linear+act] × num_lin_layers per node

Supported activations: 'tanh', 'relu' (both supported by CORA verifier).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


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


class MLPPerNode(nn.Module):
    """
    Graph-level MLP baseline (no message passing).

    Pipeline: [Linear+tanh] x num_hidden_layers per node → global_mean_pool → Linear

    Applies MLP independently to each node, then pools. No message passing.

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
        x = self.net(x)                    # per-node MLP
        return global_mean_pool(x, batch)  # pool after MLP


def _make_act(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {act!r}. Use 'tanh' or 'relu'.")


class GCN(nn.Module):
    """
    Graph-level GCN with node features only.

    Architecture matches the reference paper (Ladner et al., 2025) for Enzymes:
        [GCNConv → act] x num_conv_layers → [Linear → act] per node → global_mean_pool

    Args:
        in_channels:     Number of input node features.
        hidden_channels: Width of every hidden layer.
        out_channels:    Number of output classes.
        num_conv_layers: Number of GCN message-passing layers (default 3).
        num_lin_layers:  Number of linear layers after pooling, counting the
                         final output layer (default 1 → just the output layer).
        act:             Activation function: 'tanh' (default) or 'relu'.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int = 3,
        num_lin_layers: int = 1,
        act: str = "tanh",
    ):
        super().__init__()
        self.act = _make_act(act)

        conv_list = []
        for i in range(num_conv_layers):
            c_in = in_channels if i == 0 else hidden_channels
            conv_list.append(GCNConv(c_in, hidden_channels))
        self.convs = nn.ModuleList(conv_list)

        lin_list = []
        for i in range(num_lin_layers - 1):
            lin_list += [nn.Linear(hidden_channels, hidden_channels), _make_act(act)]
        lin_list += [nn.Linear(hidden_channels, out_channels), _make_act(act)]
        self.lins = nn.Sequential(*lin_list)
        # NOTE: output_act=True is kept for tanh (matches reference paper architecture).
        # For relu, use GCNNoOutAct or set output_act=False.

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        x = self.lins(x)
        return global_mean_pool(x, batch)


class GCNNoOutAct(GCN):
    """GCN variant with no activation on the output linear layer (for ReLU experiments)."""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_conv_layers=3, num_lin_layers=1, act="relu"):
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_conv_layers, num_lin_layers, act)
        # Replace last two elements (Linear + act) with just Linear
        layers = list(self.lins.children())
        self.lins = nn.Sequential(*layers[:-1])


class GIN(nn.Module):
    """
    Graph-level GIN (Graph Isomorphism Network, Xu et al. 2019).

    Architecture:
        [GINConv(Linear→act→Linear)] × num_conv_layers → [Linear→act] per node → global_mean_pool

    Each GINConv uses a 2-layer MLP internally (more expressive than GCN's single linear).
    Sum aggregation (vs GCN's normalized mean) gives strictly stronger expressiveness.

    Args:
        in_channels:     Number of input node features.
        hidden_channels: Width of every hidden layer.
        out_channels:    Number of output classes.
        num_conv_layers: Number of GIN message-passing layers (default 3).
        num_lin_layers:  Number of linear layers after conv, including output layer (default 1).
        act:             Activation function: 'tanh' (default) or 'relu'.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int = 3,
        num_lin_layers: int = 1,
        act: str = "tanh",
    ):
        super().__init__()

        conv_list = []
        for i in range(num_conv_layers):
            c_in = in_channels if i == 0 else hidden_channels
            inner_mlp = nn.Sequential(
                nn.Linear(c_in, hidden_channels),
                _make_act(act),
                nn.Linear(hidden_channels, hidden_channels),
            )
            conv_list.append(GINConv(inner_mlp, train_eps=True))
        self.convs = nn.ModuleList(conv_list)
        self.act = _make_act(act)

        lin_list = []
        for _ in range(num_lin_layers - 1):
            lin_list += [nn.Linear(hidden_channels, hidden_channels), _make_act(act)]
        lin_list += [nn.Linear(hidden_channels, out_channels), _make_act(act)]
        self.lins = nn.Sequential(*lin_list)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        x = self.lins(x)
        return global_mean_pool(x, batch)


class GINNoOutAct(GIN):
    """GIN variant with no activation on the output linear layer (for ReLU experiments)."""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_conv_layers=3, num_lin_layers=1, act="relu"):
        super().__init__(in_channels, hidden_channels, out_channels,
                         num_conv_layers, num_lin_layers, act)
        layers = list(self.lins.children())
        self.lins = nn.Sequential(*layers[:-1])


# ---------------------------------------------------------------------------
# Node-level models (CiteSeer, transductive node classification)
# ---------------------------------------------------------------------------

class NodeMLP(nn.Module):
    """
    Node-level MLP baseline (no message passing).

    Pipeline: Dropout → [Linear+act+Dropout] × num_hidden_layers → Linear
    Applied independently to each node. No graph structure used.

    Args:
        in_channels:       Number of input node features.
        hidden_channels:   Width of hidden layers.
        out_channels:      Number of output classes.
        num_hidden_layers: Number of hidden Linear+act layers (default 2).
        act:               Activation: 'tanh' (default) or 'relu'.
        dropout:           Dropout rate (default 0.0).
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_hidden_layers=2, act="tanh", dropout=0.0):
        super().__init__()
        self.dropout = dropout
        # No dropout on raw input features — only between hidden layers
        layers = [nn.Linear(in_channels, hidden_channels), _make_act(act)]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels), _make_act(act)]
        layers += [nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, batch=None):
        return self.net(x)


class NodeGCN(nn.Module):
    """
    Node-level GCN (no global pooling).

    Pipeline: [Dropout→GCNConv+act] × num_conv_layers → Linear

    Args:
        in_channels:     Number of input node features.
        hidden_channels: Width of every hidden layer.
        out_channels:    Number of output classes.
        num_conv_layers: Number of GCN message-passing layers (default 2).
        num_lin_layers:  Number of linear layers after conv, including output (default 1).
        act:             Activation: 'tanh' (default) or 'relu'.
        dropout:         Dropout rate (default 0.0).
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_conv_layers=2, num_lin_layers=1, act="tanh", dropout=0.0):
        super().__init__()
        self.act = _make_act(act)
        self.dropout = dropout

        conv_list = []
        for i in range(num_conv_layers):
            c_in = in_channels if i == 0 else hidden_channels
            conv_list.append(GCNConv(c_in, hidden_channels))
        self.convs = nn.ModuleList(conv_list)

        lin_list = []
        for _ in range(num_lin_layers - 1):
            lin_list += [nn.Linear(hidden_channels, hidden_channels), _make_act(act)]
        lin_list += [nn.Linear(hidden_channels, out_channels)]
        self.lins = nn.Sequential(*lin_list)

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins(x)
