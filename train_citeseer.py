"""
Training script for CiteSeer node classification.

Models trained:
  1. NodeMLP       - per-node MLP, no message passing
  2. NodeGCN       - GCN without global pooling
  3. SIGN K=1,2,3  - precomputed K-hop features + NodeMLP
  4. Fair MLP      - NodeMLP with increased hidden to match SIGN K=2 params
  5. MLP-KD        - NodeMLP trained with knowledge distillation from NodeGCN

All models use tanh activation (CORA verifier compatible).
Standard Planetoid split: 120 train / 500 val / 1000 test nodes.
"""

import argparse
import json
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_self_loops, degree

from models import NodeMLP, NodeGCN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED   = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR  = "/tmp/citeseer"
SAVE_ROOT = "/home/hep3/sign/results/citeseer"

HPARAMS = {
    "lr":          0.01,
    "weight_decay": 5e-4,
    "dropout":     0.5,
    "max_epochs":  10000,
    "patience":    20,        # in eval_freq units
    "eval_freq":   10,
}

MLP_HIDDEN   = 64
MLP_ACT      = "relu"
MLP_DROPOUT  = 0.2
GCN_HIDDEN   = 64
GCN_CONV     = 2
GCN_LIN      = 2
GCN_ACT      = "relu"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading & SIGN precomputation
# ---------------------------------------------------------------------------

def load_citeseer():
    ds   = Planetoid(root=DATA_DIR, name="CiteSeer")
    data = ds[0]
    # CiteSeer features are bag-of-words in [0,1] — no additional normalization needed.
    feat_mean = torch.zeros(data.x.shape[1])
    feat_std  = torch.ones(data.x.shape[1])
    return data, feat_mean, feat_std


def precompute_sign(data, K):
    """Return tensor of shape [N, (K+1)*F] with [X, AX, A²X, …, AᴷX] concatenated."""
    N    = data.num_nodes
    edge_index = data.edge_index

    # symmetric normalised adjacency  D^{-1/2} A D^{-1/2}
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=N)
    row, col = edge_index_sl
    deg      = degree(col, N, dtype=torch.float)
    deg_inv  = deg.pow(-0.5)
    deg_inv[deg_inv == float("inf")] = 0.0
    norm     = deg_inv[row] * deg_inv[col]

    # Build dense-sparse aggregation using PyG scatter
    adj_t = torch.sparse_coo_tensor(
        torch.stack([col, row]), norm, (N, N)
    ).to(data.x.device)

    xs = [data.x]
    x  = data.x
    for _ in range(K):
        x = torch.sparse.mm(adj_t, x)
        xs.append(x)
    return torch.cat(xs, dim=1)   # [N, (K+1)*F]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
    acc  = (out[data.val_mask].argmax(dim=-1) == data.y[data.val_mask]).float().mean().item()
    return loss, acc


def run_training(model, data, lr, wd):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_loss, best_acc, patience_count, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_epoch(model, data, optimizer)

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_acc = evaluate(model, data)
            if val_loss < best_loss:
                best_loss, best_acc, patience_count = val_loss, val_acc, 0
                best_state = deepcopy(model.state_dict())
            else:
                patience_count += 1
            if patience_count >= HPARAMS["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_acc


# ---------------------------------------------------------------------------
# Export helper (node-level, no pooling)
# ---------------------------------------------------------------------------

def _set_act(layers, layer):
    if isinstance(layer, torch.nn.Tanh) and layers:
        layers[-1]["act"] = "tanh"
    elif isinstance(layer, torch.nn.ReLU) and layers:
        layers[-1]["act"] = "relu"


def export_node_model(name, model, save_dir, in_channels):
    """Export node-level model to model_export.json for CORA verifier."""
    layers = []

    if isinstance(model, NodeGCN):
        gcn_act = "relu" if isinstance(model.act, torch.nn.ReLU) else "tanh"
        for conv in model.convs:
            W = conv.lin.weight.detach().cpu()
            b = conv.bias.detach().cpu() if conv.bias is not None else torch.zeros(W.shape[0])
            layers.append({"type": "gcn", "act": gcn_act,
                           "W": W.tolist(), "b": b.tolist()})
        for layer in model.lins:
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().cpu()
                b = layer.bias.detach().cpu()
                layers.append({"type": "lin", "act": "", "W": W.tolist(), "b": b.tolist()})
            else:
                _set_act(layers, layer)
    else:
        # NodeMLP (or SIGN MLP)
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().cpu()
                b = layer.bias.detach().cpu()
                layers.append({"type": "lin", "act": "", "W": W.tolist(), "b": b.tolist()})
            else:
                _set_act(layers, layer)

    export = {"model": name, "layers": layers}
    path   = os.path.join(save_dir, "model_export.json")
    with open(path, "w") as f:
        json.dump(export, f)

    print(f"  Exported {name} → {path}")
    for l in layers:
        W = l["W"]
        rows = len(W)
        cols = len(W[0]) if rows > 0 else 0
        print(f"    type={l['type']:<6} act={l['act']:<6} W=({rows}×{cols})")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_results(name, model, val_acc, feat_mean, feat_std, extra_hparams, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "norm_stats.json"), "w") as f:
        json.dump({"feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist()}, f, indent=2)
    results = {"model": name, "val_acc": round(val_acc, 6),
               "seed": SEED, "hparams": {**HPARAMS, **extra_hparams}}
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  val_acc = {val_acc:.4f}  →  {save_dir}/")


# ---------------------------------------------------------------------------
# Helper: param count
# ---------------------------------------------------------------------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def fair_mlp_hidden(sign_params, in_ch=3703, layers=2):
    """Solve for hidden h such that NodeMLP(in_ch, h, 6, layers) ≈ sign_params."""
    # params = in_ch*h + h + (layers-1)*(h*h+h) + h*6 + 6
    # ≈ (layers-1)*h² + (in_ch + (layers-1) + 6 + 1)*h
    a = layers - 1
    b = in_ch + layers + 6
    c = 6 - sign_params
    if a == 0:
        return int((sign_params - 6) / b)
    h = int((-b + math.sqrt(b**2 - 4*a*c)) / (2*a))
    return max(h, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    data, feat_mean, feat_std = load_citeseer()
    data = data.to(DEVICE)
    in_ch = data.x.shape[1]   # 3703
    out_ch = 6

    print(f"CiteSeer | nodes={data.num_nodes}  in={in_ch}  out={out_ch}")
    print(f"  train={data.train_mask.sum().item()}  val={data.val_mask.sum().item()}  "
          f"test={data.test_mask.sum().item()}")
    print(f"Device: {DEVICE}\n")

    # ------------------------------------------------------------------
    # 1. NodeMLP
    # ------------------------------------------------------------------
    print("=" * 50)
    print("1. NodeMLP")
    set_seed(SEED)
    mlp = NodeMLP(in_ch, MLP_HIDDEN, out_ch, num_hidden_layers=2, act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
    mlp, acc = run_training(mlp, data, HPARAMS["lr"], HPARAMS["weight_decay"])
    d = os.path.join(SAVE_ROOT, "mlp")
    save_results("node_mlp", mlp, acc, feat_mean, feat_std,
                 {"hidden": MLP_HIDDEN, "layers": 2, "act": MLP_ACT}, d)
    export_node_model("node_mlp", mlp, d, in_ch)

    # ------------------------------------------------------------------
    # 2. NodeGCN
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. NodeGCN")
    set_seed(SEED)
    gcn = NodeGCN(in_ch, GCN_HIDDEN, out_ch, GCN_CONV, GCN_LIN, act=GCN_ACT, dropout=HPARAMS["dropout"]).to(DEVICE)
    gcn, acc = run_training(gcn, data, HPARAMS["lr"], HPARAMS["weight_decay"])
    d = os.path.join(SAVE_ROOT, "gcn")
    save_results("node_gcn", gcn, acc, feat_mean, feat_std,
                 {"hidden": GCN_HIDDEN, "conv_layers": GCN_CONV}, d)
    export_node_model("node_gcn", gcn, d, in_ch)

    # ------------------------------------------------------------------
    # 3. Fair MLP K=1,2,3,4 (params match SIGN K=1,2,3,4; no SIGN models saved)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. Fair MLP (params ≈ SIGN K=1/2/3/4)")
    for K in [1, 2, 3, 4]:
        sign_in = (K + 1) * in_ch          # input dim of the corresponding SIGN model
        sign_ref = NodeMLP(sign_in, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                           act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
        sign_params = count_params(sign_ref)
        fair_h = fair_mlp_hidden(sign_params, in_ch=in_ch, layers=2)
        print(f"  K={K}  SIGN params={sign_params:,}  →  Fair MLP hidden={fair_h}")

        set_seed(SEED)
        fair_mlp = NodeMLP(in_ch, fair_h, out_ch, num_hidden_layers=2,
                           act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
        fair_mlp, acc = run_training(fair_mlp, data, HPARAMS["lr"], HPARAMS["weight_decay"])
        d = os.path.join(SAVE_ROOT, f"mlp_fair_k{K}")
        save_results(f"node_mlp_fair_k{K}", fair_mlp, acc, feat_mean, feat_std,
                     {"K": K, "hidden": fair_h, "layers": 2, "sign_params": sign_params}, d)
        export_node_model(f"node_mlp_fair_k{K}", fair_mlp, d, in_ch)

    # ------------------------------------------------------------------
    # 5. MLP-KD (distill from NodeGCN)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("5. MLP-KD  (T=4, lambda=0.5)")
    LAM, TEMP = 0.5, 4.0
    gcn.eval()

    set_seed(SEED)
    student = NodeMLP(in_ch, MLP_HIDDEN, out_ch, num_hidden_layers=2, act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(),
                                  lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"])
    best_loss, best_acc, patience_count, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        student.train()
        optimizer.zero_grad()
        out = student(data.x, data.edge_index)
        loss_ce = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        with torch.no_grad():
            t_soft = F.softmax(gcn(data.x, data.edge_index)[data.train_mask] / TEMP, dim=-1)
        loss_kd = F.kl_div(F.log_softmax(out[data.train_mask] / TEMP, dim=-1),
                            t_soft, reduction="batchmean") * (TEMP ** 2)
        (LAM * loss_ce + (1 - LAM) * loss_kd).backward()
        optimizer.step()

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_acc = evaluate(student, data)
            if val_loss < best_loss:
                best_loss, best_acc, patience_count = val_loss, val_acc, 0
                best_state = deepcopy(student.state_dict())
            else:
                patience_count += 1
            if patience_count >= HPARAMS["patience"]:
                break

    student.load_state_dict(best_state)
    d = os.path.join(SAVE_ROOT, "mlp_kd")
    save_results("node_mlp_kd", student, best_acc, feat_mean, feat_std,
                 {"lambda": LAM, "temperature": TEMP, "hidden": MLP_HIDDEN}, d)
    export_node_model("node_mlp_kd", student, d, in_ch)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Summary (CiteSeer val acc)")
    print("=" * 50)
    for subdir in ["mlp", "gcn", "mlp_fair_k1", "mlp_fair_k2", "mlp_fair_k3", "mlp_fair_k4", "mlp_kd"]:
        rpath = os.path.join(SAVE_ROOT, subdir, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            print(f"  {subdir:<12}  {r['val_acc']:.4f}")


if __name__ == "__main__":
    main()
