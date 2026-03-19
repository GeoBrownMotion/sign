"""
Training script for PPI node classification (inductive, multi-label).

Models trained:
  1. NodeMLP       - per-node MLP, no message passing
  2. NodeGCN       - GCN without global pooling
  3. Fair MLP K=1,2,3,4 - NodeMLP with hidden matched to SIGN K=1,2,3,4 params
  4. MLP-KD        - NodeMLP trained with knowledge distillation from NodeGCN

Setting:
  - Inductive: 20 train / 2 val / 2 test graphs
  - Multi-label: 121 binary labels per node
  - Loss: BCEWithLogitsLoss
  - Metric: micro-F1
  - SIGN features precomputed per graph (inductive-safe)
"""

import json
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree

from models import NodeMLP, NodeGCN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED   = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR  = "/tmp/ppi"
SAVE_ROOT = "/home/hep3/sign/results/ppi"

HPARAMS = {
    "lr":           0.005,
    "weight_decay": 0.0,
    "max_epochs":   2000,
    "patience":     50,       # in eval_freq units
    "eval_freq":    5,
    "batch_size":   2,        # graphs per batch
}

MLP_HIDDEN  = 512
MLP_ACT     = "relu"
MLP_DROPOUT = 0.0

GCN_HIDDEN  = 256
GCN_CONV    = 3
GCN_LIN     = 1
GCN_ACT     = "relu"
GCN_DROPOUT = 0.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# SIGN precomputation (per graph, inductive-safe)
# ---------------------------------------------------------------------------

def precompute_sign_graph(data, K):
    """Precompute [X, AX, ..., A^K X] for a single graph. Returns [N, (K+1)*F]."""
    N = data.num_nodes
    edge_index = data.edge_index

    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=N)
    row, col = edge_index_sl
    deg     = degree(col, N, dtype=torch.float)
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv == float("inf")] = 0.0
    norm    = deg_inv[row] * deg_inv[col]

    adj_t = torch.sparse_coo_tensor(
        torch.stack([col, row]), norm, (N, N)
    ).to(data.x.device)

    xs = [data.x]
    x  = data.x
    for _ in range(K):
        x = torch.sparse.mm(adj_t, x)
        xs.append(x)
    return torch.cat(xs, dim=1)


def make_sign_dataset(dataset, K):
    """Return a new list of Data objects with SIGN features replacing x."""
    out = []
    for data in dataset:
        d = data.clone()
        d.x = precompute_sign_graph(data, K)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    for data in loader:
        data = data.to(DEVICE)
        out  = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        total_loss += loss.item()
        preds.append((out.sigmoid() > 0.5).cpu().numpy())
        labels.append(data.y.cpu().numpy())
    preds  = np.concatenate(preds,  axis=0)
    labels = np.concatenate(labels, axis=0)
    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return total_loss / len(loader), f1


def run_training(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])
    best_loss, best_f1, patience_count, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_epoch(model, train_loader, optimizer)

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_f1 = evaluate(model, val_loader)
            if val_loss < best_loss:
                best_loss, best_f1, patience_count = val_loss, val_f1, 0
                best_state = deepcopy(model.state_dict())
            else:
                patience_count += 1
            if patience_count >= HPARAMS["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_f1


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _set_act(layers, layer):
    if isinstance(layer, torch.nn.Tanh) and layers:
        layers[-1]["act"] = "tanh"
    elif isinstance(layer, torch.nn.ReLU) and layers:
        layers[-1]["act"] = "relu"


def export_node_model(name, model, save_dir, in_channels):
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

def save_results(name, model, val_f1, extra_hparams, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    results = {"model": name, "val_f1": round(val_f1, 6),
               "seed": SEED, "hparams": {**HPARAMS, **extra_hparams}}
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  val_f1 = {val_f1:.4f}  →  {save_dir}/")


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def fair_mlp_hidden(sign_params, in_ch, layers=2, out_ch=121):
    # params = in_ch*h + h + (layers-1)*(h^2+h) + h*out_ch + out_ch
    a = layers - 1
    b = in_ch + layers + out_ch
    c = out_ch - sign_params
    if a == 0:
        return max(int((sign_params - out_ch) / b), 1)
    h = int((-b + math.sqrt(max(b**2 - 4*a*c, 0))) / (2*a))
    return max(h, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    train_ds = PPI(DATA_DIR, split="train")
    val_ds   = PPI(DATA_DIR, split="val")

    in_ch  = train_ds[0].x.shape[1]   # 50
    out_ch = train_ds[0].y.shape[1]   # 121

    print(f"PPI | in={in_ch}  out={out_ch}  (multi-label)")
    print(f"  train graphs={len(train_ds)}  val graphs={len(val_ds)}")
    print(f"Device: {DEVICE}\n")

    train_loader = DataLoader(train_ds, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=HPARAMS["batch_size"], shuffle=False)

    # ------------------------------------------------------------------
    # 1. NodeMLP
    # ------------------------------------------------------------------
    print("=" * 50)
    print("1. NodeMLP")
    set_seed(SEED)
    mlp = NodeMLP(in_ch, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                  act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
    mlp, f1 = run_training(mlp, train_loader, val_loader)
    d = os.path.join(SAVE_ROOT, "mlp")
    save_results("node_mlp", mlp, f1,
                 {"hidden": MLP_HIDDEN, "layers": 2, "act": MLP_ACT}, d)
    export_node_model("node_mlp", mlp, d, in_ch)

    # ------------------------------------------------------------------
    # 2. NodeGCN
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. NodeGCN")
    set_seed(SEED)
    gcn = NodeGCN(in_ch, GCN_HIDDEN, out_ch, GCN_CONV, GCN_LIN,
                  act=GCN_ACT, dropout=GCN_DROPOUT).to(DEVICE)
    gcn, f1 = run_training(gcn, train_loader, val_loader)
    d = os.path.join(SAVE_ROOT, "gcn")
    save_results("node_gcn", gcn, f1,
                 {"hidden": GCN_HIDDEN, "conv_layers": GCN_CONV,
                  "lin_layers": GCN_LIN, "act": GCN_ACT}, d)
    export_node_model("node_gcn", gcn, d, in_ch)

    # ------------------------------------------------------------------
    # 3. Fair MLP K=1,2,3,4
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. Fair MLP (params ≈ SIGN K=1/2/3/4)")
    for K in [1, 2, 3, 4]:
        sign_in  = (K + 1) * in_ch
        sign_ref = NodeMLP(sign_in, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                           act=MLP_ACT, dropout=MLP_DROPOUT)
        sign_params = count_params(sign_ref)
        fair_h = fair_mlp_hidden(sign_params, in_ch=in_ch, layers=2, out_ch=out_ch)
        print(f"  K={K}  SIGN params={sign_params:,}  →  Fair MLP hidden={fair_h}")

        set_seed(SEED)
        fair_mlp = NodeMLP(in_ch, fair_h, out_ch, num_hidden_layers=2,
                           act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
        fair_mlp, f1 = run_training(fair_mlp, train_loader, val_loader)
        d = os.path.join(SAVE_ROOT, f"mlp_fair_k{K}")
        save_results(f"node_mlp_fair_k{K}", fair_mlp, f1,
                     {"K": K, "hidden": fair_h, "layers": 2, "sign_params": sign_params}, d)
        export_node_model(f"node_mlp_fair_k{K}", fair_mlp, d, in_ch)

    # ------------------------------------------------------------------
    # 4. MLP-KD (distill from NodeGCN)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. MLP-KD  (T=4, lambda=0.5)")
    LAM, TEMP = 0.5, 4.0
    gcn.eval()

    set_seed(SEED)
    student  = NodeMLP(in_ch, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                       act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(),
                                 lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])
    best_loss, best_f1, patience_count, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        student.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out     = student(data.x, data.edge_index)
            loss_ce = F.binary_cross_entropy_with_logits(out, data.y)
            with torch.no_grad():
                t_soft = torch.sigmoid(gcn(data.x, data.edge_index) / TEMP)
            loss_kd = F.binary_cross_entropy(
                torch.sigmoid(out / TEMP), t_soft
            )
            (LAM * loss_ce + (1 - LAM) * loss_kd).backward()
            optimizer.step()

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_f1 = evaluate(student, val_loader)
            if val_loss < best_loss:
                best_loss, best_f1, patience_count = val_loss, val_f1, 0
                best_state = deepcopy(student.state_dict())
            else:
                patience_count += 1
            if patience_count >= HPARAMS["patience"]:
                break

    student.load_state_dict(best_state)
    d = os.path.join(SAVE_ROOT, "mlp_kd")
    save_results("node_mlp_kd", student, best_f1,
                 {"lambda": LAM, "temperature": TEMP, "hidden": MLP_HIDDEN}, d)
    export_node_model("node_mlp_kd", student, d, in_ch)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Summary (PPI val micro-F1)")
    print("=" * 50)
    for subdir in ["mlp", "gcn", "mlp_fair_k1", "mlp_fair_k2",
                   "mlp_fair_k3", "mlp_fair_k4", "mlp_kd"]:
        rpath = os.path.join(SAVE_ROOT, subdir, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            print(f"  {subdir:<14}  {r['val_f1']:.4f}")


if __name__ == "__main__":
    main()
