"""
CiteSeer with PCA preprocessing (3703 → 32 dims).

PCA is fit on training nodes only (120 nodes), then applied to all nodes.
Models trained:
  1. NodeMLP            - per-node MLP, no message passing
  2. NodeGCN            - GCN without global pooling
  3. Fair MLP K=1..4    - NodeMLP with hidden matched to SIGN K=1..4 params
  4. MLP-KD             - NodeMLP distilled from NodeGCN teacher
"""

import json
import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid

from models import NodeMLP, NodeGCN


SEED      = 8
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = "/tmp/citeseer"
SAVE_ROOT = "/home/hep3/sign/results/citeseer_pca"
PCA_DIM   = 32

HPARAMS = {
    "lr":           0.01,
    "weight_decay": 5e-4,
    "max_epochs":   10000,
    "patience":     20,
    "eval_freq":    10,
}

MLP_HIDDEN  = 64
MLP_ACT     = "relu"
MLP_DROPOUT = 0.2
GCN_HIDDEN  = 64
GCN_CONV    = 2
GCN_LIN     = 2
GCN_ACT     = "relu"
GCN_DROPOUT = 0.5


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_citeseer_pca():
    ds   = Planetoid(root=DATA_DIR, name="CiteSeer")
    data = ds[0]

    x_np   = data.x.numpy()                    # (3327, 3703)
    x_train = x_np[data.train_mask.numpy()]    # (120, 3703) — fit PCA here only

    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(x_train)

    x_reduced = pca.transform(x_np)            # (3327, 32)
    data.x = torch.tensor(x_reduced, dtype=torch.float32)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: 3703 → {PCA_DIM} dims  (explained variance: {explained:.3f})")

    return data, pca


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
    acc  = (out[data.val_mask].argmax(-1) == data.y[data.val_mask]).float().mean().item()
    return loss, acc


def run_training(model, data):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"])
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


def _set_act(layers, layer):
    if isinstance(layer, torch.nn.ReLU) and layers:
        layers[-1]["act"] = "relu"
    elif isinstance(layer, torch.nn.Tanh) and layers:
        layers[-1]["act"] = "tanh"


def export_model(name, model, save_dir):
    layers = []
    if isinstance(model, NodeGCN):
        act_str = "relu" if isinstance(model.act, torch.nn.ReLU) else "tanh"
        for conv in model.convs:
            W = conv.lin.weight.detach().cpu()
            b = conv.bias.detach().cpu() if conv.bias is not None else torch.zeros(W.shape[0])
            layers.append({"type": "gcn", "act": act_str, "W": W.tolist(), "b": b.tolist()})
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

    path = os.path.join(save_dir, "model_export.json")
    with open(path, "w") as f:
        json.dump({"model": name, "layers": layers}, f)
    print(f"  Exported {name} → {path}")
    for l in layers:
        rows = len(l["W"]); cols = len(l["W"][0]) if rows > 0 else 0
        print(f"    type={l['type']:<6} act={l['act']:<6} W=({rows}×{cols})")


def save_results(name, model, val_acc, extra, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    results = {"model": name, "val_acc": round(val_acc, 6), "seed": SEED,
               "pca_dim": PCA_DIM, "hparams": {**HPARAMS, **extra}}
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  val_acc = {val_acc:.4f}  →  {save_dir}/")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def fair_mlp_hidden(sign_params, in_ch, layers=2, out_ch=6):
    """Solve for hidden h such that NodeMLP(in_ch, h, out_ch, layers) ≈ sign_params."""
    a = layers - 1
    b = in_ch + layers + out_ch
    c = out_ch - sign_params
    if a == 0:
        return max(int((sign_params - out_ch) / b), 1)
    h = int((-b + math.sqrt(max(b**2 - 4*a*c, 0))) / (2*a))
    return max(h, 1)


if __name__ == "__main__":
    set_seed(SEED)
    data, pca = load_citeseer_pca()
    data = data.to(DEVICE)
    in_ch, out_ch = PCA_DIM, 6

    print(f"CiteSeer (PCA) | in={in_ch}  out={out_ch}  device={DEVICE}\n")

    # ------------------------------------------------------------------
    # 1. NodeMLP
    # ------------------------------------------------------------------
    print("=" * 50)
    print("1. NodeMLP")
    set_seed(SEED)
    mlp = NodeMLP(in_ch, MLP_HIDDEN, out_ch,
                  num_hidden_layers=2, act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
    mlp, acc = run_training(mlp, data)
    d = os.path.join(SAVE_ROOT, "mlp")
    save_results("node_mlp_pca", mlp, acc,
                 {"hidden": MLP_HIDDEN, "act": MLP_ACT, "dropout": MLP_DROPOUT}, d)
    export_model("node_mlp_pca", mlp, d)

    # ------------------------------------------------------------------
    # 2. NodeGCN
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. NodeGCN")
    set_seed(SEED)
    gcn = NodeGCN(in_ch, GCN_HIDDEN, out_ch, GCN_CONV, GCN_LIN,
                  act=GCN_ACT, dropout=GCN_DROPOUT).to(DEVICE)
    gcn, acc = run_training(gcn, data)
    d = os.path.join(SAVE_ROOT, "gcn")
    save_results("node_gcn_pca", gcn, acc,
                 {"hidden": GCN_HIDDEN, "conv_layers": GCN_CONV,
                  "lin_layers": GCN_LIN, "dropout": GCN_DROPOUT}, d)
    export_model("node_gcn_pca", gcn, d)

    # ------------------------------------------------------------------
    # 3. Fair MLP K=1,2,3,4 (params match SIGN K=1..4 on PCA features)
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
        fair_mlp, acc = run_training(fair_mlp, data)
        d = os.path.join(SAVE_ROOT, f"mlp_fair_k{K}")
        save_results(f"node_mlp_pca_fair_k{K}", fair_mlp, acc,
                     {"K": K, "hidden": fair_h, "layers": 2,
                      "act": MLP_ACT, "dropout": MLP_DROPOUT,
                      "sign_params": sign_params}, d)
        export_model(f"node_mlp_pca_fair_k{K}", fair_mlp, d)

    # ------------------------------------------------------------------
    # 4. MLP-KD (distill from NodeGCN)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. MLP-KD  (T=4, lambda=0.5)")
    LAM, TEMP = 0.5, 4.0
    gcn.eval()

    set_seed(SEED)
    student = NodeMLP(in_ch, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                      act=MLP_ACT, dropout=MLP_DROPOUT).to(DEVICE)
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
    save_results("node_mlp_pca_kd", student, best_acc,
                 {"lambda": LAM, "temperature": TEMP, "hidden": MLP_HIDDEN,
                  "act": MLP_ACT, "dropout": MLP_DROPOUT}, d)
    export_model("node_mlp_pca_kd", student, d)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Summary (CiteSeer PCA val acc)")
    print("=" * 50)
    for subdir in ["mlp", "gcn", "mlp_fair_k1", "mlp_fair_k2",
                   "mlp_fair_k3", "mlp_fair_k4", "mlp_kd"]:
        rpath = os.path.join(SAVE_ROOT, subdir, "results.json")
        if os.path.exists(rpath):
            r = json.load(open(rpath))
            print(f"  {subdir:<14}  {r['val_acc']:.4f}")
