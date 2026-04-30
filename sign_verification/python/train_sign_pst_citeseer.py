"""SIGN(p,s,t) sweep on CiteSeer raw 3703-dim (transductive node classification)."""
from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from precompute_sign import build_operator_sequence

SEED = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/citeseer"

HPARAMS = {
    "lr": 0.01,
    "weight_decay": 5e-4,
    "max_epochs": 10000,
    "patience": 20,
    "eval_freq": 10,
}
MLP_HIDDEN = 64
MLP_LAYERS = 2
MLP_ACT = "relu"
MLP_DROPOUT = 0.2
PPR_ALPHA = 0.05


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_citeseer():
    ds = Planetoid(root=DATA_DIR, name="CiteSeer")
    return ds[0]


def precompute_sign_pst(data, p, s, t, ppr_alpha=0.05):
    """Compute SIGN(p,s,t) features for all nodes. Returns [N, (1+p+s+t)*F]."""
    X = data.x.cpu().numpy().astype(np.float64)
    edge_index = data.edge_index.cpu().numpy().astype(np.int64)
    n_nodes = data.num_nodes

    ops = build_operator_sequence(
        edge_index=edge_index, n_nodes=n_nodes,
        p=p, s=s, t=t, ppr_alpha=ppr_alpha,
    )

    parts = []
    for name, op in ops:
        if op is None:
            parts.append(X)
        else:
            parts.append(op.dot(X))

    return torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))


class NodeMLP(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, num_layers=2, act="relu", dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_ch, hidden)]
        if act == "tanh":
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            if act == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)


def train_one(p, s, t, save_dir, ppr_alpha=PPR_ALPHA):
    set_seed(SEED)
    data = load_citeseer()

    sign_x = precompute_sign_pst(data, p, s, t, ppr_alpha)
    data_gpu = data.to(DEVICE)
    sign_x_gpu = sign_x.to(DEVICE)

    in_ch = sign_x.shape[1]
    out_ch = int(data.y.max().item()) + 1

    set_seed(SEED)
    model = NodeMLP(in_ch, MLP_HIDDEN, out_ch, MLP_LAYERS, MLP_ACT, MLP_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])

    best_loss = float("inf")
    best_acc = 0.0
    patience_count = 0
    best_state = None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(sign_x_gpu)
        loss = F.cross_entropy(out[data_gpu.train_mask], data_gpu.y[data_gpu.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % HPARAMS["eval_freq"] == 0:
            model.eval()
            with torch.no_grad():
                out = model(sign_x_gpu)
                val_loss = F.cross_entropy(out[data_gpu.val_mask], data_gpu.y[data_gpu.val_mask]).item()
                val_acc = (out[data_gpu.val_mask].argmax(-1) == data_gpu.y[data_gpu.val_mask]).float().mean().item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                patience_count = 0
                best_state = deepcopy(model.state_dict())
            else:
                patience_count += 1
            if patience_count >= HPARAMS["patience"]:
                break

    model.load_state_dict(best_state)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    layers = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            layers.append({
                "type": "lin", "act": "",
                "W": module.weight.detach().cpu().tolist(),
                "b": module.bias.detach().cpu().tolist(),
            })
        elif isinstance(module, nn.Tanh):
            if layers:
                layers[-1]["act"] = "tanh"
        elif isinstance(module, nn.ReLU):
            if layers:
                layers[-1]["act"] = "relu"

    sign_config = {
        "p": p, "s": s, "t": t, "ppr_alpha": ppr_alpha,
        "operator_names": ["x0"] + [f"p{k}" for k in range(1, p+1)]
                          + [f"s{k}" for k in range(1, s+1)]
                          + [f"t{k}" for k in range(1, t+1)],
    }
    model_export = {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "sign",
            "task_level": "node",
            "input_mode": "sign_node_vector",
            "feature_dim": int(data.x.shape[1]),
            "sign_config": sign_config,
            "uses_edge_features": False,
        },
        "layers": layers,
    }
    with open(os.path.join(save_dir, "model_export.json"), "w") as f:
        json.dump(model_export, f)

    results = {
        "model": f"sign_p{p}_s{s}_t{t}",
        "dataset": "CiteSeer",
        "select_by": "val_loss",
        "sign_config": sign_config,
        "feature_dim": int(data.x.shape[1]),
        "sign_dim": in_ch,
        "num_classes": out_ch,
        "seed": SEED,
        "val_acc": best_acc,
        "val_loss": best_loss,
        "hparams": {**HPARAMS, "hidden": MLP_HIDDEN, "layers": MLP_LAYERS,
                    "act": MLP_ACT, "dropout": MLP_DROPOUT},
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_acc, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out_root", type=str,
                        default="artifacts/citeseer_sign_pst")
    args = parser.parse_args()

    if args.sweep:
        print("SIGN(p,s,t) sweep for CiteSeer")
        print("=" * 60)
        results = []
        for p in range(5):
            for s in range(5):
                for t in range(5):
                    if p == 0 and s == 0 and t == 0:
                        continue
                    tag = f"p{p}_s{s}_t{t}"
                    save_dir = os.path.join(args.out_root, tag)
                    print(f"\n--- {tag} ---")
                    acc, loss = train_one(p, s, t, save_dir)
                    print(f"  val_acc={acc:.4f}  val_loss={loss:.4f}")
                    results.append({"p": p, "s": s, "t": t,
                                    "val_acc": acc, "val_loss": loss})

        # Write summary
        with open(os.path.join(args.out_root, "sweep_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep complete. {len(results)} configs saved to {args.out_root}/")
    else:
        save_dir = os.path.join(args.out_root, f"p{args.p}_s{args.s}_t{args.t}")
        acc, loss = train_one(args.p, args.s, args.t, save_dir)
        print(f"p={args.p} s={args.s} t={args.t}  val_acc={acc:.4f}  val_loss={loss:.4f}")


if __name__ == "__main__":
    main()
