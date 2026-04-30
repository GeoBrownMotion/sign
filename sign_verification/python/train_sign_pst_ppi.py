"""SIGN(p,s,t) sweep on PPI (inductive multi-label node classification)."""
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
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from precompute_sign import build_operator_sequence

SEED = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/ppi"

HPARAMS = {
    "lr": 0.005,
    "weight_decay": 0.0,
    "max_epochs": 2000,
    "patience": 50,
    "eval_freq": 5,
    "batch_size": 2,
}
MLP_HIDDEN = 512
MLP_LAYERS = 2
MLP_ACT = "relu"
MLP_DROPOUT = 0.0
PPR_ALPHA = 0.05


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def precompute_sign_pst_graph(data, p, s, t, ppr_alpha=0.05):
    """Compute SIGN(p,s,t) features for one graph. Returns [N, (1+p+s+t)*F]."""
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


def make_sign_dataset(dataset, p, s, t, ppr_alpha=0.05):
    """Return dataset with SIGN(p,s,t) features replacing x."""
    out = []
    for data in dataset:
        d = data.clone()
        d.x = precompute_sign_pst_graph(data, p, s, t, ppr_alpha)
        out.append(d)
    return out


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


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
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
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        total_loss += loss.item()
        preds.append((out.sigmoid() > 0.5).cpu().numpy())
        labels.append(data.y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return total_loss / len(loader), f1


def train_one(p, s, t, save_dir, ppr_alpha=PPR_ALPHA):
    set_seed(SEED)

    train_ds = PPI(DATA_DIR, split="train")
    val_ds = PPI(DATA_DIR, split="val")

    in_ch_raw = train_ds[0].x.shape[1]
    out_ch = train_ds[0].y.shape[1]

    print(f"  Precomputing SIGN(p={p},s={s},t={t}) for {len(train_ds)} train + {len(val_ds)} val graphs...")
    train_sign = make_sign_dataset(train_ds, p, s, t, ppr_alpha)
    val_sign = make_sign_dataset(val_ds, p, s, t, ppr_alpha)

    in_ch = train_sign[0].x.shape[1]
    print(f"  sign_dim = {in_ch}  (raw={in_ch_raw})")

    train_loader = DataLoader(train_sign, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_sign, batch_size=HPARAMS["batch_size"], shuffle=False)

    set_seed(SEED)
    model = NodeMLP(in_ch, MLP_HIDDEN, out_ch, MLP_LAYERS, MLP_ACT, MLP_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])

    best_loss = float("inf")
    best_f1 = 0.0
    patience_count = 0
    best_state = None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_epoch(model, train_loader, optimizer)

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_f1 = evaluate(model, val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                best_f1 = val_f1
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
            "task_level": "multilabel",
            "input_mode": "sign_node_vector",
            "feature_dim": in_ch_raw,
            "num_outputs": out_ch,
            "sign_config": sign_config,
            "uses_edge_features": False,
        },
        "layers": layers,
    }
    with open(os.path.join(save_dir, "model_export.json"), "w") as f:
        json.dump(model_export, f)

    results = {
        "model": f"sign_p{p}_s{s}_t{t}",
        "dataset": "PPI",
        "select_by": "val_loss",
        "sign_config": sign_config,
        "feature_dim": in_ch_raw,
        "sign_dim": in_ch,
        "num_outputs": out_ch,
        "seed": SEED,
        "val_f1": best_f1,
        "val_loss": best_loss,
        "hparams": {**HPARAMS, "hidden": MLP_HIDDEN, "layers": MLP_LAYERS,
                    "act": MLP_ACT, "dropout": MLP_DROPOUT},
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_f1, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out_root", type=str,
                        default="artifacts/ppi_sign_pst")
    args = parser.parse_args()

    if args.sweep:
        print("SIGN(p,s,t) sweep for PPI")
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
                    try:
                        f1, loss = train_one(p, s, t, save_dir)
                        print(f"  val_f1={f1:.4f}  val_loss={loss:.4f}")
                        results.append({"p": p, "s": s, "t": t,
                                        "val_f1": f1, "val_loss": loss})
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        results.append({"p": p, "s": s, "t": t,
                                        "val_f1": None, "error": str(e)})

        with open(os.path.join(args.out_root, "sweep_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep complete. {len(results)} configs saved to {args.out_root}/")
    else:
        save_dir = os.path.join(args.out_root, f"p{args.p}_s{args.s}_t{args.t}")
        f1, loss = train_one(args.p, args.s, args.t, save_dir)
        print(f"p={args.p} s={args.s} t={args.t}  val_f1={f1:.4f}  val_loss={loss:.4f}")


if __name__ == "__main__":
    main()
