"""
Training and evaluation script for MLP and GCN baselines on the Enzymes dataset.

Usage:
    python train.py --model mlp
    python train.py --model gcn
    python train.py --model all

Methodology matches the reference paper (Ladner et al., 2025):
  - Single 80/20 train/val split (no separate test set)
  - Multiple random seeds; report mean ± std
  - max_epochs=20000, eval_freq=100, patience=10 (early stopping on val loss)
"""

import argparse
import random
import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from models import MLP, GCN


SEEDS = [8]

HPARAMS = {
    "lr": 0.003,
    "weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 20000,
    "patience": 10,       # in eval_freq units → 1000 effective epochs
    "eval_freq": 100,
    "train_split": 0.8,
}

# Model-specific architecture configs
MLP_HPARAMS = {
    "hidden_channels": 256,
    "num_layers": 4,
}

GCN_HPARAMS = {
    "hidden_channels": 128,
    "num_layers": 4,
    "num_lin_layers": 1,
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_enzymes(seed: int, data_dir: str = "/tmp/enzymes_attr"):
    dataset = TUDataset(root=data_dir, name="ENZYMES", use_node_attr=True)

    set_seed(seed)
    perm = torch.randperm(len(dataset))
    n_train = int(HPARAMS["train_split"] * len(dataset))

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    train_data = [dataset[i] for i in train_idx]
    val_data   = [dataset[i] for i in val_idx]

    # Normalise: fit on train, apply to both
    all_x    = torch.cat([d.x for d in train_data], dim=0)
    feat_mean = all_x.mean(dim=0)
    feat_std  = all_x.std(dim=0).clamp(min=1e-6)

    def apply_norm(split):
        out = []
        for data in split:
            d = deepcopy(data)
            d.x = (d.x - feat_mean) / feat_std
            out.append(d)
        return out

    return apply_norm(train_data), apply_norm(val_data), feat_mean, feat_std


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    for batch in loader:
        batch = batch.to(device)
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y, reduction="sum")
        total_loss    += loss.item()
        total_correct += (out.argmax(dim=-1) == batch.y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------
def train_one_seed(model_name: str, seed: int, device: torch.device):
    set_seed(seed)

    train_data, val_data, feat_mean, feat_std = load_enzymes(seed)

    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)

    in_channels  = train_data[0].x.shape[1]
    out_channels = 6

    if model_name == "mlp":
        model = MLP(
            in_channels=in_channels,
            hidden_channels=MLP_HPARAMS["hidden_channels"],
            out_channels=out_channels,
            num_hidden_layers=MLP_HPARAMS["num_layers"],
        )
    else:
        model = GCN(
            in_channels=in_channels,
            hidden_channels=GCN_HPARAMS["hidden_channels"],
            out_channels=out_channels,
            num_conv_layers=GCN_HPARAMS["num_layers"],
            num_lin_layers=GCN_HPARAMS["num_lin_layers"],
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
        betas=(0.9, 0.999),
    )

    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    best_state_dict = None
    patience_count  = 0

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_epoch(model, train_loader, optimizer, device)

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_val_acc    = val_acc
                best_state_dict = deepcopy(model.state_dict())
                patience_count  = 0
            else:
                patience_count += 1

            if patience_count >= HPARAMS["patience"]:
                break

    model.load_state_dict(best_state_dict)
    return model, best_val_acc, feat_mean, feat_std


# ---------------------------------------------------------------------------
# Multi-seed training
# ---------------------------------------------------------------------------
def train_model(model_name: str, device: torch.device, save_dir: str = "/home/hep3/sign/results"):
    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()} on ENZYMES ({len(SEEDS)} seeds)")
    print(f"{'='*60}")

    seed_accs  = []
    best_acc   = -1.0
    best_model = None
    best_norm  = None

    for seed in SEEDS:
        model, val_acc, feat_mean, feat_std = train_one_seed(model_name, seed, device)
        seed_accs.append(val_acc)
        print(f"  Seed {seed:2d} | val_acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_model = deepcopy(model)
            best_norm  = (feat_mean, feat_std)

    mean_acc = float(np.mean(seed_accs))
    std_acc  = float(np.std(seed_accs))
    print(f"\n  {model_name.upper()} | mean={mean_acc:.4f}  std={std_acc:.4f}")

    run_dir = os.path.join(save_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    torch.save(best_model.state_dict(), os.path.join(run_dir, "model.pt"))

    norm_stats = {
        "feat_mean": best_norm[0].tolist(),
        "feat_std":  best_norm[1].tolist(),
    }
    with open(os.path.join(run_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    model_hparams = {**HPARAMS, **(MLP_HPARAMS if model_name == "mlp" else GCN_HPARAMS)}
    results = {
        "model":      model_name,
        "mean_acc":   round(mean_acc, 6),
        "std_acc":    round(std_acc, 6),
        "seed_accs":  [round(a, 6) for a in seed_accs],
        "best_acc":   round(best_acc, 6),
        "seeds":      SEEDS,
        "hparams":    model_hparams,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {run_dir}/")
    return best_model, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn", "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    models_to_train = ["mlp", "gcn"] if args.model == "all" else [args.model]
    all_results = {}

    for m in models_to_train:
        _, results = train_model(m, device)
        all_results[m] = results

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  {'Model':<8} {'Mean Acc':>10} {'Std':>8} {'Best':>8}")
    print(f"  {'-'*36}")
    for m, r in all_results.items():
        print(f"  {m.upper():<8} {r['mean_acc']:>10.4f} {r['std_acc']:>8.4f} {r['best_acc']:>8.4f}")
