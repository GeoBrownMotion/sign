"""
Training and evaluation script for MLP and GCN baselines on the Enzymes dataset.

Usage:
    python train.py --model mlp
    python train.py --model gcn
    python train.py --model all   # train both sequentially

Hyperparameters match the reference paper (Ladner et al., 2025) for Enzymes:
    hidden_channels=64, num_layers=3, act=tanh, glob_pool=mean,
    lr=0.003, batch_size=256, Adam, no dropout, no weight_decay.
"""

import argparse
import random
import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

from models import MLP, GCN


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Hyperparameters (matching reference paper for Enzymes)
# ---------------------------------------------------------------------------
HPARAMS = {
    "hidden_channels": 64,
    "num_layers": 3,        # GCN conv layers / MLP hidden layers
    "num_lin_layers": 1,    # linear layers after pooling (GCN only)
    "lr": 0.003,
    "weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 5000,
    "patience": 50,         # early-stopping patience (in eval_freq units)
    "eval_freq": 10,        # evaluate every N epochs
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_enzymes(data_dir: str = "/tmp/enzymes_attr"):
    """
    Load the ENZYMES TUDataset with all 21 continuous node attributes.
    Applies per-feature standardisation (zero mean, unit variance) across the
    training set, matching `normalization=True` in the reference.
    """
    dataset = TUDataset(root=data_dir, name="ENZYMES", use_node_attr=True)

    # Shuffle and split: 80% train, 10% val, 10% test (fixed seed)
    set_seed(SEED)
    perm = torch.randperm(len(dataset))
    n_train = int(0.8 * len(dataset))
    n_val   = int(0.1 * len(dataset))

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_dataset = dataset[train_idx]
    val_dataset   = dataset[val_idx]
    test_dataset  = dataset[test_idx]

    # Compute per-feature mean/std on training node features
    all_x = torch.cat([d.x for d in train_dataset], dim=0)  # (N_train_nodes, 21)
    feat_mean = all_x.mean(dim=0)
    feat_std  = all_x.std(dim=0).clamp(min=1e-6)

    def normalise(dataset_split):
        normalised = []
        for data in dataset_split:
            d = deepcopy(data)
            d.x = (d.x - feat_mean) / feat_std
            normalised.append(d)
        return normalised

    train_data = normalise(train_dataset)
    val_data   = normalise(val_dataset)
    test_data  = normalise(test_dataset)

    return train_data, val_data, test_data, feat_mean, feat_std


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
# Main training loop
# ---------------------------------------------------------------------------
def train_model(model_name: str, device: torch.device, save_dir: str = "/home/hep3/sign/results"):
    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()} on ENZYMES")
    print(f"{'='*60}")
    set_seed(SEED)

    train_data, val_data, test_data, feat_mean, feat_std = load_enzymes()

    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=HPARAMS["batch_size"], shuffle=False)

    in_channels  = train_data[0].x.shape[1]   # 21
    out_channels = 6                            # 6 enzyme classes

    if model_name == "mlp":
        model = MLP(
            in_channels=in_channels,
            hidden_channels=HPARAMS["hidden_channels"],
            out_channels=out_channels,
            num_hidden_layers=HPARAMS["num_layers"],
        )
    elif model_name == "gcn":
        model = GCN(
            in_channels=in_channels,
            hidden_channels=HPARAMS["hidden_channels"],
            out_channels=out_channels,
            num_conv_layers=HPARAMS["num_layers"],
            num_lin_layers=HPARAMS["num_lin_layers"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
        betas=(0.9, 0.999),
    )

    best_val_loss   = float("inf")
    best_state_dict = None
    patience_count  = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_state_dict = deepcopy(model.state_dict())
                patience_count  = 0
            else:
                patience_count += 1

            if epoch % (HPARAMS["eval_freq"] * 10) == 0:
                print(f"  Epoch {epoch:5d} | train_loss={train_loss:.4f} "
                      f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

            if patience_count >= HPARAMS["patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # Restore best checkpoint and evaluate on test set
    model.load_state_dict(best_state_dict)
    test_loss, test_acc = evaluate(model, test_loader, device)
    val_loss_best, val_acc_best = evaluate(model, val_loader, device)

    print(f"\n  Results ({model_name.upper()}):")
    print(f"    Val  accuracy : {val_acc_best:.4f}")
    print(f"    Test accuracy : {test_acc:.4f}")
    print(f"    Test loss     : {test_loss:.4f}")

    # Save checkpoint and metrics
    os.makedirs(save_dir, exist_ok=True)
    run_dir = os.path.join(save_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    torch.save(best_state_dict, os.path.join(run_dir, "model.pt"))

    results = {
        "model": model_name,
        "val_acc":  round(val_acc_best, 6),
        "test_acc": round(test_acc, 6),
        "test_loss": round(test_loss, 6),
        "hparams": HPARAMS,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save normalisation stats (needed for verifier input construction)
    norm_stats = {
        "feat_mean": feat_mean.tolist(),
        "feat_std":  feat_std.tolist(),
    }
    with open(os.path.join(run_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"  Saved to {run_dir}/")
    return model, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn", "all"], default="all",
                        help="Which model to train (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    models_to_train = ["mlp", "gcn"] if args.model == "all" else [args.model]
    all_results = {}

    for m in models_to_train:
        model, results = train_model(m, device)
        all_results[m] = results

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  {'Model':<8} {'Val Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*30}")
    for m, r in all_results.items():
        print(f"  {m.upper():<8} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")
