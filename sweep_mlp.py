"""
Sweep MLP variants to find one with ~70-72% accuracy.

Variants:
  - mlp_pool_first:   global_mean_pool → MLP  (pool-first, varying hidden/layers)
  - mlp_per_node:     per-node MLP → global_mean_pool  (pool-last, varying hidden/layers)
"""

import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models import MLP, MLPPerNode
from train import load_enzymes, train_epoch, evaluate, set_seed, HPARAMS

SEED   = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = [
    # (label,           model_cls,   hidden, layers)
    ("mlp_h64_l3",      MLP,         64,     3),
    ("mlp_h128_l3",     MLP,         128,    3),
    ("mlp_h128_l4",     MLP,         128,    4),
    ("mlp_h256_l3",     MLP,         256,    3),
    ("pernode_h64_l3",  MLPPerNode,  64,     3),
    ("pernode_h128_l3", MLPPerNode,  128,    3),
    ("pernode_h128_l4", MLPPerNode,  128,    4),
    ("pernode_h256_l3", MLPPerNode,  256,    3),
]


def run_variant(label, model_cls, hidden, layers):
    set_seed(SEED)
    train_data, val_data, *_ = load_enzymes(SEED)

    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)

    in_channels  = train_data[0].x.shape[1]
    out_channels = 6

    model = model_cls(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out_channels,
        num_hidden_layers=layers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
        betas=(0.9, 0.999),
    )

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_epoch(model, train_loader, optimizer, DEVICE)

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_acc = evaluate(model, val_loader, DEVICE)

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_val_acc   = val_acc
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= HPARAMS["patience"]:
                print(f"  {label}: early stop at epoch {epoch}, val_acc={best_val_acc:.4f}")
                return best_val_acc

    print(f"  {label}: finished all epochs, val_acc={best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Seed: {SEED}\n")

    results = {}
    for label, model_cls, hidden, layers in VARIANTS:
        acc = run_variant(label, model_cls, hidden, layers)
        results[label] = round(acc, 4)

    print("\n" + "="*50)
    print(f"  {'Variant':<22} {'Val Acc':>8}")
    print(f"  {'-'*32}")
    for k, v in results.items():
        print(f"  {k:<22} {v:>8.4f}")

    with open("/home/hep3/sign/results/sweep_mlp.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /home/hep3/sign/results/sweep_mlp.json")
