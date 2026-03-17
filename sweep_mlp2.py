"""
Second sweep: push pool-first MLP toward 70-72%.
"""
import json
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from models import MLP
from train import load_enzymes, train_epoch, evaluate, set_seed, HPARAMS

SEED   = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = [
    # (label,        hidden, layers)
    ("mlp_h256_l4",  256,    4),
    ("mlp_h256_l5",  256,    5),
    ("mlp_h512_l3",  512,    3),
    ("mlp_h512_l4",  512,    4),
    ("mlp_h384_l3",  384,    3),
    ("mlp_h384_l4",  384,    4),
]


def run_variant(label, hidden, layers):
    set_seed(SEED)
    train_data, val_data, *_ = load_enzymes(SEED)

    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)

    in_channels  = train_data[0].x.shape[1]

    model = MLP(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=6,
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

    print(f"  {label}: finished, val_acc={best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    print(f"Device: {DEVICE}, Seed: {SEED}\n")

    results = {}
    for label, hidden, layers in VARIANTS:
        acc = run_variant(label, hidden, layers)
        results[label] = round(acc, 4)

    print("\n" + "="*45)
    print(f"  {'Variant':<18} {'Val Acc':>8}")
    print(f"  {'-'*28}")
    for k, v in results.items():
        print(f"  {k:<18} {v:>8.4f}")

    with open("/home/hep3/sign/results/sweep_mlp2.json", "w") as f:
        json.dump(results, f, indent=2)
