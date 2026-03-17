"""
Sweep GIN configurations vs GCN baseline.
Varies: hidden_channels, num_conv_layers, num_lin_layers, activation.
"""

import json
import torch
from torch_geometric.loader import DataLoader

from models import GCN, GIN, GCNNoOutAct, GINNoOutAct
from train import load_enzymes, train_epoch, evaluate, set_seed, HPARAMS

SEED   = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (label, model_cls, hidden, conv_layers, lin_layers, act)
VARIANTS = [
    # GCN baseline (tanh, original architecture)
    ("gcn_tanh_h64_c3_l1",    GCN, 64,  3, 1, "tanh"),

    # GIN tanh
    ("gin_tanh_h64_c3_l1",    GIN, 64,  3, 1, "tanh"),
    ("gin_tanh_h128_c3_l1",   GIN, 128, 3, 1, "tanh"),
    ("gin_tanh_h128_c4_l1",   GIN, 128, 4, 1, "tanh"),
    ("gin_tanh_h128_c3_l2",   GIN, 128, 3, 2, "tanh"),
    ("gin_tanh_h256_c3_l1",   GIN, 256, 3, 1, "tanh"),
    ("gin_tanh_h256_c4_l1",   GIN, 256, 4, 1, "tanh"),
    ("gin_tanh_h64_c5_l1",    GIN, 64,  5, 1, "tanh"),

    # GIN relu (broken: output activation clips logits to ≥0)
    # ("gin_relu_h64_c3_l1",  GIN, 64,  3, 1, "relu"),

    # GCN relu fixed (no output activation)
    ("gcn_relu_h64_c3_l1",    GCNNoOutAct, 64,  3, 1, "relu"),
    ("gcn_relu_h128_c3_l1",   GCNNoOutAct, 128, 3, 1, "relu"),
    ("gcn_relu_h128_c4_l1",   GCNNoOutAct, 128, 4, 1, "relu"),
    ("gcn_relu_h256_c3_l1",   GCNNoOutAct, 256, 3, 1, "relu"),

    # GIN relu fixed (no output activation)
    ("gin_relu_h64_c3_l1",    GINNoOutAct, 64,  3, 1, "relu"),
    ("gin_relu_h128_c3_l1",   GINNoOutAct, 128, 3, 1, "relu"),
    ("gin_relu_h128_c4_l1",   GINNoOutAct, 128, 4, 1, "relu"),
    ("gin_relu_h128_c3_l2",   GINNoOutAct, 128, 3, 2, "relu"),
    ("gin_relu_h256_c3_l1",   GINNoOutAct, 256, 3, 1, "relu"),
    ("gin_relu_h256_c4_l1",   GINNoOutAct, 256, 4, 1, "relu"),
    ("gin_relu_h64_c5_l1",    GINNoOutAct, 64,  5, 1, "relu"),
    ("gin_relu_h128_c5_l1",   GINNoOutAct, 128, 5, 1, "relu"),
]


def run_variant(label, model_cls, hidden, conv_layers, lin_layers, act):
    set_seed(SEED)
    train_data, val_data, *_ = load_enzymes(SEED)

    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)

    model = model_cls(
        in_channels=train_data[0].x.shape[1],
        hidden_channels=hidden,
        out_channels=6,
        num_conv_layers=conv_layers,
        num_lin_layers=lin_layers,
        act=act,
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
                break

    print(f"  {label:<28} val_acc={best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    print(f"Device: {DEVICE}, Seed: {SEED}\n")

    results = {}
    for label, model_cls, hidden, conv_layers, lin_layers, act in VARIANTS:
        acc = run_variant(label, model_cls, hidden, conv_layers, lin_layers, act)
        results[label] = round(acc, 4)

    print("\n" + "="*52)
    print(f"  {'Variant':<28} {'Val Acc':>8}")
    print(f"  {'-'*38}")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        marker = " <-- GCN baseline" if k == "gcn_tanh_h64_c3_l1" else ""
        print(f"  {k:<28} {v:>8.4f}{marker}")

    with open("/home/hep3/sign/results/sweep_gin.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /home/hep3/sign/results/sweep_gin.json")
