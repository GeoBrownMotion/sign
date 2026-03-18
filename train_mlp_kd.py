"""
MLP-KD training script: Graph-Less Neural Network (GLNN) framework.

Trains a pool-first MLP student using knowledge distillation from a trained GCN teacher.

Loss: L = lambda * CE(student, true_labels) + (1-lambda) * T^2 * KL(student/T, teacher/T)

Reference:
  Zhang et al. "Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation"
  ICLR 2022.

Usage:
    python train_mlp_kd.py
    python train_mlp_kd.py --lam 0.5 --temp 1.0
    python train_mlp_kd.py --sweep   # sweep over lambda x temperature grid
"""

import argparse
import json
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from export import export_model
from models import GCN, MLP
from train import (
    GCN_HPARAMS,
    HPARAMS,
    MLP_HPARAMS,
    evaluate,
    load_enzymes,
    set_seed,
)

SEED       = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR   = "/home/hep3/sign/results/mlp_kd"
TEACHER_PT = "/home/hep3/sign/results/gcn/model.pt"


def load_teacher(in_ch: int) -> torch.nn.Module:
    teacher = GCN(
        in_ch,
        GCN_HPARAMS["hidden_channels"],
        6,
        GCN_HPARAMS["num_layers"],
        GCN_HPARAMS["num_lin_layers"],
    )
    teacher.load_state_dict(torch.load(TEACHER_PT, map_location="cpu"))
    return teacher.to(DEVICE).eval()


def train_kd(lam: float, temp: float, train_data, val_data, teacher) -> tuple:
    """Train one student MLP with KD. Returns (val_acc, best_state_dict)."""
    set_seed(SEED)
    in_ch = train_data[0].x.shape[1]
    student = MLP(in_ch, MLP_HPARAMS["hidden_channels"], 6, MLP_HPARAMS["num_layers"]).to(DEVICE)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
        betas=(0.9, 0.999),
    )
    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)

    best_loss, best_acc, patience, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        student.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = student(batch.x, batch.edge_index, batch.batch)

            loss_ce = F.cross_entropy(out, batch.y)

            with torch.no_grad():
                t_soft = F.softmax(
                    teacher(batch.x, batch.edge_index, batch.batch) / temp, dim=-1
                )
            loss_kd = (
                F.kl_div(
                    F.log_softmax(out / temp, dim=-1),
                    t_soft,
                    reduction="batchmean",
                )
                * (temp ** 2)
            )

            loss = lam * loss_ce + (1 - lam) * loss_kd
            loss.backward()
            optimizer.step()

        if epoch % HPARAMS["eval_freq"] == 0:
            val_loss, val_acc = evaluate(student, val_loader, DEVICE)
            if val_loss < best_loss:
                best_loss  = val_loss
                best_acc   = val_acc
                patience   = 0
                best_state = deepcopy(student.state_dict())
            else:
                patience += 1
            if patience >= HPARAMS["patience"]:
                break

    return best_acc, best_state


def save_model(student, best_acc, lam, temp, feat_mean, feat_std):
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.save(student.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

    with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
        json.dump({"feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist()}, f, indent=2)

    results = {
        "model":           "mlp_kd",
        "description":     "Pool-first MLP with GCN knowledge distillation (GLNN framework)",
        "hidden_channels": MLP_HPARAMS["hidden_channels"],
        "num_layers":      MLP_HPARAMS["num_layers"],
        "lambda_ce":       lam,
        "temperature":     temp,
        "teacher":         "gcn_h128_c4",
        "val_acc":         round(best_acc, 6),
        "seed":            SEED,
        "hparams":         {**HPARAMS},
    }
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    export_model("mlp_kd", student, SAVE_DIR)
    print(f"Saved to {SAVE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam",   type=float, default=0.5)
    parser.add_argument("--temp",  type=float, default=1.0)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over lambda x temperature grid and save best model")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    set_seed(SEED)
    train_data, val_data, feat_mean, feat_std = load_enzymes(SEED)
    in_ch = train_data[0].x.shape[1]
    teacher = load_teacher(in_ch)
    print("Teacher GCN loaded.")

    if args.sweep:
        TEMPS   = [1, 2, 4, 8, 16]
        LAMBDAS = [0.1, 0.3, 0.5, 0.7]

        print(f"\n  {'':10}", end="")
        for lam in LAMBDAS:
            print(f"  lam={lam}", end="")
        print("\n  " + "-" * 50)

        best_overall_acc   = 0.0
        best_overall_state = None
        best_overall_cfg   = (None, None)
        sweep_results      = {}

        for T in TEMPS:
            print(f"  T={T:<8}", end="", flush=True)
            for lam in LAMBDAS:
                acc, state = train_kd(lam, T, train_data, val_data, teacher)
                sweep_results[f"T{T}_lam{lam}"] = round(acc, 4)
                print(f"  {acc:.4f}", end="", flush=True)
                if acc > best_overall_acc:
                    best_overall_acc   = acc
                    best_overall_state = state
                    best_overall_cfg   = (lam, T)
            print()

        best_lam, best_T = best_overall_cfg
        print(f"\nBest: T={best_T}, lambda={best_lam}, val_acc={best_overall_acc:.4f}")

        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(os.path.join(SAVE_DIR, "sweep_results.json"), "w") as f:
            json.dump(sweep_results, f, indent=2)

        # Save best model
        in_ch   = train_data[0].x.shape[1]
        student = MLP(in_ch, MLP_HPARAMS["hidden_channels"], 6, MLP_HPARAMS["num_layers"])
        student.load_state_dict(best_overall_state)
        student = student.to(DEVICE).eval()
        save_model(student, best_overall_acc, best_lam, best_T, feat_mean, feat_std)

    else:
        print(f"Training MLP-KD: lambda={args.lam}, T={args.temp}")
        acc, state = train_kd(args.lam, args.temp, train_data, val_data, teacher)
        print(f"Val acc: {acc:.4f}")

        in_ch   = train_data[0].x.shape[1]
        student = MLP(in_ch, MLP_HPARAMS["hidden_channels"], 6, MLP_HPARAMS["num_layers"])
        student.load_state_dict(state)
        student = student.to(DEVICE).eval()
        save_model(student, acc, args.lam, args.temp, feat_mean, feat_std)
