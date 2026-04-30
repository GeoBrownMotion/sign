"""Pooled SIGN(p,s,t) training/export for ENZYMES; --head_style fair|paperlike."""
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset

from precompute_sign import compute_pooled_sign_vector


SEED = 8
HPARAMS = {
    "lr": 0.003,
    "weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 20000,
    "patience": 10,
    "eval_freq": 100,
    "train_split": 0.8,
}
MODEL_HPARAMS = {
    "hidden_channels": 256,
    "num_layers": 4,
    "dropout": 0.5,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_enzymes_with_splits(data_dir: str = "/tmp/enzymes_attr"):
    """
    Match the teammate ENZYMES split and node-feature standardization exactly.
    """
    dataset = TUDataset(root=data_dir, name="ENZYMES", use_node_attr=True)

    set_seed(SEED)
    perm = torch.randperm(len(dataset))
    n_train = int(HPARAMS["train_split"] * len(dataset))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]

    all_x = torch.cat([d.x for d in train_dataset], dim=0)
    feat_mean = all_x.mean(dim=0)
    feat_std = all_x.std(dim=0).clamp(min=1e-6)

    def normalize(dataset_split):
        normalized = []
        for data in dataset_split:
            d = deepcopy(data)
            d.x = (d.x - feat_mean) / feat_std
            normalized.append(d)
        return normalized

    return (
        normalize(train_dataset),
        normalize(val_dataset),
        feat_mean,
        feat_std,
        {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        },
    )


def compute_sign_features(
    data,
    p: int,
    s: int,
    t: int,
    ppr_alpha: float,
    triangle_keep_self_loops: bool = False,
) -> np.ndarray:
    x = data.x.detach().cpu().numpy().astype(np.float64)
    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)
    pooled, _ = compute_pooled_sign_vector(
        X=x,
        edge_index=edge_index,
        p=p,
        s=s,
        t=t,
        ppr_alpha=ppr_alpha,
        triangle_keep_self_loops=triangle_keep_self_loops,
    )
    return pooled.astype(np.float32)


def build_feature_tensor(
    dataset_split: list,
    p: int,
    s: int,
    t: int,
    ppr_alpha: float,
    triangle_keep_self_loops: bool,
) -> torch.Tensor:
    features = [
        compute_sign_features(data, p, s, t, ppr_alpha, triangle_keep_self_loops)
        for data in dataset_split
    ]
    return torch.from_numpy(np.stack(features, axis=0))


def build_label_tensor(dataset_split: list) -> torch.Tensor:
    labels = [int(data.y.item()) for data in dataset_split]
    return torch.tensor(labels, dtype=torch.long)


class FairSIGNMLP(nn.Module):
    """Exactly match the teammate pooled-vector MLP head."""

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        out_dim: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_channels), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_channels, hidden_channels), nn.Tanh()]
        layers += [nn.Linear(hidden_channels, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PaperlikeSIGNMLP(nn.Module):
    """Pooled-vector MLP head with BN/ReLU/Dropout blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        self.lins.append(nn.Linear(in_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


def _fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold BatchNorm1d into the preceding Linear for export-time inference."""
    W = linear.weight.detach().cpu()
    b = linear.bias.detach().cpu()
    gamma = bn.weight.detach().cpu()
    beta = bn.bias.detach().cpu()
    mean = bn.running_mean.detach().cpu()
    var = bn.running_var.detach().cpu()
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)
    fused_W = scale.unsqueeze(1) * W
    fused_b = beta + scale * (b - mean)
    return fused_W, fused_b


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    select_by: str,
) -> tuple[nn.Module, dict]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"],
        betas=(0.9, 0.999),
    )

    best_state = deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improve = 0
    train_loss = float("nan")
    val_loss = float("nan")

    if select_by not in {"val_loss", "val_acc"}:
        raise ValueError(f"select_by must be 'val_loss' or 'val_acc', got {select_by!r}")

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % HPARAMS["eval_freq"] != 0 and epoch != 1:
            continue

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            val_logits = model(X_val)
            train_loss = F.cross_entropy(train_logits, y_train).item()
            val_loss = F.cross_entropy(val_logits, y_val).item()
            train_acc = (train_logits.argmax(dim=-1) == y_train).float().mean().item()
            val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()

        if select_by == "val_loss":
            improved = val_loss < best_val_loss
        else:
            improved = val_acc > best_val_acc

        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= HPARAMS["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train)
        val_logits = model(X_val)
        train_loss = F.cross_entropy(train_logits, y_train).item()
        val_loss = F.cross_entropy(val_logits, y_val).item()
        train_acc = (train_logits.argmax(dim=-1) == y_train).float().mean().item()
        val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()

    return model, {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }


def export_model_json(
    model: nn.Module,
    out_path: Path,
    sign_config: dict,
    feature_dim: int,
    head_style: str,
) -> None:
    layers = []
    if head_style == "fair":
        linears = [module for module in model.net.modules() if isinstance(module, nn.Linear)]
        for idx, linear in enumerate(linears):
            layers.append(
                {
                    "type": "lin",
                    "act": "" if idx == len(linears) - 1 else "tanh",
                    "W": linear.weight.detach().cpu().tolist(),
                    "b": linear.bias.detach().cpu().tolist(),
                }
            )
    elif head_style == "paperlike":
        for linear, bn in zip(model.lins[:-1], model.bns):
            fused_W, fused_b = _fuse_linear_bn(linear, bn)
            layers.append(
                {
                    "type": "lin",
                    "act": "relu",
                    "W": fused_W.tolist(),
                    "b": fused_b.tolist(),
                }
            )
        last_linear = model.lins[-1]
        layers.append(
            {
                "type": "lin",
                "act": "",
                "W": last_linear.weight.detach().cpu().tolist(),
                "b": last_linear.bias.detach().cpu().tolist(),
            }
        )
    else:
        raise ValueError(f"Unsupported head_style: {head_style}")

    obj = {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "sign",
            "task_level": "graph",
            "input_mode": "sign_pooled_vector",
            "feature_dim": feature_dim,
            "sign_config": sign_config,
            "head_style": head_style,
            "uses_edge_features": False,
        },
        "layers": layers,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def export_eval_data_json(
    model: nn.Module,
    eval_dataset: list,
    out_path: Path,
    p: int,
    s: int,
    t: int,
    ppr_alpha: float,
    triangle_keep_self_loops: bool,
) -> float:
    headers = [
        "input",
        "output",
        "output_label",
        "target_label",
        "loss",
        "edge_index",
        "edge_weight",
    ]
    rows = [headers]
    correct = 0

    model.eval()
    with torch.no_grad():
        for data in eval_dataset:
            feature_vec = torch.from_numpy(
                compute_sign_features(
                    data,
                    p=p,
                    s=s,
                    t=t,
                    ppr_alpha=ppr_alpha,
                    triangle_keep_self_loops=triangle_keep_self_loops,
                )
            ).unsqueeze(0)
            logits = model(feature_vec).squeeze(0)
            pred = int(logits.argmax().item())
            target = int(data.y.item())
            loss = float(F.cross_entropy(logits.unsqueeze(0), data.y.view(1)).item())
            correct += int(pred == target)

            rows.append(
                [
                    data.x.detach().cpu().tolist(),
                    [logits.detach().cpu().tolist()],
                    pred,
                    target,
                    loss,
                    data.edge_index.detach().cpu().tolist(),
                    [],
                ]
            )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f)
    return correct / len(eval_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fair SIGN baseline on ENZYMES.")
    parser.add_argument("--out_dir", type=str, default="artifacts/fair_sign")
    parser.add_argument("--p", type=int, default=2, help="Number of simple/GCN-normalized powers.")
    parser.add_argument("--s", type=int, default=0, help="Number of PPR powers.")
    parser.add_argument("--t", type=int, default=0, help="Number of triangle powers.")
    parser.add_argument("--ppr_alpha", type=float, default=0.05)
    parser.add_argument("--triangle_keep_self_loops", action="store_true")
    parser.add_argument(
        "--head_style",
        choices=["fair", "paperlike"],
        default="fair",
        help="Use teammate-matched head (`fair`) or SIGN-repo-style head (`paperlike`).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="sign_pst_fair",
        help="Model label stored in exported metrics.",
    )
    parser.add_argument(
        "--select_by",
        choices=["val_loss", "val_acc"],
        default="val_loss",
        help="Checkpoint selection and early stopping metric.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    train_data, val_data, feat_mean, feat_std, split_meta = load_enzymes_with_splits()
    feature_dim = int(train_data[0].x.shape[1])
    num_classes = int(max(int(d.y.item()) for d in train_data + val_data)) + 1

    X_train = build_feature_tensor(
        train_data,
        p=args.p,
        s=args.s,
        t=args.t,
        ppr_alpha=args.ppr_alpha,
        triangle_keep_self_loops=args.triangle_keep_self_loops,
    )
    X_val = build_feature_tensor(
        val_data,
        p=args.p,
        s=args.s,
        t=args.t,
        ppr_alpha=args.ppr_alpha,
        triangle_keep_self_loops=args.triangle_keep_self_loops,
    )
    y_train = build_label_tensor(train_data)
    y_val = build_label_tensor(val_data)

    sign_config = {
        "p": args.p,
        "s": args.s,
        "t": args.t,
        "ppr_alpha": args.ppr_alpha,
        "triangle_keep_self_loops": args.triangle_keep_self_loops,
        "operator_names": (
            ["x0"]
            + [f"p{k}" for k in range(1, args.p + 1)]
            + [f"s{k}" for k in range(1, args.s + 1)]
            + [f"t{k}" for k in range(1, args.t + 1)]
        ),
    }

    if args.head_style == "fair":
        model = FairSIGNMLP(
            in_dim=X_train.shape[1],
            hidden_channels=MODEL_HPARAMS["hidden_channels"],
            out_dim=num_classes,
            num_hidden_layers=MODEL_HPARAMS["num_layers"],
        )
    else:
        model = PaperlikeSIGNMLP(
            in_dim=X_train.shape[1],
            hidden_channels=MODEL_HPARAMS["hidden_channels"],
            out_dim=num_classes,
            num_layers=MODEL_HPARAMS["num_layers"],
            dropout=MODEL_HPARAMS["dropout"],
        )
    model, train_metrics = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        select_by=args.select_by,
    )

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
        val_loss = F.cross_entropy(val_logits, y_val).item()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    export_model_json(
        model,
        out_dir / "model_export.json",
        sign_config,
        feature_dim,
        args.head_style,
    )
    exported_acc = export_eval_data_json(
        model,
        val_data,
        out_dir / "data_export.json",
        p=args.p,
        s=args.s,
        t=args.t,
        ppr_alpha=args.ppr_alpha,
        triangle_keep_self_loops=args.triangle_keep_self_loops,
    )

    results = {
        "model": args.label,
        "head_style": args.head_style,
        "select_by": args.select_by,
        "sign_config": sign_config,
        "feature_dim": feature_dim,
        "sign_dim": int(X_train.shape[1]),
        "num_classes": num_classes,
        "seed": SEED,
        "hparams": {**HPARAMS, **MODEL_HPARAMS},
        "train_loss": train_metrics["train_loss"],
        "val_loss": train_metrics["val_loss"],
        "best_val_loss": train_metrics["best_val_loss"],
        "train_acc": train_metrics["train_acc"],
        "val_acc": train_metrics["val_acc"],
        "eval_acc": val_acc,
        "eval_loss": val_loss,
        "exported_eval_acc": exported_acc,
        "train_size": len(train_data),
        "val_size": len(val_data),
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    split_payload = {
        **split_meta,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
    }
    with (out_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    print(f"Saved fair SIGN artifacts -> {out_dir}")
    print(
        f"  train={results['train_size']} val={results['val_size']} "
        f"sign_dim={results['sign_dim']} "
        f"config=p{args.p}_s{args.s}_t{args.t} head={args.head_style} select_by={args.select_by}"
    )
    print(
        f"  best_val_acc={results['val_acc']:.4f} eval_acc={results['eval_acc']:.4f} "
        f"exported_eval_acc={results['exported_eval_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
