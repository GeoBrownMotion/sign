"""Export perturbed val data (L_inf feature noise + random edge removal)."""

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

SEED = 42
FEATURE_EPS = [0.001, 0.005, 0.01, 0.05]
EDGE_FRAC = [0.001, 0.005, 0.01, 0.05]

SIGN_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = SIGN_ROOT / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def perturb_features(x: torch.Tensor, eps: float) -> torch.Tensor:
    noise = torch.empty_like(x).uniform_(-eps, eps)
    return x + noise


def perturb_edges(edge_index: torch.Tensor, num_nodes: int, frac: float) -> torch.Tensor:
    """Remove a fraction of undirected edges (both directions dropped together)."""
    src, dst = edge_index[0], edge_index[1]

    # Separate self-loops (never remove) from non-self-loop edges
    self_loop_mask = src == dst
    non_self_src = src[~self_loop_mask]
    non_self_dst = dst[~self_loop_mask]

    # Get unique undirected edges (u < v)
    u = torch.min(non_self_src, non_self_dst)
    v = torch.max(non_self_src, non_self_dst)
    unique_edges = set()
    for i in range(u.shape[0]):
        unique_edges.add((u[i].item(), v[i].item()))
    unique_edges = list(unique_edges)

    num_remove = int(len(unique_edges) * frac)
    if num_remove == 0 and frac > 0 and len(unique_edges) > 0:
        num_remove = 1  # remove at least 1 edge if frac > 0

    remove_set = set(random.sample(unique_edges, min(num_remove, len(unique_edges))))

    # Rebuild edge_index: keep self-loops + non-removed edges (both directions)
    keep_src, keep_dst = [], []

    # Self-loops
    self_src = src[self_loop_mask]
    self_dst = dst[self_loop_mask]
    keep_src.append(self_src)
    keep_dst.append(self_dst)

    # Non-self-loop edges
    for i in range(non_self_src.shape[0]):
        s, d = non_self_src[i].item(), non_self_dst[i].item()
        key = (min(s, d), max(s, d))
        if key not in remove_set:
            keep_src.append(torch.tensor([s]))
            keep_dst.append(torch.tensor([d]))

    new_src = torch.cat(keep_src)
    new_dst = torch.cat(keep_dst)
    return torch.stack([new_src, new_dst], dim=0)


def export_enzymes_perturbed():
    from train import load_enzymes

    print("=" * 60)
    print("ENZYMES - Perturbed Dataset Export")
    print("=" * 60)

    _, val_data, feat_mean, feat_std = load_enzymes(seed=8)
    out_dir = RESULTS_ROOT / "perturbed" / "enzymes"
    os.makedirs(out_dir, exist_ok=True)

    headers = ["input", "output", "output_label", "target_label",
               "loss", "edge_index", "edge_weight"]

    # Feature perturbation
    for eps in FEATURE_EPS:
        set_seed(SEED)
        rows = [headers]
        for data in val_data:
            x_pert = perturb_features(data.x, eps)
            target_label = int(data.y.item())
            rows.append([
                x_pert.tolist(),
                [],  # no model output — teammate will run her models
                -1,
                target_label,
                -1,
                data.edge_index.tolist(),
                [],
            ])

        fname = f"data_feat_eps{eps}.json"
        path = out_dir / fname
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"  {fname}: {len(val_data)} samples")

    # Edge perturbation
    for frac in EDGE_FRAC:
        set_seed(SEED)
        rows = [headers]
        for data in val_data:
            ei_pert = perturb_edges(data.edge_index, data.num_nodes, frac)
            target_label = int(data.y.item())
            rows.append([
                data.x.tolist(),  # clean features
                [],
                -1,
                target_label,
                -1,
                ei_pert.tolist(),
                [],
            ])

        fname = f"data_edge_frac{frac}.json"
        path = out_dir / fname
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"  {fname}: {len(val_data)} samples")

    # Also export clean for reference (same format)
    rows = [headers]
    for data in val_data:
        rows.append([
            data.x.tolist(),
            [],
            -1,
            int(data.y.item()),
            -1,
            data.edge_index.tolist(),
            [],
        ])
    path = out_dir / "data_clean.json"
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"  data_clean.json: {len(val_data)} samples")
    print()


def export_citeseer_perturbed():
    from torch_geometric.datasets import Planetoid

    print("=" * 60)
    print("CiteSeer - Perturbed Dataset Export")
    print("=" * 60)

    ds = Planetoid(root="/tmp/citeseer", name="CiteSeer")
    data = ds[0]
    out_dir = RESULTS_ROOT / "perturbed" / "citeseer"
    os.makedirs(out_dir, exist_ok=True)

    headers = ["input", "output", "output_label", "target_label",
               "edge_index", "edge_weight"]

    # Feature perturbation
    for eps in FEATURE_EPS:
        set_seed(SEED)
        x_pert = perturb_features(data.x, eps)
        row = [
            x_pert.tolist(),
            [],
            [],
            data.y.tolist(),
            data.edge_index.tolist(),
            [],
        ]
        path = out_dir / f"data_feat_eps{eps}.json"
        with open(path, "w") as f:
            json.dump([headers, row], f)
        print(f"  data_feat_eps{eps}.json: {data.num_nodes} nodes")

    # Edge perturbation
    for frac in EDGE_FRAC:
        set_seed(SEED)
        ei_pert = perturb_edges(data.edge_index, data.num_nodes, frac)
        row = [
            data.x.tolist(),
            [],
            [],
            data.y.tolist(),
            ei_pert.tolist(),
            [],
        ]
        path = out_dir / f"data_edge_frac{frac}.json"
        with open(path, "w") as f:
            json.dump([headers, row], f)
        print(f"  data_edge_frac{frac}.json: {data.num_nodes} nodes, "
              f"edges {data.edge_index.shape[1]} -> {ei_pert.shape[1]}")

    # Clean reference
    row = [
        data.x.tolist(),
        [],
        [],
        data.y.tolist(),
        data.edge_index.tolist(),
        [],
    ]
    path = out_dir / "data_clean.json"
    with open(path, "w") as f:
        json.dump([headers, row], f)
    print(f"  data_clean.json: {data.num_nodes} nodes")
    print()


def export_ppi_perturbed():
    from torch_geometric.datasets import PPI

    print("=" * 60)
    print("PPI - Perturbed Dataset Export")
    print("=" * 60)

    val_ds = PPI("/tmp/ppi", split="val")
    data = val_ds[0]  # first val graph, consistent with existing exports
    out_dir = RESULTS_ROOT / "perturbed" / "ppi"
    os.makedirs(out_dir, exist_ok=True)

    headers = ["input", "output", "output_label", "target_label",
               "edge_index", "edge_weight"]

    # Feature perturbation
    for eps in FEATURE_EPS:
        set_seed(SEED)
        x_pert = perturb_features(data.x, eps)
        row = [
            x_pert.tolist(),
            [],
            [],
            data.y.int().tolist(),
            data.edge_index.tolist(),
            [],
        ]
        path = out_dir / f"data_feat_eps{eps}.json"
        with open(path, "w") as f:
            json.dump([headers, row], f)
        print(f"  data_feat_eps{eps}.json: {data.num_nodes} nodes")

    # Edge perturbation
    for frac in EDGE_FRAC:
        set_seed(SEED)
        ei_pert = perturb_edges(data.edge_index, data.num_nodes, frac)
        row = [
            data.x.tolist(),
            [],
            [],
            data.y.int().tolist(),
            ei_pert.tolist(),
            [],
        ]
        path = out_dir / f"data_edge_frac{frac}.json"
        with open(path, "w") as f:
            json.dump([headers, row], f)
        print(f"  data_edge_frac{frac}.json: {data.num_nodes} nodes, "
              f"edges {data.edge_index.shape[1]} -> {ei_pert.shape[1]}")

    # Clean reference
    row = [
        data.x.tolist(),
        [],
        [],
        data.y.int().tolist(),
        data.edge_index.tolist(),
        [],
    ]
    path = out_dir / "data_clean.json"
    with open(path, "w") as f:
        json.dump([headers, row], f)
    print(f"  data_clean.json: {data.num_nodes} nodes")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export perturbed datasets")
    parser.add_argument("--dataset", choices=["enzymes", "citeseer", "ppi", "all"],
                        default="all")
    args = parser.parse_args()

    datasets = ["enzymes", "citeseer", "ppi"] if args.dataset == "all" else [args.dataset]

    print(f"Seed: {SEED}")
    print(f"Feature eps: {FEATURE_EPS}")
    print(f"Edge frac:   {EDGE_FRAC}")
    print()

    for ds in datasets:
        if ds == "enzymes":
            export_enzymes_perturbed()
        elif ds == "citeseer":
            export_citeseer_perturbed()
        elif ds == "ppi":
            export_ppi_perturbed()

    print("Done!")
