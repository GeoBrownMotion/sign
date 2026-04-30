"""Synthetic MLP/GCN/SIGN fixtures for the multi-label verification smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

THRESHOLD = 0.0


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def make_data_json(samples: list[dict]) -> list:
    """Build row-oriented data_export.json content."""
    headers = [
        "input", "output", "output_labels", "target_labels",
        "edge_index", "edge_weight",
    ]
    rows = [headers]
    for s in samples:
        rows.append([
            s["input"],
            [s["output"]],
            s["output_labels"],
            s["target_labels"],
            s["edge_index"],
            s["edge_weight"],
        ])
    return rows


def print_samples(name: str, samples: list[dict]):
    print(f"\n  [{name}] {len(samples)} samples, threshold={THRESHOLD}")
    for i, s in enumerate(samples):
        logits = s["output"]
        labels = s["target_labels"]
        logit_str = ", ".join(f"{v:+.4f}" for v in logits)
        print(f"    sample {i}: logits=[{logit_str}]  labels={labels}")


# ═══════════════════════════════════════════════════════════════════════════
#  MLP fixture  (same as before, but only 3 samples for brevity)
# ═══════════════════════════════════════════════════════════════════════════

def make_mlp_fixture():
    W1 = np.array([
        [ 1.0,  0.5, -0.3,  0.2],
        [-0.4,  1.0,  0.1, -0.5],
        [ 0.3, -0.2,  1.0,  0.4],
        [-0.1,  0.6, -0.5,  0.8],
        [ 0.7, -0.3,  0.2, -0.1],
        [-0.6,  0.4,  0.3,  0.5],
        [ 0.2,  0.1, -0.7,  0.3],
        [ 0.5, -0.8,  0.4, -0.2],
    ], dtype=np.float64)
    b1 = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05])

    W2 = np.array([
        [ 0.8,  0.3, -0.5,  0.2,  0.6, -0.4,  0.1, -0.3],
        [-0.3,  0.7,  0.4, -0.6, -0.2,  0.5,  0.3,  0.1],
        [ 0.5, -0.4,  0.6,  0.3, -0.1,  0.2, -0.5,  0.4],
    ], dtype=np.float64)
    b2 = np.array([0.1, -0.2, 0.15])

    layers = [
        {"type": "lin", "act": "tanh", "W": W1.tolist(), "b": b1.tolist()},
        {"type": "lin", "act": "",     "W": W2.tolist(), "b": b2.tolist()},
    ]

    def forward(x):
        return W2 @ np.tanh(W1 @ x + b1) + b2

    inputs = [
        np.array([ 1.0,  0.5, -0.3,  0.2]),
        np.array([-1.0, -1.0, -1.0, -1.0]),
        np.array([-0.5,  0.7,  0.3, -0.2]),
    ]

    samples = []
    for x in inputs:
        logits = forward(x)
        pred = (logits > THRESHOLD).astype(int).tolist()
        samples.append({
            "input": x.tolist(),
            "output": logits.tolist(),
            "output_labels": pred,
            "target_labels": pred,
            "edge_index": [[0], [0]],
            "edge_weight": [],
        })

    model = {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "mlp",
            "task_level": "multilabel",
            "input_mode": "pooled_vector",
            "feature_dim": 4,
            "num_outputs": 3,
            "uses_edge_features": False,
        },
        "layers": layers,
    }
    return model, samples


# ═══════════════════════════════════════════════════════════════════════════
#  GCN fixture
#  Architecture: GCN(feat=2, out=4)+Tanh -> Linear(4,3) -> global_mean_pool
#  This matches the existing single-label GCN test layout.
# ═══════════════════════════════════════════════════════════════════════════

def _gcn_norm_dense(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} for a dense adjacency matrix (with self-loops)."""
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def _gcn_forward(X, edge_index, num_nodes, W_gcn, b_gcn, W_lin, b_lin):
    """Forward pass: GCN layer + tanh -> Linear -> mean pool."""
    # Build adjacency
    A = np.zeros((num_nodes, num_nodes))
    src, dst = edge_index
    for s, d in zip(src, dst):
        A[s, d] = 1.0
        A[d, s] = 1.0
    A += np.eye(num_nodes)  # self-loops
    A_hat = _gcn_norm_dense(A)

    # GCN layer: A_hat @ X @ W^T + b, then tanh
    H = np.tanh(A_hat @ X @ W_gcn.T + b_gcn)
    # Linear layer (per-node): H @ W_lin^T + b_lin
    Z = H @ W_lin.T + b_lin
    # Global mean pool
    return Z.mean(axis=0)


def make_gcn_fixture():
    # GCN layer: input_feat=2 -> hidden=4, with tanh
    W_gcn = np.array([
        [ 0.8, -0.3],
        [-0.5,  0.9],
        [ 0.4,  0.6],
        [-0.2, -0.7],
    ], dtype=np.float64)  # (4, 2)
    b_gcn = np.array([0.1, -0.1, 0.05, -0.05])

    # Linear layer (per-node before pooling): hidden=4 -> output=3
    W_lin = np.array([
        [ 0.6,  0.3, -0.4,  0.2],
        [-0.3,  0.5,  0.7, -0.1],
        [ 0.4, -0.6,  0.2,  0.5],
    ], dtype=np.float64)  # (3, 4)
    b_lin = np.array([0.1, -0.15, 0.05])

    layers = [
        {"type": "gcn",  "act": "tanh", "W": W_gcn.tolist(), "b": b_gcn.tolist()},
        {"type": "lin",  "act": "",     "W": W_lin.tolist(), "b": b_lin.tolist()},
        {"type": "global_mean_pool", "act": "", "W": [], "b": []},
    ]

    # 3 small graphs with 2-3 nodes, feat_dim=2
    graph_inputs = [
        {
            "X": np.array([[1.0, 0.5], [0.3, -0.4]]),
            "edge_index": [[0, 1], [1, 0]],
        },
        {
            "X": np.array([[-0.5, 0.8], [0.6, -0.3], [0.2, 0.4]]),
            "edge_index": [[0, 1, 1, 2], [1, 0, 2, 1]],
        },
        {
            "X": np.array([[0.7, -0.2], [-0.3, 0.9]]),
            "edge_index": [[0, 1], [1, 0]],
        },
    ]

    samples = []
    for g in graph_inputs:
        X = g["X"]
        ei = g["edge_index"]
        num_nodes = X.shape[0]
        logits = _gcn_forward(X, ei, num_nodes, W_gcn, b_gcn, W_lin, b_lin)
        pred = (logits > THRESHOLD).astype(int).tolist()
        samples.append({
            "input": X.tolist(),
            "output": logits.tolist(),
            "output_labels": pred,
            "target_labels": pred,
            "edge_index": ei,
            "edge_weight": [],
        })

    model = {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "gcn",
            "task_level": "multilabel",
            "uses_edge_features": False,
        },
        "layers": layers,
    }
    return model, samples


# ═══════════════════════════════════════════════════════════════════════════
#  SIGN fixture
#  Architecture: pooled SIGN(k=1) features -> Linear(4,8)+Tanh -> Linear(8,3)
#  input_dim per sample = (1+k) * feat_dim = 2*2 = 4
#  This exercises the SIGN projection path in verification.
# ═══════════════════════════════════════════════════════════════════════════

def _sign_pooled_features(X, edge_index, num_nodes, k=1):
    """Compute pooled SIGN features: [mean(X), mean(A_hat @ X)]."""
    A = np.zeros((num_nodes, num_nodes))
    src, dst = edge_index
    for s, d in zip(src, dst):
        A[s, d] = 1.0
        A[d, s] = 1.0
    A += np.eye(num_nodes)
    A_hat = _gcn_norm_dense(A)

    parts = [X.mean(axis=0)]  # x0: mean of raw features
    M = A_hat.copy()
    for hop in range(1, k + 1):
        parts.append((M @ X).mean(axis=0))
        M = A_hat @ M
    return np.concatenate(parts)


def make_sign_fixture():
    feat_dim = 2
    sign_k = 1
    sign_dim = (1 + sign_k) * feat_dim  # = 4

    # MLP head: Linear(4,8)+Tanh -> Linear(8,3)
    W1 = np.array([
        [ 0.7,  0.4, -0.3,  0.5],
        [-0.2,  0.8,  0.1, -0.4],
        [ 0.5, -0.3,  0.6,  0.2],
        [-0.1,  0.3, -0.5,  0.7],
        [ 0.6, -0.2,  0.4, -0.1],
        [-0.4,  0.5,  0.3,  0.6],
        [ 0.3,  0.1, -0.6,  0.2],
        [ 0.4, -0.7,  0.2, -0.3],
    ], dtype=np.float64)  # (8, 4)
    b1 = np.array([0.1, -0.1, 0.15, -0.15, 0.05, -0.05, 0.1, -0.1])

    W2 = np.array([
        [ 0.5,  0.3, -0.4,  0.2,  0.6, -0.3,  0.1, -0.2],
        [-0.2,  0.6,  0.3, -0.5, -0.1,  0.4,  0.2,  0.1],
        [ 0.4, -0.3,  0.5,  0.3, -0.2,  0.1, -0.4,  0.3],
    ], dtype=np.float64)  # (3, 8)
    b2 = np.array([0.05, -0.1, 0.1])

    layers = [
        {"type": "lin", "act": "tanh", "W": W1.tolist(), "b": b1.tolist()},
        {"type": "lin", "act": "",     "W": W2.tolist(), "b": b2.tolist()},
    ]

    def sign_forward(X, edge_index, num_nodes):
        pooled = _sign_pooled_features(X, edge_index, num_nodes, k=sign_k)
        return W2 @ np.tanh(W1 @ pooled + b1) + b2

    # 3 small graphs
    graph_inputs = [
        {
            "X": np.array([[1.0, 0.5], [0.3, -0.4]]),
            "edge_index": [[0, 1], [1, 0]],
        },
        {
            "X": np.array([[-0.5, 0.8], [0.6, -0.3], [0.2, 0.4]]),
            "edge_index": [[0, 1, 1, 2], [1, 0, 2, 1]],
        },
        {
            "X": np.array([[0.7, -0.2], [-0.3, 0.9]]),
            "edge_index": [[0, 1], [1, 0]],
        },
    ]

    samples = []
    for g in graph_inputs:
        X = g["X"]
        ei = g["edge_index"]
        num_nodes = X.shape[0]
        logits = sign_forward(X, ei, num_nodes)
        pred = (logits > THRESHOLD).astype(int).tolist()
        samples.append({
            "input": X.tolist(),
            "output": logits.tolist(),
            "output_labels": pred,
            "target_labels": pred,
            "edge_index": ei,
            "edge_weight": [],
        })

    model = {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "sign",
            "task_level": "multilabel",
            "input_mode": "sign_pooled_vector",
            "feature_dim": feat_dim,
            "sign_hops": sign_k,
            "sign_operator_family": "multi_scale_concat",
            "num_outputs": 3,
            "uses_edge_features": False,
        },
        "layers": layers,
    }
    return model, samples


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="artifacts/multilabel_smoke")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    for name, make_fn in [("mlp", make_mlp_fixture),
                           ("gcn", make_gcn_fixture),
                           ("sign", make_sign_fixture)]:
        model, samples = make_fn()
        d = out_dir / name
        write_json(d / "model_export.json", model)
        write_json(d / "data_export.json", make_data_json(samples))
        print_samples(name, samples)

    print(f"\nAll fixtures written to {out_dir}/{{mlp,gcn,sign}}/")


if __name__ == "__main__":
    main()
