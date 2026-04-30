"""SIGN(p,s,t) operator bank (Â^k / PPR^k / triangle^k) and pooled features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import savemat

from parse_export import load_data_export


def _make_undirected_adj(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    src, dst = edge_index[0], edge_index[1]
    src_full = np.concatenate([src, dst])
    dst_full = np.concatenate([dst, src])
    mask = (src_full >= 0) & (src_full < n) & (dst_full >= 0) & (dst_full < n)
    src_full = src_full[mask]
    dst_full = dst_full[mask]
    pairs = np.stack([src_full, dst_full], axis=1)
    pairs = np.unique(pairs, axis=0)
    src_full, dst_full = pairs[:, 0], pairs[:, 1]
    data = np.ones(len(src_full), dtype=np.float64)
    return sp.csr_matrix((data, (src_full, dst_full)), shape=(n, n))


def _gcn_norm(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(d, dtype=np.float64)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def _row_norm(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.zeros_like(d, dtype=np.float64)
    mask = d > 0
    d_inv[mask] = 1.0 / d[mask]
    return sp.diags(d_inv) @ A


def _ppr_operator(A_with_self_loops: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"ppr alpha must be in (0, 1], got {alpha}")
    n = A_with_self_loops.shape[0]
    T = _row_norm(A_with_self_loops).tocsc()
    I = sp.eye(n, dtype=np.float64, format="csc")
    system = I - (1.0 - alpha) * T
    dense_sol = spla.spsolve(system, I.toarray())
    return sp.csr_matrix(alpha * dense_sol)


def _triangle_operator(A: sp.csr_matrix, keep_self_loops: bool = False) -> sp.csr_matrix:
    B = A.copy().astype(np.float64).tocsr()
    B.data[:] = 1.0
    if not keep_self_loops:
        B.setdiag(0.0)
        B.eliminate_zeros()
    common_neighbors = (B @ B).tocsr()
    tri = B.multiply(common_neighbors)
    tri.eliminate_zeros()
    return _row_norm(tri).tocsr()


def build_operator_sequence(
    edge_index: np.ndarray,
    n_nodes: int,
    p: int,
    s: int,
    t: int,
    ppr_alpha: float,
    triangle_keep_self_loops: bool = False,
) -> list[tuple[str, sp.csr_matrix | None]]:
    """Returns [(name, op)]; op=None marks the raw-feature branch x0."""
    if min(p, s, t) < 0:
        raise ValueError(f"p, s, t must be non-negative, got {(p, s, t)}")

    A = _make_undirected_adj(edge_index, n_nodes)
    I = sp.eye(n_nodes, dtype=np.float64, format="csr")

    ops: list[tuple[str, sp.csr_matrix | None]] = [("x0", None)]

    if p > 0:
        simple = _gcn_norm(A + I).tocsr()
        cur = simple
        for hop in range(1, p + 1):
            ops.append((f"p{hop}", cur))
            cur = (cur @ simple).tocsr()

    if s > 0:
        ppr = _ppr_operator(A + I, alpha=ppr_alpha).tocsr()
        cur = ppr
        for hop in range(1, s + 1):
            ops.append((f"s{hop}", cur))
            cur = (cur @ ppr).tocsr()

    if t > 0:
        triangle = _triangle_operator(A, keep_self_loops=triangle_keep_self_loops).tocsr()
        cur = triangle
        for hop in range(1, t + 1):
            ops.append((f"t{hop}", cur))
            cur = (cur @ triangle).tocsr()

    return ops


def compute_pooled_sign_vector(
    X: np.ndarray,
    edge_index: np.ndarray,
    p: int,
    s: int,
    t: int,
    ppr_alpha: float,
    triangle_keep_self_loops: bool = False,
) -> tuple[np.ndarray, list[str]]:
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    op_sequence = build_operator_sequence(
        edge_index=edge_index,
        n_nodes=X.shape[0],
        p=p,
        s=s,
        t=t,
        ppr_alpha=ppr_alpha,
        triangle_keep_self_loops=triangle_keep_self_loops,
    )

    pooled_parts: list[np.ndarray] = []
    op_names: list[str] = []
    for name, op in op_sequence:
        H = X if op is None else op.dot(X)
        pooled_parts.append(H.mean(axis=0))
        op_names.append(name)

    g_sign = np.concatenate(pooled_parts, axis=0).astype(np.float64)
    return g_sign, op_names


def sign_feature_dim(feature_dim: int, p: int, s: int, t: int) -> int:
    return (1 + p + s + t) * feature_dim


# ── main pipeline ────────────────────────────────────────────────────────────

def precompute(
    samples: list,
    p: int,
    s: int,
    t: int,
    ppr_alpha: float,
    triangle_keep_self_loops: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, list[str]]:
    d: Optional[int] = None
    g_sign_list: List[np.ndarray] = []
    g_base_list: List[np.ndarray] = []
    y_pred_list: List[int] = []
    y_true_list: List[int] = []
    has_y_true = True
    op_names: list[str] = ["x0"]

    for idx, sample in enumerate(samples):
        X = sample["X"]
        edge_index = sample["edge_index"]

        if X.ndim != 2:
            raise ValueError(f"sample {idx}: X must be 2-D, got shape {X.shape}")
        n_nodes, feat_dim = X.shape

        if d is None:
            d = feat_dim
        elif feat_dim != d:
            raise ValueError(
                f"sample {idx}: feature dim mismatch (expected {d}, got {feat_dim})"
            )

        if edge_index.size == 0:
            raise ValueError(f"sample {idx}: edge_index is empty")

        g_sign, op_names = compute_pooled_sign_vector(
            X=X,
            edge_index=edge_index,
            p=p,
            s=s,
            t=t,
            ppr_alpha=ppr_alpha,
            triangle_keep_self_loops=triangle_keep_self_loops,
        )
        g_base = X.mean(axis=0)

        # sanity
        if not np.all(np.isfinite(g_sign)):
            raise ValueError(f"sample {idx}: g_sign contains NaN/Inf")

        g_sign_list.append(g_sign)
        g_base_list.append(g_base)
        y_pred_list.append(sample["y_pred"])
        if "y_true" in sample:
            y_true_list.append(sample["y_true"])
        else:
            has_y_true = False

    base_dim = 0 if d is None else d
    sign_dim = sign_feature_dim(base_dim, p, s, t)
    g_sign_all = np.stack(g_sign_list, axis=0)   # [N, sign_dim]
    g_base_all = np.stack(g_base_list, axis=0)   # [N, d]
    y_pred_all = np.array(y_pred_list, dtype=np.int64)  # [N]
    y_true_all = np.array(y_true_list, dtype=np.int64) if has_y_true else None

    return g_sign_all, g_base_all, y_pred_all, y_true_all, sign_dim, op_names


def main() -> None:
    parser = argparse.ArgumentParser(description="SIGN precompute -> sign_pack.mat")
    parser.add_argument("--input", type=str, default="artifacts/raw/data_export.json")
    parser.add_argument("--out", type=str, default="artifacts/derived/sign_pack.mat")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--ppr_alpha", type=float, default=0.05)
    parser.add_argument("--triangle_keep_self_loops", action="store_true")
    parser.add_argument("--eps_list", type=str, default="0.01,0.02,0.05")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    eps_list = np.array([float(e) for e in args.eps_list.split(",")], dtype=np.float64)

    print(f"Loading {args.input} ...")
    samples = load_data_export(args.input)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    N = len(samples)
    d = samples[0]["X"].shape[1] if N > 0 else 0
    sign_dim = sign_feature_dim(d, args.p, args.s, args.t)
    print(
        f"N={N}, d={d}, p={args.p}, s={args.s}, t={args.t}, "
        f"alpha={args.ppr_alpha}, sign_dim={sign_dim}"
    )

    g_sign_all, g_base_all, y_pred_all, y_true_all, sign_dim, op_names = precompute(
        samples,
        p=args.p,
        s=args.s,
        t=args.t,
        ppr_alpha=args.ppr_alpha,
        triangle_keep_self_loops=args.triangle_keep_self_loops,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "g_sign_all": g_sign_all,
        "g_base_all": g_base_all,
        "y_pred_all": y_pred_all,
        "eps_list": eps_list,
        "meta": {
            "K": args.p,
            "p": args.p,
            "s": args.s,
            "t": args.t,
            "ppr_alpha": args.ppr_alpha,
            "triangle_keep_self_loops": args.triangle_keep_self_loops,
            "d": d,
            "sign_dim": sign_dim,
            "operator_names": np.asarray(op_names, dtype=object),
        },
    }
    if y_true_all is not None:
        payload["y_true_all"] = y_true_all

    savemat(str(out_path), payload)
    print(f"Saved sign_pack.mat -> {out_path}  (N={N})")


if __name__ == "__main__":
    main()
