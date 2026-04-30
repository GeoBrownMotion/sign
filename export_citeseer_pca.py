"""Export PCA-reduced data + perturbed variants for CiteSeer PCA models."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid

SEED = 8
PCA_DIM = 32
PERTURB_SEED = 42
FEATURE_EPS = [0.001, 0.005, 0.01, 0.05]
EDGE_FRAC = [0.001, 0.005, 0.01, 0.05]

SIGN_ROOT = Path(__file__).resolve().parent
SAVE_ROOT = SIGN_ROOT / "results" / "citeseer_pca"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_citeseer_pca():
    ds = Planetoid(root="/tmp/citeseer", name="CiteSeer")
    data = ds[0]

    x_np = data.x.numpy().astype(np.float64)
    x_train = x_np[data.train_mask.numpy()]

    set_seed(SEED)
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(x_train)
    x_reduced = pca.transform(x_np).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {x_np.shape} -> {x_reduced.shape}  (explained: {explained:.4f})")

    data.x = torch.tensor(x_reduced, dtype=torch.float32)
    return data


def patch_metadata(model_dir, model_family):
    path = model_dir / "model_export.json"
    with open(path) as f:
        obj = json.load(f)

    if "metadata" not in obj:
        obj["metadata"] = {}

    obj["metadata"].update({
        "backend_format": "cora-json-v1",
        "model_family": model_family,
        "task_level": "node",
        "uses_edge_features": False,
        "feature_dim": PCA_DIM,
        "pca_dim": PCA_DIM,
    })

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Patched metadata: {path}")


def export_data(data, model_dir, model_name):
    model_path = model_dir / "model_export.json"
    with open(model_path) as f:
        model_obj = json.load(f)

    has_gcn = any(l["type"] == "gcn" for l in model_obj["layers"])

    if has_gcn:
        from models import NodeGCN
        gcn_results = json.load(open(model_dir / "results.json"))
        hp = gcn_results["hparams"]
        model = NodeGCN(
            in_channels=PCA_DIM, hidden_channels=hp["hidden"],
            out_channels=6, num_conv_layers=hp["conv_layers"],
            num_lin_layers=hp["lin_layers"],
            act="relu", dropout=0.0,
        )
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            logits = model(data.x, data.edge_index)

        preds = logits.argmax(dim=1)
        correct = (preds == data.y).sum().item()
        total = data.y.shape[0]

        headers = ["input", "output", "output_label", "target_label",
                    "edge_index", "edge_weight"]
        row = [
            data.x.tolist(),
            logits.tolist(),
            preds.tolist(),
            data.y.tolist(),
            data.edge_index.tolist(),
            [],
        ]
        print(f"  {model_name} (GCN): {total} nodes, acc={correct/total:.4f}")
    else:
        modules = []
        for l in model_obj["layers"]:
            if l["type"] == "lin":
                W = torch.tensor(l["W"], dtype=torch.float32)
                b = torch.tensor(l["b"], dtype=torch.float32)
                lin = nn.Linear(W.shape[1], W.shape[0])
                lin.weight.data = W
                lin.bias.data = b
                modules.append(lin)
                act = l.get("act", "")
                if act == "relu":
                    modules.append(nn.ReLU())
                elif act == "tanh":
                    modules.append(nn.Tanh())
        model = nn.Sequential(*modules)
        model.eval()

        with torch.no_grad():
            logits = model(data.x)

        preds = logits.argmax(dim=1)
        correct = (preds == data.y).sum().item()
        total = data.y.shape[0]

        headers = ["input", "output", "output_label", "target_label",
                    "edge_index", "edge_weight"]
        row = [
            data.x.tolist(),
            logits.tolist(),
            preds.tolist(),
            data.y.tolist(),
            data.edge_index.tolist(),
            [],
        ]
        print(f"  {model_name}: {total} nodes, acc={correct/total:.4f}")

    data_path = model_dir / "data_export.json"
    with open(data_path, "w") as f:
        json.dump([headers, row], f)
    print(f"  Exported data -> {data_path}")


def perturb_features(x, eps):
    noise = torch.empty_like(x).uniform_(-eps, eps)
    return x + noise


def perturb_edges(edge_index, num_nodes, frac):
    src, dst = edge_index[0], edge_index[1]
    self_loop_mask = src == dst
    non_self_src = src[~self_loop_mask]
    non_self_dst = dst[~self_loop_mask]

    u = torch.min(non_self_src, non_self_dst)
    v = torch.max(non_self_src, non_self_dst)
    unique_edges = list(set((u[i].item(), v[i].item()) for i in range(u.shape[0])))

    num_remove = max(1, int(len(unique_edges) * frac)) if frac > 0 else 0
    remove_set = set(random.sample(unique_edges, min(num_remove, len(unique_edges))))

    keep_src, keep_dst = [src[self_loop_mask]], [dst[self_loop_mask]]
    for i in range(non_self_src.shape[0]):
        s, d = non_self_src[i].item(), non_self_dst[i].item()
        if (min(s, d), max(s, d)) not in remove_set:
            keep_src.append(torch.tensor([s]))
            keep_dst.append(torch.tensor([d]))

    return torch.stack([torch.cat(keep_src), torch.cat(keep_dst)], dim=0)


def export_perturbed(data):
    out_dir = SIGN_ROOT / "results" / "perturbed" / "citeseer_pca"
    os.makedirs(out_dir, exist_ok=True)

    headers = ["input", "output", "output_label", "target_label",
               "edge_index", "edge_weight"]

    for eps in FEATURE_EPS:
        set_seed(PERTURB_SEED)
        x_pert = perturb_features(data.x, eps)
        row = [x_pert.tolist(), [], [], data.y.tolist(), data.edge_index.tolist(), []]
        path = out_dir / f"data_feat_eps{eps}.json"
        with open(path, "w") as f:
            json.dump([headers, row], f)
        print(f"  data_feat_eps{eps}.json: {data.num_nodes} nodes")

    for frac in EDGE_FRAC:
        set_seed(PERTURB_SEED)
        ei_pert = perturb_edges(data.edge_index, data.num_nodes, frac)
        row = [data.x.tolist(), [], [], data.y.tolist(), ei_pert.tolist(), []]
        path = out_dir / f"data_edge_frac{frac}.json"
        with open(path, "w") as f:
            json.dump([headers, row], f)
        print(f"  data_edge_frac{frac}.json: edges {data.edge_index.shape[1]} -> {ei_pert.shape[1]}")

    row = [data.x.tolist(), [], [], data.y.tolist(), data.edge_index.tolist(), []]
    path = out_dir / "data_clean.json"
    with open(path, "w") as f:
        json.dump([headers, row], f)
    print(f"  data_clean.json: {data.num_nodes} nodes")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 60)
    print("CiteSeer PCA - Export & Perturb")
    print("=" * 60)

    data = load_citeseer_pca()

    print("\n--- Patching metadata ---")
    patch_metadata(SAVE_ROOT / "mlp", "mlp")
    patch_metadata(SAVE_ROOT / "gcn", "gcn")

    print("\n--- Exporting data ---")
    export_data(data, SAVE_ROOT / "mlp", "mlp")
    export_data(data, SAVE_ROOT / "gcn", "gcn")

    print("\n--- Exporting perturbed datasets ---")
    export_perturbed(data)

    print("\nDone!")
