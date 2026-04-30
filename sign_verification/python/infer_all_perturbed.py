"""Run all model families on perturbed data; write perturbed_results.json per model."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGN_REPO = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
PERT = SIGN_REPO / "results" / "perturbed"

sys.path.insert(0, str(SIGN_REPO))


# ─── Model loaders ─────────────────────────────────────────────────────────

def load_mlp_head_from_json(model_json_path):
    """Reconstruct a nn.Sequential MLP from cora-json-v1 model_export.json."""
    with open(model_json_path) as f:
        obj = json.load(f)
    layers_json = obj["layers"]
    # strip pooling
    layers_json = [l for l in layers_json
                   if l["type"] not in ("global_mean_pool", "global_add_pool")]

    modules = []
    for l in layers_json:
        if l["type"] == "lin":
            W = torch.tensor(l["W"], dtype=torch.float32)
            b = torch.tensor(l["b"], dtype=torch.float32)
            lin = nn.Linear(W.shape[1], W.shape[0])
            lin.weight.data = W; lin.bias.data = b
            modules.append(lin)
            act = l.get("act", "")
            if act == "relu": modules.append(nn.ReLU())
            elif act == "tanh": modules.append(nn.Tanh())
    return nn.Sequential(*modules)


def load_sign_config(results_json_path):
    with open(results_json_path) as f:
        return json.load(f)["sign_config"]


# ─── ENZYMES SIGN inference ─────────────────────────────────────────────────

def infer_enzymes_sign(sign_tags, perturbed_files):
    from precompute_sign import compute_pooled_sign_vector

    sign_base = ARTIFACTS / "sweep_fair_valloss_pst"

    for tag in sign_tags:
        model_dir = sign_base / tag
        if not (model_dir / "model_export.json").exists():
            print(f"  SKIP {tag}: missing model")
            continue

        model = load_mlp_head_from_json(model_dir / "model_export.json")
        model.eval()
        sc = load_sign_config(model_dir / "results.json")

        results = {"model": f"sign_{tag}", "dataset": "enzymes",
                   "metric": "accuracy", "perturbed_results": {}}

        for pf in perturbed_files:
            with open(pf) as f: data = json.load(f)
            header = data[0]; rows = data[1:]
            col = {h:i for i,h in enumerate(header)}

            correct = 0
            for row in rows:
                X = np.array(row[col["input"]], dtype=np.float64)
                ei = np.array(row[col["edge_index"]], dtype=np.int64)
                target = int(row[col["target_label"]])

                # Compute pooled SIGN feature vector
                pooled, _ = compute_pooled_sign_vector(
                    X=X, edge_index=ei,
                    p=sc["p"], s=sc["s"], t=sc["t"],
                    ppr_alpha=sc.get("ppr_alpha", 0.05),
                    triangle_keep_self_loops=sc.get("triangle_keep_self_loops", False),
                )
                x_in = torch.tensor(pooled, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x_in).squeeze(0)
                pred = int(logits.argmax().item())
                correct += int(pred == target)

            acc = correct / len(rows)
            key = os.path.basename(pf).replace("data_","").replace(".json","")
            results["perturbed_results"][key] = round(acc, 4)
            print(f"  [{tag}] {key}: acc={acc:.4f}")

        # Save
        out_dir = ARTIFACTS / "sweep_fair_valloss_pst" / tag
        with open(out_dir / "perturbed_results.json", "w") as f:
            json.dump(results, f, indent=2)


# ─── CiteSeer raw inference (teammate models) ───────────────────────────────

def infer_citeseer_raw():
    from models import NodeMLP, NodeGCN
    import torch_geometric
    from torch_geometric.datasets import Planetoid

    # Load teammate models
    print("Loading CiteSeer raw models...")

    model_specs = [
        ("mlp",         "node_mlp",      64, 2, "relu", 0.2, False),
        ("mlp_fair",    "node_mlp_fair", None, 2, "relu", 0.2, False),
        ("mlp_fair_k1", "node_mlp_fair_k1", None, 2, "relu", 0.2, False),
        ("mlp_fair_k2", "node_mlp_fair_k2", None, 2, "relu", 0.2, False),
        ("mlp_fair_k3", "node_mlp_fair_k3", None, 2, "relu", 0.2, False),
        ("mlp_fair_k4", "node_mlp_fair_k4", None, 2, "relu", 0.2, False),
        ("mlp_kd",      "node_mlp_kd",   64, 2, "relu", 0.2, False),
    ]

    models = {}
    for name, export_name, hidden, num_layers, act, dropout, is_gcn in model_specs:
        path = SIGN_REPO / "results" / "citeseer" / name
        rjson = path / "results.json"
        if not rjson.exists():
            print(f"  SKIP citeseer/{name}: no results.json"); continue
        with open(rjson) as f: r = json.load(f)
        hp = r.get("hparams", {})
        h = hp.get("hidden", hidden)

        m = NodeMLP(3703, h, 6, num_hidden_layers=num_layers, act=act, dropout=0.0)
        m.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        m.eval()
        models[name] = m

    # GCN
    gcn_dir = SIGN_REPO / "results" / "citeseer" / "gcn"
    with open(gcn_dir / "results.json") as f:
        gcn_r = json.load(f)["hparams"]
    state = torch.load(gcn_dir / "model.pt", map_location="cpu")
    lin_layers = sum(1 for k in state.keys() if k.startswith("lins.") and k.endswith(".weight"))
    gcn = NodeGCN(3703, gcn_r["hidden"], 6,
                  num_conv_layers=gcn_r["conv_layers"],
                  num_lin_layers=lin_layers, act="relu", dropout=0.0)
    gcn.load_state_dict(state); gcn.eval()
    models["gcn"] = gcn

    # Infer on perturbed data
    pert_dir = PERT / "citeseer"
    pert_files = sorted(pert_dir.glob("data_*.json"))

    for name, model in models.items():
        results = {"model": name, "dataset": "citeseer",
                   "metric": "accuracy", "perturbed_results": {}}

        for pf in pert_files:
            with open(pf) as f: data = json.load(f)
            row = data[1]
            col = {h:i for i,h in enumerate(data[0])}

            X = torch.tensor(row[col["input"]], dtype=torch.float32)
            ei = torch.tensor(row[col["edge_index"]], dtype=torch.long)
            y = torch.tensor(row[col["target_label"]], dtype=torch.long)

            with torch.no_grad():
                if name == "gcn":
                    out = model(X, ei)
                else:
                    out = model(X, ei)  # NodeMLP ignores edge_index
            preds = out.argmax(dim=-1)
            acc = (preds == y).float().mean().item()

            key = pf.name.replace("data_","").replace(".json","")
            results["perturbed_results"][key] = round(acc, 4)
            print(f"  [citeseer/{name}] {key}: acc={acc:.4f}")

        out_dir = SIGN_REPO / "results" / "citeseer" / name
        with open(out_dir / "perturbed_results.json", "w") as f:
            json.dump(results, f, indent=2)


# ─── CiteSeer PCA inference ─────────────────────────────────────────────────

def infer_citeseer_pca():
    """Teammate has PCA MLP and GCN. We also have PCA SIGN models."""
    from models import NodeMLP, NodeGCN

    pert_dir = PERT / "citeseer_pca"
    pert_files = sorted(pert_dir.glob("data_*.json"))

    # Teammate's PCA MLP and GCN
    tm_dir = SIGN_REPO / "results" / "citeseer_pca"

    # MLP
    with open(tm_dir / "mlp" / "results.json") as f: r = json.load(f)
    hp = r["hparams"]
    mlp = NodeMLP(32, hp["hidden"], 6, num_hidden_layers=2, act=hp["act"], dropout=0.0)
    mlp.load_state_dict(torch.load(tm_dir / "mlp" / "model.pt", map_location="cpu"))
    mlp.eval()

    # GCN
    with open(tm_dir / "gcn" / "results.json") as f: r = json.load(f)
    hp = r["hparams"]
    state = torch.load(tm_dir / "gcn" / "model.pt", map_location="cpu")
    lin_layers = sum(1 for k in state.keys() if k.startswith("lins.") and k.endswith(".weight"))
    gcn = NodeGCN(32, hp["hidden"], 6,
                  num_conv_layers=hp["conv_layers"],
                  num_lin_layers=lin_layers, act="relu", dropout=0.0)
    gcn.load_state_dict(state); gcn.eval()

    teammate_models = {"mlp": mlp, "gcn": gcn}

    # Teammate's new PCA fair/KD variants
    for name in ["mlp_fair_k1", "mlp_fair_k2", "mlp_fair_k3", "mlp_fair_k4", "mlp_kd"]:
        d = tm_dir / name
        if not (d / "results.json").exists():
            continue
        with open(d / "results.json") as f: rr = json.load(f)
        hh = rr["hparams"]["hidden"]
        m = NodeMLP(32, hh, 6, num_hidden_layers=2, act="relu", dropout=0.0)
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        m.eval()
        teammate_models[name] = m

    for name, model in teammate_models.items():
        results = {"model": name, "dataset": "citeseer_pca",
                   "metric": "accuracy", "perturbed_results": {}}
        for pf in pert_files:
            with open(pf) as f: data = json.load(f)
            row = data[1]; col = {h:i for i,h in enumerate(data[0])}
            X = torch.tensor(row[col["input"]], dtype=torch.float32)
            ei = torch.tensor(row[col["edge_index"]], dtype=torch.long)
            y = torch.tensor(row[col["target_label"]], dtype=torch.long)
            with torch.no_grad():
                out = model(X, ei)
            preds = out.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            key = pf.name.replace("data_","").replace(".json","")
            results["perturbed_results"][key] = round(acc, 4)
            print(f"  [citeseer_pca/{name}] {key}: acc={acc:.4f}")
        with open(tm_dir / name / "perturbed_results.json", "w") as f:
            json.dump(results, f, indent=2)

    from precompute_sign import build_operator_sequence

    sign_base = ARTIFACTS / "citeseer_pca_aligned" / "d32"
    sign_tags = ["sign_p3_s4_t1", "sign_p2_s3_t3", "sign_p3_s3_t2",
                  "sign_p1_s2_t2", "sign_p2_s0_t0", "sign_p0_s3_t0"]

    for tag in sign_tags:
        model_dir = sign_base / tag
        if not (model_dir / "model_export.json").exists():
            continue
        model = load_mlp_head_from_json(model_dir / "model_export.json")
        model.eval()
        sc = load_sign_config(model_dir / "results.json")

        results = {"model": tag, "dataset": "citeseer_pca",
                   "metric": "accuracy", "perturbed_results": {}}
        for pf in pert_files:
            with open(pf) as f: data = json.load(f)
            row = data[1]; col = {h:i for i,h in enumerate(data[0])}
            X = np.array(row[col["input"]], dtype=np.float64)
            ei = np.array(row[col["edge_index"]], dtype=np.int64)
            y = np.array(row[col["target_label"]])

            n = X.shape[0]
            ops = build_operator_sequence(ei, n, sc["p"], sc["s"], sc["t"],
                                          sc.get("ppr_alpha", 0.05))
            parts = [X if op is None else op.dot(X) for _, op in ops]
            sign_X = np.concatenate(parts, axis=1).astype(np.float32)

            x_in = torch.tensor(sign_X)
            with torch.no_grad():
                out = model(x_in)
            preds = out.argmax(dim=-1).numpy()
            acc = float((preds == y).mean())

            key = pf.name.replace("data_","").replace(".json","")
            results["perturbed_results"][key] = round(acc, 4)
            print(f"  [citeseer_pca/{tag}] {key}: acc={acc:.4f}")

        with open(model_dir / "perturbed_results.json", "w") as f:
            json.dump(results, f, indent=2)


# ─── PPI SIGN inference ─────────────────────────────────────────────────────

def infer_ppi_sign(sign_tags):
    from precompute_sign import build_operator_sequence
    from sklearn.metrics import f1_score

    sign_base = ARTIFACTS / "ppi_sign_pst"
    pert_dir = PERT / "ppi"
    pert_files = sorted(pert_dir.glob("data_*.json"))

    for tag in sign_tags:
        model_dir = sign_base / tag
        if not (model_dir / "model_export.json").exists():
            continue
        model = load_mlp_head_from_json(model_dir / "model_export.json")
        model.eval()
        sc = load_sign_config(model_dir / "results.json")

        results = {"model": f"sign_{tag}", "dataset": "ppi",
                   "metric": "micro_f1", "perturbed_results": {}}

        for pf in pert_files:
            with open(pf) as f: data = json.load(f)
            row = data[1]; col = {h:i for i,h in enumerate(data[0])}
            X = np.array(row[col["input"]], dtype=np.float64)
            ei = np.array(row[col["edge_index"]], dtype=np.int64)
            y = np.array(row[col["target_label"]])  # [N, 121]

            n = X.shape[0]
            ops = build_operator_sequence(ei, n, sc["p"], sc["s"], sc["t"],
                                          sc.get("ppr_alpha", 0.05))
            parts = [X if op is None else op.dot(X) for _, op in ops]
            sign_X = np.concatenate(parts, axis=1).astype(np.float32)

            x_in = torch.tensor(sign_X)
            with torch.no_grad():
                logits = model(x_in)
            preds = (torch.sigmoid(logits) > 0.5).int().numpy()
            f1 = f1_score(y, preds, average="micro", zero_division=0)

            key = pf.name.replace("data_","").replace(".json","")
            results["perturbed_results"][key] = round(f1, 4)
            print(f"  [ppi/sign_{tag}] {key}: F1={f1:.4f}")

        with open(model_dir / "perturbed_results.json", "w") as f:
            json.dump(results, f, indent=2)


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enzymes_pert = sorted((PERT / "enzymes").glob("data_*.json"))
    enzymes_sign = ["p0_s1_t0", "p2_s2_t1", "p0_s4_t0", "p3_s4_t1", "p1_s3_t1"]

    print("=" * 60)
    print("ENZYMES SIGN")
    print("=" * 60)
    infer_enzymes_sign(enzymes_sign, enzymes_pert)

    print("\n" + "=" * 60)
    print("CiteSeer PCA-32")
    print("=" * 60)
    infer_citeseer_pca()

    print("\n" + "=" * 60)
    print("PPI SIGN")
    print("=" * 60)
    ppi_sign = ["p1_s0_t0", "p2_s0_t0", "p0_s2_t2", "p4_s2_t2", "p4_s4_t3"]
    infer_ppi_sign(ppi_sign)

    print("\nAll done.")
