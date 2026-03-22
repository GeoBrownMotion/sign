"""
infer_perturbed.py

Loads trained GCN and MLP models, runs inference on every perturbed data file,
and writes predictions back into model-specific output directories:

  results/perturbed/enzymes/{gcn,mlp}/data_*.json
  results/perturbed/ppi/{gcn,mlp}/data_*.json

Each output file keeps the same row format as the input but with:
  - output:       model logits (ENZYMES: list[6]) or sigmoid probs (PPI: list[N][121])
  - output_label: predicted class int (ENZYMES) or binary list[N][121] (PPI)
  - loss:         cross-entropy (ENZYMES) or binary-cross-entropy (PPI), averaged over nodes

Usage:
    python infer_perturbed.py
"""

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from models import MLP, GCN, NodeMLP, NodeGCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = "/home/hep3/sign/results"
PERTURBED_DIR = "/home/hep3/sign/results/perturbed"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_mlp(hidden, num_hidden_layers, path):
    m = MLP(in_channels=21, hidden_channels=hidden, out_channels=6,
            num_hidden_layers=num_hidden_layers)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.eval().to(DEVICE)

def _load_node_mlp(hidden, num_hidden_layers, act, path):
    m = NodeMLP(in_channels=50, hidden_channels=hidden, out_channels=121,
                num_hidden_layers=num_hidden_layers, act=act, dropout=0.0)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.eval().to(DEVICE)


def load_enzymes_models():
    gcn_cfg = json.load(open(f"{RESULTS_DIR}/gcn/results.json"))["hparams"]
    gcn = GCN(
        in_channels=21,
        hidden_channels=gcn_cfg["hidden_channels"],
        out_channels=6,
        num_conv_layers=gcn_cfg["num_layers"],
        num_lin_layers=gcn_cfg.get("num_lin_layers", 1),
        act="tanh",
    )
    gcn.load_state_dict(torch.load(f"{RESULTS_DIR}/gcn/model.pt", map_location=DEVICE))
    gcn.eval().to(DEVICE)

    # Fair MLP hidden sizes derived from model_export.json (W shape: out×in)
    fair_hidden = {1: 259, 2: 262, 3: 266, 4: 269}

    models = {"gcn": gcn}
    models["mlp"]    = _load_mlp(256, 4, f"{RESULTS_DIR}/mlp/model.pt")
    models["mlp_kd"] = _load_mlp(256, 4, f"{RESULTS_DIR}/mlp_kd/model.pt")
    for k in [1, 2, 3, 4]:
        models[f"mlp_fair_k{k}"] = _load_mlp(fair_hidden[k], 4,
                                               f"{RESULTS_DIR}/mlp_fair_k{k}/model.pt")
    return models


def load_ppi_models():
    gcn_cfg = json.load(open(f"{RESULTS_DIR}/ppi/gcn/results.json"))["hparams"]
    gcn = NodeGCN(
        in_channels=50,
        hidden_channels=gcn_cfg["hidden"],
        out_channels=121,
        num_conv_layers=gcn_cfg["conv_layers"],
        num_lin_layers=gcn_cfg["lin_layers"],
        act=gcn_cfg["act"],
        dropout=0.0,
    )
    gcn.load_state_dict(torch.load(f"{RESULTS_DIR}/ppi/gcn/model.pt", map_location=DEVICE))
    gcn.eval().to(DEVICE)

    # Fair MLP hidden sizes from results.json
    fair_hidden = {1: 533, 2: 553, 3: 573, 4: 592}

    models = {"gcn": gcn}
    models["mlp"]    = _load_node_mlp(512, 2, "relu", f"{RESULTS_DIR}/ppi/mlp/model.pt")
    models["mlp_kd"] = _load_node_mlp(512, 2, "relu", f"{RESULTS_DIR}/ppi/mlp_kd/model.pt")
    for k in [1, 2, 3, 4]:
        models[f"mlp_fair_k{k}"] = _load_node_mlp(fair_hidden[k], 2, "relu",
                                                    f"{RESULTS_DIR}/ppi/mlp_fair_k{k}/model.pt")
    return models


# ---------------------------------------------------------------------------
# ENZYMES inference (graph classification)
# ---------------------------------------------------------------------------

def infer_enzymes_file(src_path, models, out_dir):
    with open(src_path) as f:
        data = json.load(f)

    header = data[0]
    rows   = data[1:]
    col    = {name: i for i, name in enumerate(header)}

    for model_name, model in models.items():
        result_rows = []
        for row in rows:
            x          = torch.tensor(row[col["input"]], dtype=torch.float32, device=DEVICE)
            edge_index = torch.tensor(row[col["edge_index"]], dtype=torch.long, device=DEVICE)
            batch      = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
            target     = int(row[col["target_label"]])

            with torch.no_grad():
                logits = model(x, edge_index, batch)  # (1, 6)

            pred = int(logits.argmax(dim=-1).item())
            loss = float(F.cross_entropy(logits, torch.tensor([target], device=DEVICE)).item())

            new_row = list(row)
            new_row[col["output"]]       = logits.squeeze(0).tolist()
            new_row[col["output_label"]] = pred
            new_row[col["loss"]]         = loss
            result_rows.append(new_row)

        out_path = os.path.join(out_dir, model_name, os.path.basename(src_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([header] + result_rows, f)

        n_correct = sum(
            1 for r in result_rows
            if r[col["output_label"]] == r[col["target_label"]]
        )
        acc = n_correct / len(result_rows)
        print(f"    [{model_name}] {os.path.basename(src_path):30s}  acc={acc:.4f}")


# ---------------------------------------------------------------------------
# PPI inference (multi-label node classification)
# ---------------------------------------------------------------------------

def infer_ppi_file(src_path, models, out_dir):
    with open(src_path) as f:
        data = json.load(f)

    header = data[0]
    rows   = data[1:]
    col    = {name: i for i, name in enumerate(header)}

    for model_name, model in models.items():
        result_rows = []
        for row in rows:
            x          = torch.tensor(row[col["input"]], dtype=torch.float32, device=DEVICE)
            edge_index = torch.tensor(row[col["edge_index"]], dtype=torch.long, device=DEVICE)
            targets    = torch.tensor(row[col["target_label"]], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits = model(x, edge_index)  # (N, 121)

            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).long()
            loss   = float(F.binary_cross_entropy_with_logits(logits, targets).item())

            new_row = list(row)
            new_row[col["output"]]       = probs.tolist()
            new_row[col["output_label"]] = preds.tolist()
            # PPI files have no 'loss' column — skip if missing
            if "loss" in col:
                new_row[col["loss"]] = loss
            result_rows.append(new_row)

        out_path = os.path.join(out_dir, model_name, os.path.basename(src_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([header] + result_rows, f)

        # Compute micro-F1
        all_pred = np.concatenate([np.array(r[col["output_label"]]) for r in result_rows])
        all_true = np.concatenate([np.array(r[col["target_label"]]) for r in result_rows])
        from sklearn.metrics import f1_score
        f1 = f1_score(all_true, all_pred, average="micro", zero_division=0)
        print(f"    [{model_name}] {os.path.basename(src_path):30s}  micro-F1={f1:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_perturbed_files(dataset_dir):
    return sorted([
        os.path.join(dataset_dir, fn)
        for fn in os.listdir(dataset_dir)
        if fn.endswith(".json")
    ])


if __name__ == "__main__":
    print("=" * 60)
    print("ENZYMES")
    print("=" * 60)
    enzymes_models = load_enzymes_models()
    enzymes_files  = get_perturbed_files(f"{PERTURBED_DIR}/enzymes")
    enzymes_out    = f"{PERTURBED_DIR}/enzymes"
    for fpath in enzymes_files:
        infer_enzymes_file(fpath, enzymes_models, enzymes_out)

    print()
    print("=" * 60)
    print("PPI")
    print("=" * 60)
    ppi_models = load_ppi_models()
    ppi_files  = get_perturbed_files(f"{PERTURBED_DIR}/ppi")
    ppi_out    = f"{PERTURBED_DIR}/ppi"
    for fpath in ppi_files:
        infer_ppi_file(fpath, ppi_models, ppi_out)

    print()
    print("Done. Results written to results/perturbed/{enzymes,ppi}/{gcn,mlp}/")
