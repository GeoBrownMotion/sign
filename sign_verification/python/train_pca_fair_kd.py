"""MLP-fair (param-matched to SIGN K=1..4) and MLP-KD on CiteSeer PCA-32."""
from __future__ import annotations
import argparse, json, os, random, math, sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid

SIGN_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SIGN_REPO))
from models import NodeMLP, NodeGCN

SEED = 8
DATA_DIR = "/tmp/citeseer"
PCA_DIM = 32

HPARAMS = {"lr": 0.01, "weight_decay": 5e-4, "max_epochs": 10000,
           "patience": 20, "eval_freq": 10}
GCN_HIDDEN = 64
MLP_HIDDEN = 64
MLP_ACT = "relu"
MLP_DROPOUT = 0.2


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_aligned():
    ds = Planetoid(root=DATA_DIR, name="CiteSeer")
    data = ds[0]
    x_np = data.x.numpy()
    x_train = x_np[data.train_mask.numpy()]
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(x_train)
    x_reduced = pca.transform(x_np)
    data_pca = data.clone()
    data_pca.x = torch.from_numpy(x_reduced.astype(np.float32))
    return data_pca


def train_node_model(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])
    best_loss, best_acc, patience, best_state = float("inf"), 0.0, 0, None
    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()
        if epoch % HPARAMS["eval_freq"] == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                vloss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
                vacc = (out[data.val_mask].argmax(-1) == data.y[data.val_mask]).float().mean().item()
            if vloss < best_loss:
                best_loss, best_acc, patience = vloss, vacc, 0
                best_state = deepcopy(model.state_dict())
            else:
                patience += 1
            if patience >= HPARAMS["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best_acc, best_loss


def fair_mlp_hidden(sign_params, in_ch=32, layers=2, out_ch=6):
    """Hidden width h such that an MLP(in_ch, h, ..., out_ch, layers) has
    approximately `sign_params` total parameters. Solves the quadratic
    (layers-1)*h^2 + (in_ch + layers + out_ch)*h + (out_ch - sign_params) = 0.
    """
    a = layers - 1
    b = in_ch + layers + out_ch
    c = out_ch - sign_params
    if a == 0:
        return max(int((sign_params - out_ch) / b), 1)
    h = int((-b + math.sqrt(max(b**2 - 4*a*c, 0))) / (2*a))
    return max(h, 1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def export_node_mlp(model, data, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    layers = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            layers.append({"type": "lin", "act": "",
                           "W": m.weight.detach().tolist(),
                           "b": m.bias.detach().tolist()})
        elif isinstance(m, nn.ReLU) and layers:
            layers[-1]["act"] = "relu"
        elif isinstance(m, nn.Tanh) and layers:
            layers[-1]["act"] = "tanh"

    meta = {"backend_format": "cora-json-v1", "model_family": "mlp",
            "task_level": "node", "uses_edge_features": False,
            "feature_dim": int(data.x.shape[1])}
    with open(os.path.join(save_dir, "model_export.json"), "w") as f:
        json.dump({"metadata": meta, "layers": layers}, f)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    headers = ["input", "output", "output_label", "target_label", "edge_index", "edge_weight"]
    row = [data.x.tolist(), logits.tolist(), preds.tolist(),
           data.y.tolist(), data.edge_index.tolist(), []]
    with open(os.path.join(save_dir, "data_export.json"), "w") as f:
        json.dump([headers, row], f)
    return (preds == data.y).float().mean().item()


def main():
    out_root = Path("artifacts/citeseer_pca_aligned/d32")
    out_root.mkdir(parents=True, exist_ok=True)

    data = load_aligned()
    in_ch = 32
    out_ch = 6

    # Train teacher GCN first (for KD)
    print("=== Training teacher GCN ===")
    set_seed(SEED)
    gcn = NodeGCN(in_ch, GCN_HIDDEN, out_ch, num_conv_layers=2, num_lin_layers=2,
                  act="relu", dropout=0.5)
    gcn, gcn_acc, _ = train_node_model(gcn, data)
    print(f"  GCN val_acc={gcn_acc:.4f}")

    # Train MLP-fair variants (match parameter count of SIGN k=1..4)
    print("\n=== Training MLP-fair (k=1..4) ===")
    for k in [1, 2, 3, 4]:
        sign_in = (k + 1) * in_ch
        sign_ref = NodeMLP(sign_in, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                           act=MLP_ACT, dropout=0.0)
        sign_params = count_params(sign_ref)
        fair_h = fair_mlp_hidden(sign_params, in_ch=in_ch, layers=2, out_ch=out_ch)
        print(f"  k={k}  target SIGN params={sign_params}  fair hidden={fair_h}")

        set_seed(SEED)
        m = NodeMLP(in_ch, fair_h, out_ch, num_hidden_layers=2,
                    act=MLP_ACT, dropout=MLP_DROPOUT)
        m, acc, loss = train_node_model(m, data)
        d = out_root / f"mlp_fair_k{k}"
        eacc = export_node_mlp(m, data, f"mlp_fair_k{k}", str(d))
        with open(d / "results.json", "w") as f:
            json.dump({"model": f"mlp_fair_k{k}", "val_acc": acc,
                        "val_loss": loss, "hidden": fair_h,
                        "pca_dim": PCA_DIM, "pca_fit_on": "train_mask"}, f, indent=2)
        print(f"  k={k} val_acc={acc:.4f}  export_acc={eacc:.4f}")

    # MLP-KD (distill from GCN)
    print("\n=== Training MLP-KD ===")
    LAM, TEMP = 0.5, 4.0
    set_seed(SEED)
    student = NodeMLP(in_ch, MLP_HIDDEN, out_ch, num_hidden_layers=2,
                      act=MLP_ACT, dropout=MLP_DROPOUT)
    optimizer = torch.optim.Adam(student.parameters(), lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])
    gcn.eval()
    best_loss, best_acc, patience, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        student.train()
        optimizer.zero_grad()
        out = student(data.x, data.edge_index)
        loss_ce = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        with torch.no_grad():
            t_soft = F.softmax(gcn(data.x, data.edge_index)[data.train_mask] / TEMP, dim=-1)
        loss_kd = F.kl_div(F.log_softmax(out[data.train_mask] / TEMP, dim=-1),
                            t_soft, reduction="batchmean") * (TEMP ** 2)
        (LAM * loss_ce + (1 - LAM) * loss_kd).backward()
        optimizer.step()

        if epoch % HPARAMS["eval_freq"] == 0:
            student.eval()
            with torch.no_grad():
                out = student(data.x, data.edge_index)
                vloss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
                vacc = (out[data.val_mask].argmax(-1) == data.y[data.val_mask]).float().mean().item()
            if vloss < best_loss:
                best_loss, best_acc, patience = vloss, vacc, 0
                best_state = deepcopy(student.state_dict())
            else:
                patience += 1
            if patience >= HPARAMS["patience"]:
                break

    student.load_state_dict(best_state)
    d = out_root / "mlp_kd"
    eacc = export_node_mlp(student, data, "mlp_kd", str(d))
    with open(d / "results.json", "w") as f:
        json.dump({"model": "mlp_kd", "val_acc": best_acc,
                    "val_loss": best_loss, "hidden": MLP_HIDDEN,
                    "pca_dim": PCA_DIM, "pca_fit_on": "train_mask",
                    "kd": {"lambda": LAM, "temperature": TEMP}}, f, indent=2)
    print(f"  MLP-KD val_acc={best_acc:.4f}  export_acc={eacc:.4f}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
