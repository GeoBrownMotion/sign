"""SIGN sweep on CiteSeer PCA-32 (PCA fit on train mask, matching the baseline)."""
from __future__ import annotations
import argparse, json, os, random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid

from precompute_sign import build_operator_sequence

SEED = 8
DATA_DIR = "/tmp/citeseer"
PCA_DIM = 32

HPARAMS = {"lr": 0.01, "weight_decay": 5e-4, "max_epochs": 10000,
           "patience": 20, "eval_freq": 10}
MLP_HIDDEN = 64
MLP_LAYERS = 2
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


def precompute_sign_pst(data, p, s, t, ppr_alpha=0.05):
    X = data.x.cpu().numpy().astype(np.float64)
    ei = data.edge_index.cpu().numpy().astype(np.int64)
    n = data.num_nodes
    ops = build_operator_sequence(ei, n, p, s, t, ppr_alpha)
    parts = [X if op is None else op.dot(X) for _, op in ops]
    return torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))


class NodeMLP(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, num_layers=2, act="relu", dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_ch, hidden), nn.ReLU() if act=="relu" else nn.Tanh()]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU() if act=="relu" else nn.Tanh())
            if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)


def train_model(model, data, sign_x=None):
    x_input = sign_x if sign_x is not None else data.x
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"],
                                 weight_decay=HPARAMS["weight_decay"])
    best_loss, best_acc, patience, best_state = float("inf"), 0.0, 0, None

    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_input)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()

        if epoch % HPARAMS["eval_freq"] == 0:
            model.eval()
            with torch.no_grad():
                out = model(x_input)
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


def export_model_and_data(model, data, sign_x, name, save_dir, metadata_extra=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
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
            "feature_dim": int(data.x.shape[1]), "pca_dim": PCA_DIM,
            "pca_fit_on": "train_mask"}
    if metadata_extra: meta.update(metadata_extra)
    with open(os.path.join(save_dir, "model_export.json"), "w") as f:
        json.dump({"metadata": meta, "layers": layers}, f)

    with torch.no_grad():
        logits = model(sign_x if sign_x is not None else data.x)
    preds = logits.argmax(dim=1)

    headers = ["input", "output", "output_label", "target_label", "edge_index", "edge_weight"]
    row = [data.x.tolist(), logits.tolist(), preds.tolist(),
           data.y.tolist(), data.edge_index.tolist(), []]
    with open(os.path.join(save_dir, "data_export.json"), "w") as f:
        json.dump([headers, row], f)

    return (preds == data.y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign_sweep", action="store_true")
    parser.add_argument("--out_root", type=str,
                        default="artifacts/citeseer_pca_aligned")
    args = parser.parse_args()

    set_seed(SEED)
    data = load_aligned()
    in_ch = data.x.shape[1]
    out_ch = int(data.y.max().item()) + 1
    out_root = Path(args.out_root) / f"d{PCA_DIM}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"CiteSeer PCA-32 aligned: {data.num_nodes} nodes, {in_ch} features")

    # MLP
    print("\n=== MLP ===")
    set_seed(SEED)
    mlp = NodeMLP(in_ch, MLP_HIDDEN, out_ch, MLP_LAYERS, MLP_ACT, MLP_DROPOUT)
    mlp, acc, loss = train_model(mlp, data)
    eacc = export_model_and_data(mlp, data, None, "mlp", str(out_root / "mlp"))
    with open(out_root / "mlp" / "results.json", "w") as f:
        json.dump({"model": "mlp_aligned", "val_acc": acc, "val_loss": loss,
                    "pca_dim": PCA_DIM, "pca_fit_on": "train_mask"}, f, indent=2)
    print(f"  val_acc={acc:.4f}  export_acc={eacc:.4f}")

    # SIGN configs
    if args.sign_sweep:
        cfgs = [(p,s,t) for p in range(5) for s in range(5) for t in range(5)
                if not (p==0 and s==0 and t==0)]
    else:
        cfgs = [(1,2,2), (2,0,0), (0,3,0), (1,0,0), (1,1,0), (4,4,3)]

    for p, s, t in cfgs:
        tag = f"sign_p{p}_s{s}_t{t}"
        print(f"\n=== {tag} ===")
        sign_x = precompute_sign_pst(data, p, s, t)
        sign_dim = sign_x.shape[1]

        set_seed(SEED)
        model = NodeMLP(sign_dim, MLP_HIDDEN, out_ch, MLP_LAYERS, MLP_ACT, MLP_DROPOUT)
        model, acc, loss = train_model(model, data, sign_x)

        meta = {"model_family": "sign", "input_mode": "sign_node_vector",
                "sign_config": {"p": p, "s": s, "t": t, "ppr_alpha": 0.05}}
        eacc = export_model_and_data(model, data, sign_x, tag,
                                     str(out_root / tag), meta)

        with open(out_root / tag / "results.json", "w") as f:
            json.dump({"model": tag, "val_acc": acc, "val_loss": loss,
                        "sign_dim": sign_dim, "pca_dim": PCA_DIM,
                        "sign_config": {"p":p,"s":s,"t":t}}, f, indent=2)
        print(f"  val_acc={acc:.4f}  export_acc={eacc:.4f}  sign_dim={sign_dim}")

    print(f"\nAll done. Models in {out_root}/")


if __name__ == "__main__":
    main()
