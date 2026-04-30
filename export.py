"""
Export trained MLP / GCN models to verifier-compatible JSON.

Output format mirrors the reference paper (Ladner et al., 2025):
  {
    "layers": [
      {"type": "gcn"|"lin"|"global_mean_pool",
       "act":  "tanh" | "",
       "W":    [[float, ...], ...],   # shape: out_channels × in_channels
       "b":    [float, ...]},         # shape: out_channels
      ...
    ]
  }

Layer ordering:
  GCN : [gcn_0, ..., gcn_k]  +  [lin_0, ..., lin_m]  +  [global_mean_pool]
  MLP : [global_mean_pool]   +  [lin_0, ..., lin_m]

The global_mean_pool entry signals to the verifier how the graph-level
aggregation is positioned in the inference graph.
"""

import json
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLP, GCN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_to_dict(layer: nn.Linear, act: str) -> dict:
    """Serialise a torch.nn.Linear into the export dict."""
    return {
        "type": "lin",
        "act":  act,
        "W":    layer.weight.detach().cpu().tolist(),   # (out, in)
        "b":    layer.bias.detach().cpu().tolist(),     # (out,)
    }


def _pool_entry() -> dict:
    return {"type": "global_mean_pool", "act": "", "W": [], "b": []}


def _activation_name(module: Optional[nn.Module]) -> str:
    if module is None:
        return ""
    if isinstance(module, nn.Tanh):
        return "tanh"
    if isinstance(module, nn.ReLU):
        return "relu"
    return ""


def _export_linear_stack(seq: nn.Sequential) -> list[dict]:
    layers = []
    children = list(seq.children())
    for idx, module in enumerate(children):
        if not isinstance(module, nn.Linear):
            continue
        act = ""
        if idx + 1 < len(children):
            act = _activation_name(children[idx + 1])
        layers.append(_linear_to_dict(module, act))
    return layers


# ---------------------------------------------------------------------------
# Model-specific exporters
# ---------------------------------------------------------------------------

def export_gcn(model: GCN) -> dict:
    """
    Export a GCN model.
    Layer order: [gcn_0 … gcn_k] + [lin_0 … lin_m] + [global_mean_pool]
    This matches the reference JSON format exactly.
    """
    layers = []

    # --- GCN (message-passing) layers ---
    num_conv = len(model.convs)
    for i, conv in enumerate(model.convs):
        is_last_conv = (i == num_conv - 1)
        # After the last GCN there may be additional lin layers;
        # tanh is always applied after every GCN layer in our model.
        layers.append({
            "type": "gcn",
            "act":  "tanh",
            "W":    conv.lin.weight.detach().cpu().tolist(),  # (out, in)
            "b":    conv.bias.detach().cpu().tolist(),        # (out,)
        })

    # --- Linear layers before pooling ---
    layers.extend(_export_linear_stack(model.lins))

    # --- Global mean pool (marker at end, matching reference) ---
    layers.append(_pool_entry())

    return {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "gcn",
            "task_level": "graph",
            "uses_edge_features": False,
        },
        "layers": layers,
    }


def export_mlp(model: MLP) -> dict:
    """
    Export a MLP model.
    Layer order: [global_mean_pool] + [lin_0 … lin_m]
    The pool is placed first because mean-pooling is applied to node
    features before the MLP runs.
    """
    layers = [_pool_entry()]

    layers.extend(_export_linear_stack(model.net))

    return {
        "metadata": {
            "backend_format": "cora-json-v1",
            "model_family": "mlp",
            "task_level": "graph",
            "uses_edge_features": False,
        },
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_model(model_name: str,
                 model: nn.Module,
                 save_dir: str) -> str:
    """
    Serialise *model* to JSON and write to *save_dir*/model_export.json.
    Returns the path of the written file.
    """
    if isinstance(model, GCN):
        export_dict = export_gcn(model)
    elif isinstance(model, MLP):
        export_dict = export_mlp(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "model_export.json")
    with open(out_path, "w") as f:
        json.dump(export_dict, f, indent=2)

    # Quick sanity summary
    print(f"  Exported {model_name.upper()} → {out_path}")
    for i, l in enumerate(export_dict["layers"]):
        w_shape = f"({len(l['W'])}×{len(l['W'][0]) if l['W'] else 0})"
        print(f"    Layer {i}: type={l['type']:<18} act={l['act']:<6} "
              f"W={w_shape}  b=({len(l['b'])})")

    return out_path


def export_data(model_name: str,
                model: nn.Module,
                dataset: list,
                save_dir: str) -> str:
    """
    Export the model-matched validation split used by the latest baselines.
    """
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

    model.eval()
    device = next(model.parameters()).device
    correct = 0

    with torch.no_grad():
        for data in dataset:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            target = data.y.view(-1).to(device)

            logits = model(x, edge_index, batch).view(-1)
            pred = int(logits.argmax().item())
            target_label = int(target.item())
            loss = float(F.cross_entropy(logits.unsqueeze(0), target).item())
            correct += int(pred == target_label)

            rows.append([
                data.x.detach().cpu().tolist(),
                [logits.detach().cpu().tolist()],
                pred,
                target_label,
                loss,
                data.edge_index.detach().cpu().tolist(),
                [],
            ])

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "data_export.json")
    with open(out_path, "w") as f:
        json.dump(rows, f)

    accuracy = correct / len(dataset) if dataset else float("nan")
    print(
        f"  Exported {model_name.upper()} val data → {out_path} "
        f"({len(dataset)} samples, clean_acc={accuracy:.4f})"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from train import load_enzymes, MLP_HPARAMS, GCN_HPARAMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn", "all"], default="all")
    parser.add_argument("--results_dir", default="/home/hep3/sign/results")
    args = parser.parse_args()

    models_to_export = ["mlp", "gcn"] if args.model == "all" else [args.model]

    train_data, val_data, *_ = load_enzymes(seed=8)
    in_channels  = train_data[0].x.shape[1]
    out_channels = 6

    for name in models_to_export:
        run_dir   = os.path.join(args.results_dir, name)
        ckpt_path = os.path.join(run_dir, "model.pt")

        if not os.path.exists(ckpt_path):
            print(f"  [skip] No checkpoint found for {name} at {ckpt_path}")
            continue

        if name == "mlp":
            model = MLP(in_channels, MLP_HPARAMS["hidden_channels"],
                        out_channels, MLP_HPARAMS["num_layers"])
        else:
            model = GCN(in_channels, GCN_HPARAMS["hidden_channels"],
                        out_channels, GCN_HPARAMS["num_layers"],
                        GCN_HPARAMS["num_lin_layers"])

        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        export_model(name, model, run_dir)
        export_data(name, model, val_data, run_dir)
