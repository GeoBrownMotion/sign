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
import torch
import torch.nn as nn

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


def _act_str(is_last: bool) -> str:
    """All hidden layers use tanh; the output layer has no activation."""
    return "" if is_last else "tanh"


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

    # --- Linear layers after pooling ---
    # model.lins is a Sequential; extract nn.Linear sub-modules in order.
    lin_modules = [m for m in model.lins.modules()
                   if isinstance(m, nn.Linear)]
    num_lin = len(lin_modules)
    for i, lin in enumerate(lin_modules):
        is_last = (i == num_lin - 1)
        layers.append(_linear_to_dict(lin, _act_str(is_last)))

    # --- Global mean pool (marker at end, matching reference) ---
    layers.append(_pool_entry())

    return {"layers": layers}


def export_mlp(model: MLP) -> dict:
    """
    Export a MLP model.
    Layer order: [global_mean_pool] + [lin_0 … lin_m]
    The pool is placed first because mean-pooling is applied to node
    features before the MLP runs.
    """
    layers = [_pool_entry()]

    lin_modules = [m for m in model.net.modules()
                   if isinstance(m, nn.Linear)]
    num_lin = len(lin_modules)
    for i, lin in enumerate(lin_modules):
        is_last = (i == num_lin - 1)
        layers.append(_linear_to_dict(lin, _act_str(is_last)))

    return {"layers": layers}


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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from train import load_enzymes, HPARAMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn", "all"], default="all")
    parser.add_argument("--results_dir", default="/home/hep3/sign/results")
    args = parser.parse_args()

    models_to_export = ["mlp", "gcn"] if args.model == "all" else [args.model]

    train_data, *_ = load_enzymes()
    in_channels  = train_data[0].x.shape[1]
    out_channels = 6

    for name in models_to_export:
        run_dir    = os.path.join(args.results_dir, name)
        ckpt_path  = os.path.join(run_dir, "model.pt")

        if not os.path.exists(ckpt_path):
            print(f"  [skip] No checkpoint found for {name} at {ckpt_path}")
            continue

        if name == "mlp":
            model = MLP(in_channels, HPARAMS["hidden_channels"],
                        out_channels, HPARAMS["num_layers"])
        else:
            model = GCN(in_channels, HPARAMS["hidden_channels"],
                        out_channels, HPARAMS["num_layers"],
                        HPARAMS["num_lin_layers"])

        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        export_model(name, model, run_dir)
