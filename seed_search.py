"""
Multi-seed search: run MLP and GCN on seeds 0-19.
Reports val_acc for each seed and the gap (GCN - MLP).
"""
import json
import torch
from torch_geometric.loader import DataLoader

from models import MLP, GCN
from train import load_enzymes, train_epoch, evaluate, set_seed, HPARAMS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = list(range(20))

# GCN config to test
GCN_HIDDEN = 128
GCN_LAYERS = 4

# MLP config (pool-first baseline)
MLP_HIDDEN = 64
MLP_LAYERS = 3


def run(model_name, seed):
    set_seed(seed)
    train_data, val_data, *_ = load_enzymes(seed)
    train_loader = DataLoader(train_data, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=HPARAMS["batch_size"], shuffle=False)
    in_ch = train_data[0].x.shape[1]

    if model_name == "mlp":
        model = MLP(in_ch, MLP_HIDDEN, 6, num_hidden_layers=MLP_LAYERS).to(DEVICE)
    else:
        model = GCN(in_ch, GCN_HIDDEN, 6,
                    num_conv_layers=GCN_LAYERS, num_lin_layers=1, act="tanh").to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=HPARAMS["lr"],
        weight_decay=HPARAMS["weight_decay"], betas=(0.9, 0.999),
    )
    best_loss, best_acc, patience = float("inf"), 0.0, 0
    for epoch in range(1, HPARAMS["max_epochs"] + 1):
        train_epoch(model, train_loader, optimizer, DEVICE)
        if epoch % HPARAMS["eval_freq"] == 0:
            loss, acc = evaluate(model, val_loader, DEVICE)
            if loss < best_loss:
                best_loss, best_acc, patience = loss, acc, 0
            else:
                patience += 1
            if patience >= HPARAMS["patience"]:
                break
    return best_acc


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"GCN: hidden={GCN_HIDDEN}, conv_layers={GCN_LAYERS}, act=tanh")
    print(f"MLP: hidden={MLP_HIDDEN}, layers={MLP_LAYERS}, pool-first\n")
    print(f"{'Seed':>4}  {'MLP':>7}  {'GCN':>7}  {'Gap':>7}")
    print("-" * 34)

    results = []
    for seed in SEEDS:
        mlp_acc = run("mlp", seed)
        gcn_acc = run("gcn", seed)
        gap = gcn_acc - mlp_acc
        results.append({"seed": seed, "mlp": round(mlp_acc, 4),
                         "gcn": round(gcn_acc, 4), "gap": round(gap, 4)})
        print(f"  {seed:2d}  {mlp_acc:>7.4f}  {gcn_acc:>7.4f}  {gap:>+7.4f}")

    print()
    best_gcn = max(results, key=lambda x: x["gcn"])
    best_gap = max(results, key=lambda x: x["gap"])
    print(f"Best GCN:  seed={best_gcn['seed']:2d}, gcn={best_gcn['gcn']:.4f}, mlp={best_gcn['mlp']:.4f}, gap={best_gcn['gap']:+.4f}")
    print(f"Best Gap:  seed={best_gap['seed']:2d}, gcn={best_gap['gcn']:.4f}, mlp={best_gap['mlp']:.4f}, gap={best_gap['gap']:+.4f}")

    with open("/home/hep3/sign/results/seed_search.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/seed_search.json")
