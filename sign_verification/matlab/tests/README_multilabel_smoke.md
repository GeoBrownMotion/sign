# Multi-Label Verification Smoke Test

Feasibility test for threshold-based multi-label verification using CORA.

## Quick Start

```bash
# from sign_verification/
python python/make_multilabel_smoke_fixture.py
matlab -batch "cd matlab; addpath('tests'); test_multilabel_smoke"
```

## What It Tests

Tiny MLP / GCN / SIGN fixtures with 3 binary outputs, against a per-output
threshold polytope spec (`y_i ≥ 0` for positives, `y_j ≤ 0` for negatives) at
ε ∈ {0, 0.01, 0.05, 0.10}. Both full label-set and per-label verification.

## Files

| File | Purpose |
|------|---------|
| `python/make_multilabel_smoke_fixture.py` | Generates synthetic MLP/GCN/SIGN fixtures |
| `matlab/build_cora_multilabel_spec.m` | Threshold-based polytope spec |
| `matlab/tests/test_multilabel_smoke.m` | End-to-end smoke test |
