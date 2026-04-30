"""Parse data_export.json: row 0 = field names, rows 1.. = records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Dict

import numpy as np


def load_data_export(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        data = obj
    elif isinstance(obj, dict) and "data" in obj:
        data = obj["data"]
    else:
        raise ValueError("JSON must be a list [fields, row1, ...] or a dict with key 'data'")

    if not data:
        return []

    field_names = data[0]
    samples = []

    for row in data[1:]:
        raw = dict(zip(field_names, row))
        if "input" not in raw:
            raise KeyError("missing field 'input'")
        if "output_label" not in raw:
            raise KeyError("missing field 'output_label'")

        edge_index = raw.get("edge_index")
        if edge_index is None or (isinstance(edge_index, (list, np.ndarray)) and len(edge_index) == 0):
            raise ValueError("edge_index is empty or None")

        edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_index.size == 0:
            raise ValueError("edge_index is empty or None")
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)

        X = np.asarray(raw["input"], dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        y_pred = int(raw["output_label"])

        sample = {
            "X": X,
            "y_pred": y_pred,
            "edge_index": edge_index,
        }

        if "target_label" in raw and raw["target_label"] is not None:
            sample["y_true"] = int(raw["target_label"])

        if "output" in raw and raw["output"] is not None:
            out = np.asarray(raw["output"], dtype=np.float64)
            if out.ndim >= 1:
                sample["logits"] = out.ravel() if out.ndim == 1 else np.asarray(out[0], dtype=np.float64)
            else:
                sample["logits"] = np.array([float(out)], dtype=np.float64)

        samples.append(sample)

    return samples
