#!/usr/bin/env python3
"""Summarize verification CSVs into markdown & JSON."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

from verification_schema import STATUSES, validate_status


def _parse_int(v: str | None) -> int:
    if v is None or v == "":
        return 0
    return int(v)


def _parse_float(v: str | None) -> float:
    if v is None or v == "":
        return float("nan")
    return float(v)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["status"] = validate_status(row["status"])
            row["eps"] = float(row["eps"])
            row["runtime_sec"] = _parse_float(row.get("runtime_sec"))
            row["margin_lb"] = _parse_float(row.get("margin_lb"))
            row["clean_correct"] = _parse_int(row.get("clean_correct"))
            row["certified_correct"] = _parse_int(row.get("certified_correct"))
            row["counterexample_available"] = _parse_int(
                row.get("counterexample_available")
            )
            rows.append(row)
    return rows


def summarize_group(rows: list[dict]) -> dict:
    runtimes = np.array([r["runtime_sec"] for r in rows], dtype=float)
    margins = np.array([r["margin_lb"] for r in rows], dtype=float)
    statuses = [r["status"] for r in rows]

    summary = {
        "n": len(rows),
        "clean_accuracy": float(np.mean([r["clean_correct"] for r in rows])),
        "certified_accuracy": float(
            np.mean([r["certified_correct"] for r in rows])
        ),
        "counterexample_rate": float(
            np.mean([r["counterexample_available"] for r in rows])
        ),
        "runtime_median": float(np.nanmedian(runtimes)),
        "runtime_p90": float(np.nanpercentile(runtimes, 90)),
        "margin_median": float(np.nanmedian(margins)),
        "margin_q25": float(np.nanpercentile(margins, 25)),
        "margin_q75": float(np.nanpercentile(margins, 75)),
    }
    for status in STATUSES:
        key = f"{status.lower()}_rate"
        summary[key] = float(np.mean([s == status for s in statuses]))
    return summary


def build_summary(csv_paths: list[str]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)

    for path in csv_paths:
        for row in load_rows(path):
            label = row.get("run_name") or row.get("model_name") or os.path.basename(path)
            groups[(label, row["eps"])].append(row)

    rows = []
    for (label, eps), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        summary = summarize_group(group)
        summary["label"] = label
        summary["eps"] = eps
        summary["backend"] = group[0].get("backend", "")
        summary["model_name"] = group[0].get("model_name", "")
        rows.append(summary)
    return rows


def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def render_markdown(rows: list[dict]) -> str:
    header = (
        "| eps | label | backend | clean_acc | certified_acc | verified | falsified "
        "| unknown | unsupported | error | rt_median | rt_p90 |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines = [header, sep]
    for r in rows:
        line = (
            f"| {r['eps']:.4g} | {r['label']} | {r['backend']} "
            f"| {fmt(r['clean_accuracy'])} "
            f"| {fmt(r['certified_accuracy'])} "
            f"| {fmt(r['verified_rate'])} "
            f"| {fmt(r['falsified_rate'])} "
            f"| {fmt(r['unknown_rate'])} "
            f"| {fmt(r['unsupported_rate'])} "
            f"| {fmt(r['error_rate'])} "
            f"| {fmt(r['runtime_median'])} "
            f"| {fmt(r['runtime_p90'])} |"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CORA verification CSVs.")
    parser.add_argument("--csv", action="append", default=[],
                        help="CSV path; repeat for multiple runs.")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    if not args.csv:
        parser.error("Provide at least one --csv path.")
    csv_paths = list(args.csv)

    rows = build_summary(csv_paths)
    md = render_markdown(rows)
    print(md)

    os.makedirs(args.out_dir, exist_ok=True)

    md_path = os.path.join(args.out_dir, "enzymes_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Written: {md_path}", file=sys.stderr)

    json_path = os.path.join(args.out_dir, "enzymes_summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Written: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
