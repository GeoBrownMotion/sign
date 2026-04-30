"""Verification result statuses and CSV schema."""
from __future__ import annotations

from dataclasses import dataclass

STATUSES = (
    "VERIFIED",
    "FALSIFIED",
    "UNKNOWN",
    "UNSUPPORTED",
    "ERROR",
)

CSV_FIELDS = (
    "backend",
    "run_name",
    "dataset",
    "model_name",
    "model_family",
    "task_level",
    "sample_id",
    "eps",
    "status",
    "status_reason",
    "y_true",
    "y_ref",
    "y_clean_pred",
    "clean_correct",
    "certified_correct",
    "counterexample_available",
    "runtime_sec",
    "margin_lb",
    "num_nodes",
    "num_edges",
    "self_loops_added",
    "spec_type",
    "input_set_type",
)


def validate_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"Unknown verification status: {status}")
    return status


@dataclass(frozen=True)
class StatusSummary:
    verified: int
    falsified: int
    unknown: int
    unsupported: int
    error: int


def flags_from_status(status: str) -> dict[str, int]:
    status = validate_status(status)
    return {
        "verified": int(status == "VERIFIED"),
        "falsified": int(status == "FALSIFIED"),
        "unknown": int(status == "UNKNOWN"),
        "unsupported": int(status == "UNSUPPORTED"),
        "error": int(status == "ERROR"),
    }
