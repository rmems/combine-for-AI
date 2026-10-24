from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from benchmarks.jsonio import ensure_dir, json_safe, write_json
from benchmarks.metrics import MetricsSummary
from benchmarks.telemetry import SystemSnapshot, TelemetrySnapshot, telemetry_to_dict
from combine_for_ai.csvio import write_csv


_TELEMETRY_SCALAR_KEYS = (
    "routing_entropy",
    "spike_density",
    "latent_stability",
    "dv_dt_reductions",
    "event_rate",
    "firing_rate",
    "membrane_pressure",
    "saaq_delta_q",
    "saaq_delta_q_last",
    "saaq_delta_q_legacy",
    "saaq_delta_q_v15",
    "kernel_occupancy",
    "vram_bandwidth_gbps",
    "notes",
    "saaq_rule",
    "saaq_model_family",
    "saaq_delta_q_trajectory",
    "saaq_delta_q_legacy_trajectory",
    "saaq_delta_q_v15_trajectory",
)


def metrics_to_row(metrics: MetricsSummary) -> dict[str, Any]:
    row = asdict(metrics)
    return row


def telemetry_to_row(telemetry: TelemetrySnapshot | None) -> dict[str, Any]:
    """Flatten telemetry into prefixed CSV/JSON-safe fields."""
    if telemetry is None:
        return _empty_telemetry_row()
    row: dict[str, Any] = {}
    for key, value in telemetry_to_dict(telemetry).items():
        if key == "gpu_metrics":
            continue
        if key == "telemetry_notes":
            row["telemetry_notes"] = value
            continue
        if key.endswith("_trajectory"):
            row[f"telemetry_{key}"] = _trajectory_cell(tuple(value) if value else None)
            continue
        row[f"telemetry_{key}"] = value
    return row


def _trajectory_cell(values: tuple[float, ...] | None) -> str | None:
    if not values:
        return None
    return json.dumps(list(values))


def _empty_telemetry_row() -> dict[str, Any]:
    row = {f"telemetry_sys_{key}": None for key in SystemSnapshot.__dataclass_fields__}
    for key in _TELEMETRY_SCALAR_KEYS:
        row[f"telemetry_{key}"] = None
    return row


__all__ = [
    "ensure_dir",
    "json_safe",
    "write_json",
    "write_csv",
    "metrics_to_row",
    "telemetry_to_row",
]
