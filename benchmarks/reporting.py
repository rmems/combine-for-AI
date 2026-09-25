from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.jsonio import ensure_dir, json_safe, write_json
from benchmarks.metrics import MetricsSummary
from benchmarks.telemetry import RoutingMetrics, SystemSnapshot, TelemetrySnapshot


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


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError("no rows to write to csv report")

    fieldnames = list(rows[0].keys())

    def csv_safe(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
            return f"'{value}"
        return value

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: csv_safe(value) for key, value in row.items()} for row in rows
        )


def metrics_to_row(metrics: MetricsSummary) -> dict[str, Any]:
    row = asdict(metrics)
    return row


def telemetry_to_row(telemetry: TelemetrySnapshot | None) -> dict[str, Any]:
    """Flatten telemetry into prefixed CSV/JSON-safe fields."""
    if telemetry is None:
        return _empty_telemetry_row()
    d: dict[str, Any] = {}
    sys_dict = asdict(telemetry.system)
    for k, v in sys_dict.items():
        d[f"telemetry_sys_{k}"] = v
    routing = telemetry.routing or RoutingMetrics()
    d["telemetry_routing_entropy"] = routing.routing_entropy
    d["telemetry_spike_density"] = routing.spike_density
    d["telemetry_latent_stability"] = routing.latent_stability
    d["telemetry_dv_dt_reductions"] = routing.dv_dt_reductions
    d["telemetry_event_rate"] = routing.event_rate
    d["telemetry_firing_rate"] = routing.firing_rate
    d["telemetry_membrane_pressure"] = routing.membrane_pressure
    d["telemetry_saaq_delta_q"] = routing.saaq_delta_q
    d["telemetry_saaq_delta_q_last"] = routing.saaq_delta_q_last
    d["telemetry_saaq_delta_q_legacy"] = routing.saaq_delta_q_legacy
    d["telemetry_saaq_delta_q_v15"] = routing.saaq_delta_q_v15
    d["telemetry_kernel_occupancy"] = telemetry.kernel_occupancy
    d["telemetry_vram_bandwidth_gbps"] = telemetry.vram_bandwidth_gbps
    d["telemetry_notes"] = telemetry.notes
    d["telemetry_saaq_rule"] = telemetry.saaq_rule
    d["telemetry_saaq_model_family"] = telemetry.saaq_model_family
    d["telemetry_saaq_delta_q_trajectory"] = _trajectory_cell(telemetry.saaq_delta_q_trajectory)
    d["telemetry_saaq_delta_q_legacy_trajectory"] = _trajectory_cell(
        telemetry.saaq_delta_q_legacy_trajectory
    )
    d["telemetry_saaq_delta_q_v15_trajectory"] = _trajectory_cell(
        telemetry.saaq_delta_q_v15_trajectory
    )
    return d


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
