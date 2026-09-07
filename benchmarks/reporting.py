from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.metrics import MetricsSummary
from benchmarks.telemetry import TelemetrySnapshot


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``NaN`` and ``Infinity`` are not JSON: Python writes them as bare literals
    that other parsers reject. Report metrics reach the writer straight from
    upstream experiment files, and a degenerate block legitimately produces a
    ``NaN`` cosine, so the writer has to make them representable.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a report as standard JSON that any conforming parser can read."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        # allow_nan=False turns any non-finite value json_safe missed into a
        # ValueError here rather than an unreadable report discovered later.
        json.dump(
            json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError("no rows to write to csv report")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_to_row(metrics: MetricsSummary) -> dict[str, Any]:
    row = asdict(metrics)
    return row


def telemetry_to_row(telemetry: TelemetrySnapshot | None) -> dict[str, Any]:
    """Flatten telemetry into prefixed CSV/JSON-safe fields."""
    if telemetry is None:
        return {
            "telemetry_sys_cpu_count_logical": None,
            "telemetry_sys_cpu_count_physical": None,
            "telemetry_sys_memory_total_gb": None,
            "telemetry_sys_memory_available_gb": None,
            "telemetry_sys_gpu_count": None,
            "telemetry_sys_gpu_names": None,
            "telemetry_sys_gpu_driver_version": None,
            "telemetry_sys_cuda_version": None,
            "telemetry_sys_platform": None,
            "telemetry_sys_python_version": None,
            "telemetry_routing_entropy": None,
            "telemetry_spike_density": None,
            "telemetry_latent_stability": None,
            "telemetry_dv_dt_reductions": None,
            "telemetry_event_rate": None,
            "telemetry_kernel_occupancy": None,
            "telemetry_vram_bandwidth_gbps": None,
            "telemetry_notes": None,
        }
    d: dict[str, Any] = {}
    sys_dict = asdict(telemetry.system)
    for k, v in sys_dict.items():
        d[f"telemetry_sys_{k}"] = v
    d["telemetry_routing_entropy"] = telemetry.routing.routing_entropy if telemetry.routing else None
    d["telemetry_spike_density"] = telemetry.routing.spike_density if telemetry.routing else None
    d["telemetry_latent_stability"] = telemetry.routing.latent_stability if telemetry.routing else None
    d["telemetry_dv_dt_reductions"] = telemetry.routing.dv_dt_reductions if telemetry.routing else None
    d["telemetry_event_rate"] = telemetry.routing.event_rate if telemetry.routing else None
    d["telemetry_kernel_occupancy"] = telemetry.kernel_occupancy
    d["telemetry_vram_bandwidth_gbps"] = telemetry.vram_bandwidth_gbps
    d["telemetry_notes"] = telemetry.notes
    return d
