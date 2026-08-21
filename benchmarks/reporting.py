from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.metrics import MetricsSummary
from benchmarks.telemetry import TelemetrySnapshot


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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
