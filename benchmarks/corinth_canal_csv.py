"""CSV and tick parsers for corinth-canal SAAQ run artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.corinth_canal_values import _cell, _mean, _parse_float, _parse_int

LATENT_CSV_HEADER = (
    "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,"
    "saaq_delta_q_prev,saaq_delta_q_target,gpu_temp_c,gpu_power_w,"
    "cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,"
    "saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target"
)
ACTIVITY_PRESSURE_SCALE = 24.0
MEMBRANE_PRESSURE_SCALE = 12.0

_DELTA_Q_FIELDS = (
    ("saaq_delta_q_target", "delta_q"),
    ("saaq_delta_q_legacy_target", "delta_q_legacy"),
    ("saaq_delta_q_v15_target", "delta_q_v15"),
)


@dataclass(frozen=True)
class LatentTelemetrySeries:
    """Aggregated dual-SAAQ trajectory parsed from ``latent_telemetry.csv``."""

    row_count: int
    firing_rate_mean: float | None = None
    membrane_dv_dt_mean: float | None = None
    membrane_pressure_mean: float | None = None
    activity_pressure_mean: float | None = None
    routing_entropy_mean: float | None = None
    delta_q_mean: float | None = None
    delta_q_last: float | None = None
    delta_q_legacy_mean: float | None = None
    delta_q_v15_mean: float | None = None
    delta_q_trajectory: tuple[float, ...] = ()
    delta_q_legacy_trajectory: tuple[float, ...] = ()
    delta_q_v15_trajectory: tuple[float, ...] = ()
    timestamps_ms: tuple[int, ...] = ()


@dataclass(frozen=True)
class TickTelemetrySeries:
    """Lightweight parse of ``tick_telemetry.txt`` key=value lines."""

    tick_count: int
    mean_elapsed_us: float | None = None
    last_best_walker: int | None = None


@dataclass
class _LatentCsvColumns:
    firing_rates: list[float]
    membrane_dv_dts: list[float]
    membrane_pressures: list[float]
    activity_pressures: list[float]
    routing_entropies: list[float]
    delta_q: list[float]
    delta_q_legacy: list[float]
    delta_q_v15: list[float]
    timestamps: list[int]


def membrane_pressure(membrane_dv_dt: float) -> float:
    """SAAQ 1.0 membrane pressure: ``clamp(membrane_dv_dt / 12, -1, 1)``."""

    return max(-1.0, min(1.0, membrane_dv_dt / MEMBRANE_PRESSURE_SCALE))


def activity_pressure(firing_rate_hz: float) -> float:
    """SAAQ 1.0 activity pressure: ``clamp(avg_pop_firing_rate_hz / 24, 0, 1)``."""

    return max(0.0, min(1.0, firing_rate_hz / ACTIVITY_PRESSURE_SCALE))


def parse_latent_telemetry_csv(path: Path) -> LatentTelemetrySeries:
    """Parse dual-SAAQ ``latent_telemetry.csv``. Missing columns are ignored."""
    columns = _LatentCsvColumns(
        firing_rates=[],
        membrane_dv_dts=[],
        membrane_pressures=[],
        activity_pressures=[],
        routing_entropies=[],
        delta_q=[],
        delta_q_legacy=[],
        delta_q_v15=[],
        timestamps=[],
    )
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {name for name in (reader.fieldnames or []) if name}
        for row in reader:
            _ingest_csv_row(columns, row, fieldnames)
            row_count += 1
    return LatentTelemetrySeries(
        row_count=row_count,
        firing_rate_mean=_mean(columns.firing_rates),
        membrane_dv_dt_mean=_mean(columns.membrane_dv_dts),
        membrane_pressure_mean=_mean(columns.membrane_pressures),
        activity_pressure_mean=_mean(columns.activity_pressures),
        routing_entropy_mean=_mean(columns.routing_entropies),
        delta_q_mean=_mean(columns.delta_q),
        delta_q_last=columns.delta_q[-1] if columns.delta_q else None,
        delta_q_legacy_mean=_mean(columns.delta_q_legacy),
        delta_q_v15_mean=_mean(columns.delta_q_v15),
        delta_q_trajectory=tuple(columns.delta_q),
        delta_q_legacy_trajectory=tuple(columns.delta_q_legacy),
        delta_q_v15_trajectory=tuple(columns.delta_q_v15),
        timestamps_ms=tuple(columns.timestamps),
    )


def parse_tick_telemetry(path: Path) -> TickTelemetrySeries:
    """Parse ``tick=… best_walker=… elapsed_us=…`` lines. Bad lines are skipped."""

    ticks = 0
    elapsed: list[float] = []
    last_walker: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_tick_line(line)
            if parsed is None:
                continue
            ticks += 1
            walker, elapsed_us = parsed
            if walker is not None:
                last_walker = walker
            if elapsed_us is not None:
                elapsed.append(elapsed_us)
    return TickTelemetrySeries(
        tick_count=ticks,
        mean_elapsed_us=_mean(elapsed),
        last_best_walker=last_walker,
    )


def _append_if(values: list[Any], parsed: Any) -> None:
    if parsed is not None:
        values.append(parsed)


def _ingest_csv_row(
    columns: _LatentCsvColumns, row: dict[str, str | None], fieldnames: set[str]
) -> None:
    _append_if(columns.timestamps, _parse_int(_cell(row, "timestamp_ms")))
    rate = _parse_float(_cell(row, "avg_pop_firing_rate_hz"))
    _append_if(columns.firing_rates, rate)
    if rate is not None:
        columns.activity_pressures.append(activity_pressure(rate))
    dv_dt = _parse_float(_cell(row, "membrane_dv_dt"))
    _append_if(columns.membrane_dv_dts, dv_dt)
    if dv_dt is not None:
        columns.membrane_pressures.append(membrane_pressure(dv_dt))
    _append_if(columns.routing_entropies, _parse_float(_cell(row, "routing_entropy")))
    _append_delta_q(columns, row, fieldnames)


def _append_delta_q(
    columns: _LatentCsvColumns, row: dict[str, str | None], fieldnames: set[str]
) -> None:
    parsed = {
        name: _parse_float(_cell(row, name))
        for name, _ in _DELTA_Q_FIELDS
        if name in fieldnames
    }
    if "saaq_delta_q_legacy_target" not in parsed and "saaq_delta_q_v15_target" not in parsed:
        _append_if(columns.delta_q, parsed.get("saaq_delta_q_target"))
        return
    if None in parsed.values():
        return
    for name, attr in _DELTA_Q_FIELDS:
        if name in parsed:
            getattr(columns, attr).append(parsed[name])


def _parse_tick_line(line: str) -> tuple[int | None, float | None] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields: dict[str, str] = {}
    for token in stripped.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        fields[key] = value
    if _parse_int(fields.get("tick")) is None:
        return None
    return _parse_int(fields.get("best_walker")), _parse_float(fields.get("elapsed_us"))
