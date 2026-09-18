"""Parsers for corinth-canal SAAQ run artifacts.

corinth-canal's ``saaq_latent_calibration`` writes a per-run directory:

- ``latent_telemetry.csv`` — dual-SAAQ trajectory (1.0 + 1.5)
- ``summary.json`` — compact ``ExperimentSummary``
- ``run_manifest.json`` — full ``ExperimentManifest``
- ``tick_telemetry.txt`` — per-tick spike / timing lines

Missing pieces are skipped rather than treated as errors so a combine
benchmark can still finish when only GOZ1 (or nothing upstream) is present.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never

# Matches ``src/latent.rs`` in rmems/corinth-canal.
LATENT_CSV_HEADER = (
    "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,"
    "saaq_delta_q_prev,saaq_delta_q_target,gpu_temp_c,gpu_power_w,"
    "cpu_tctl_c,cpu_package_power_w,saaq_delta_q_legacy_prev,"
    "saaq_delta_q_legacy_target,saaq_delta_q_v15_prev,saaq_delta_q_v15_target"
)
ACTIVITY_PRESSURE_SCALE = 24.0
MEMBRANE_PRESSURE_SCALE = 12.0
SAAQ_ARTIFACT_VERSION = "corinth-canal.saaq.v1"

CANONICAL_LATENT_CSV = "latent_telemetry.csv"
CANONICAL_SUMMARY = "summary.json"
CANONICAL_MANIFEST = "run_manifest.json"
CANONICAL_TICKS = "tick_telemetry.txt"

ArtifactKind = Literal["summary", "manifest", "legacy"]


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


@dataclass(frozen=True)
class CorinthCanalRun:
    """Normalized SAAQ run used to overlay combine telemetry and metrics."""

    experiment_id: str
    artifact_version: str = SAAQ_ARTIFACT_VERSION
    routing_entropy: float | None = None
    spike_density: float | None = None
    event_rate: float | None = None
    latent_stability: float | None = None
    dv_dt_reductions: float | None = None
    firing_rate: float | None = None
    membrane_pressure: float | None = None
    membrane_dv_dt: float | None = None
    saaq_delta_q: float | None = None
    saaq_delta_q_last: float | None = None
    saaq_delta_q_legacy: float | None = None
    saaq_delta_q_v15: float | None = None
    saaq_delta_q_trajectory: tuple[float, ...] = ()
    saaq_delta_q_legacy_trajectory: tuple[float, ...] = ()
    saaq_delta_q_v15_trajectory: tuple[float, ...] = ()
    saaq_rule: str | None = None
    saaq_primary_rule: str | None = None
    saaq_dual_emit: bool | None = None
    model_family: str | None = None
    model_slug: str | None = None
    projection_mode: str | None = None
    ticks_completed: int | None = None
    latent_rows: int | None = None
    raw_path: str | None = None
    summary: dict[str, Any] | None = None
    run_manifest: dict[str, Any] | None = None
    skipped: tuple[str, ...] = ()


def membrane_pressure(membrane_dv_dt: float) -> float:
    """SAAQ 1.0 membrane pressure: ``clamp(membrane_dv_dt / 12, -1, 1)``."""

    return max(-1.0, min(1.0, membrane_dv_dt / MEMBRANE_PRESSURE_SCALE))


def activity_pressure(firing_rate_hz: float) -> float:
    """SAAQ 1.0 activity pressure: ``clamp(avg_pop_firing_rate_hz / 24, 0, 1)``."""

    return max(0.0, min(1.0, firing_rate_hz / ACTIVITY_PRESSURE_SCALE))


def try_load_corinth_canal(path: Path) -> CorinthCanalRun | None:
    """Load a run directory or file; return None when absent or unreadable."""

    if not path.exists():
        return None
    try:
        return load_corinth_canal(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, csv.Error, ValueError):
        return None


def load_corinth_canal(path: Path) -> CorinthCanalRun:
    """Load a corinth-canal artifact path (run directory, CSV, or JSON)."""

    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    else:
        resolved = resolved.resolve()

    if resolved.is_dir():
        return load_run_directory(resolved)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return _run_from_series(
            parse_latent_telemetry_csv(resolved),
            experiment_id=resolved.stem,
            raw_path=str(resolved),
        )
    if suffix == ".json":
        parent = resolved.parent
        if _dir_looks_like_run(parent) and resolved.name in {
            CANONICAL_SUMMARY,
            CANONICAL_MANIFEST,
        }:
            return load_run_directory(parent)
        return load_json_artifact(resolved)
    raise ValueError(f"unsupported corinth-canal artifact: {resolved}")


def load_run_directory(path: Path) -> CorinthCanalRun:
    """Load canonical filenames from a corinth-canal run directory."""

    skipped: list[str] = []
    csv_path = path / CANONICAL_LATENT_CSV
    summary_path = path / CANONICAL_SUMMARY
    manifest_path = path / CANONICAL_MANIFEST
    tick_path = path / CANONICAL_TICKS

    series = parse_latent_telemetry_csv(csv_path) if csv_path.exists() else None
    if not csv_path.exists():
        skipped.append(CANONICAL_LATENT_CSV)

    summary: dict[str, Any] | None = None
    if summary_path.exists():
        summary = _read_json_object(summary_path)
    else:
        skipped.append(CANONICAL_SUMMARY)

    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = _read_json_object(manifest_path)
    else:
        skipped.append(CANONICAL_MANIFEST)

    ticks = parse_tick_telemetry(tick_path) if tick_path.exists() else None
    if not tick_path.exists():
        skipped.append(CANONICAL_TICKS)

    return _merge_run(
        series=series,
        summary=summary,
        manifest=manifest,
        ticks=ticks,
        experiment_id_fallback=path.name,
        raw_path=str(path),
        skipped=tuple(skipped),
    )


def load_json_artifact(path: Path) -> CorinthCanalRun:
    raw = _read_json_object(path)
    kind = classify_json_artifact(raw)
    if kind == "summary":
        return _run_from_summary(raw, raw_path=str(path))
    if kind == "manifest":
        return _run_from_manifest(raw, raw_path=str(path))
    if kind == "legacy":
        return _run_from_legacy(raw, raw_path=str(path))
    unreachable: Never = kind
    raise TypeError(f"unhandled corinth-canal JSON kind: {unreachable}")


def classify_json_artifact(raw: dict[str, Any]) -> ArtifactKind:
    if "metrics" in raw and "latent_telemetry_path" in raw:
        return "summary"
    if "generated_files" in raw and "run_id" in raw and (
        "saaq_rule" in raw or "saaq_primary_rule" in raw
    ):
        return "manifest"
    return "legacy"


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


def _empty_csv_columns() -> _LatentCsvColumns:
    return _LatentCsvColumns(
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


def _append_if(values: list[Any], parsed: Any) -> None:
    if parsed is not None:
        values.append(parsed)


def _ingest_csv_row(columns: _LatentCsvColumns, row: dict[str, str | None]) -> None:
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
    _append_if(columns.delta_q, _parse_float(_cell(row, "saaq_delta_q_target")))
    _append_if(columns.delta_q_legacy, _parse_float(_cell(row, "saaq_delta_q_legacy_target")))
    _append_if(columns.delta_q_v15, _parse_float(_cell(row, "saaq_delta_q_v15_target")))


def parse_latent_telemetry_csv(path: Path) -> LatentTelemetrySeries:
    """Parse dual-SAAQ ``latent_telemetry.csv``. Missing columns are ignored."""
    columns = _empty_csv_columns()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            _ingest_csv_row(columns, row)
    return LatentTelemetrySeries(
        row_count=max(
            len(columns.firing_rates),
            len(columns.delta_q),
            len(columns.timestamps),
            len(columns.delta_q_legacy),
            len(columns.delta_q_v15),
        ),
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


def parse_summary(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


def parse_run_manifest(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


# ---------------------------------------------------------------------------
# Internal assembly
# ---------------------------------------------------------------------------

def _dir_looks_like_run(path: Path) -> bool:
    return (path / CANONICAL_LATENT_CSV).exists() or (path / CANONICAL_SUMMARY).exists()


def _run_from_series(
    series: LatentTelemetrySeries,
    *,
    experiment_id: str,
    raw_path: str,
    summary: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    skipped: tuple[str, ...] = (),
) -> CorinthCanalRun:
    return _merge_run(
        series=series,
        summary=summary,
        manifest=manifest,
        ticks=None,
        experiment_id_fallback=experiment_id,
        raw_path=raw_path,
        skipped=skipped,
    )


def _run_from_summary(raw: dict[str, Any], *, raw_path: str) -> CorinthCanalRun:
    return _merge_run(
        series=None,
        summary=raw,
        manifest=None,
        ticks=None,
        experiment_id_fallback=str(raw.get("run_id") or Path(raw_path).stem),
        raw_path=raw_path,
        skipped=(),
    )


def _run_from_manifest(raw: dict[str, Any], *, raw_path: str) -> CorinthCanalRun:
    return _merge_run(
        series=None,
        summary=None,
        manifest=raw,
        ticks=None,
        experiment_id_fallback=str(raw.get("run_id") or Path(raw_path).stem),
        raw_path=raw_path,
        skipped=(),
    )


def _run_from_legacy(raw: dict[str, Any], *, raw_path: str) -> CorinthCanalRun:
    trajectory = _float_tuple(raw.get("delta_q_trajectory") or raw.get("saaq_delta_q_trajectory"))
    legacy_traj = _float_tuple(raw.get("delta_q_legacy_trajectory"))
    v15_traj = _float_tuple(raw.get("delta_q_v15_trajectory"))
    firing_rate = _as_float(raw.get("firing_rate"))
    membrane = _as_float(raw.get("membrane_pressure"))
    return CorinthCanalRun(
        experiment_id=str(raw.get("experiment_id") or raw.get("run_id") or ""),
        artifact_version=str(raw.get("artifact_version") or SAAQ_ARTIFACT_VERSION),
        routing_entropy=_as_float(raw.get("routing_entropy")),
        spike_density=_as_float(raw.get("spike_density")),
        event_rate=_as_float(raw.get("event_rate")),
        latent_stability=_as_float(raw.get("latent_stability")),
        dv_dt_reductions=_as_float(raw.get("dv_dt_reductions")),
        firing_rate=firing_rate,
        membrane_pressure=membrane,
        membrane_dv_dt=_as_float(raw.get("membrane_dv_dt")),
        saaq_delta_q=_first_float(raw.get("saaq_delta_q"), raw.get("saaq_delta_q_mean")),
        saaq_delta_q_last=_as_float(raw.get("saaq_delta_q_last")),
        saaq_delta_q_legacy=_as_float(raw.get("saaq_delta_q_legacy")),
        saaq_delta_q_v15=_as_float(raw.get("saaq_delta_q_v15")),
        saaq_delta_q_trajectory=trajectory,
        saaq_delta_q_legacy_trajectory=legacy_traj,
        saaq_delta_q_v15_trajectory=v15_traj,
        saaq_rule=_as_str(raw.get("saaq_rule")),
        saaq_primary_rule=_as_str(raw.get("saaq_primary_rule")),
        model_family=_as_str(raw.get("model_family")),
        model_slug=_as_str(raw.get("model_slug")),
        ticks_completed=_as_int(raw.get("ticks_completed")),
        latent_rows=_as_int(raw.get("latent_rows")),
        raw_path=str(raw.get("raw_path") or raw_path),
        skipped=(),
    )


def _first_str(*values: Any) -> str | None:
    for value in values:
        parsed = _as_str(value)
        if parsed is not None:
            return parsed
    return None


def _overlay_from_series(series: LatentTelemetrySeries | None) -> LatentTelemetrySeries:
    if series is not None:
        return series
    return LatentTelemetrySeries(row_count=0)


def _run_identity(
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    fallback: str,
) -> tuple[str, str | None, str | None, str | None, str | None, bool | None]:
    summary = summary or {}
    manifest = manifest or {}
    experiment_id = _first_str(summary.get("run_id"), manifest.get("run_id")) or fallback
    saaq_rule = _first_str(summary.get("saaq_rule"), manifest.get("saaq_rule"))
    return (
        experiment_id,
        saaq_rule,
        _first_str(summary.get("model_family"), manifest.get("model_family")),
        _first_str(summary.get("model_slug"), manifest.get("model_slug")),
        _first_str(summary.get("projection_mode"), manifest.get("projection_mode")),
        _as_bool(manifest.get("saaq_dual_emit")),
    )


def _run_counts(
    summary: dict[str, Any] | None,
    series: LatentTelemetrySeries,
    ticks: TickTelemetrySeries | None,
) -> tuple[int | None, int | None]:
    metrics = _summary_metrics(summary)
    ticks_completed = _as_int(metrics.get("ticks_completed"))
    if ticks_completed is None and ticks is not None:
        ticks_completed = ticks.tick_count
    latent_rows = _as_int(metrics.get("latent_rows"))
    if latent_rows is None and series.row_count:
        latent_rows = series.row_count
    return ticks_completed, latent_rows


def _merge_run(
    *,
    series: LatentTelemetrySeries | None,
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    ticks: TickTelemetrySeries | None,
    experiment_id_fallback: str,
    raw_path: str,
    skipped: tuple[str, ...],
) -> CorinthCanalRun:
    overlay = _overlay_from_series(series)
    experiment_id, saaq_rule, family, slug, projection, dual_emit = _run_identity(
        summary, manifest, experiment_id_fallback
    )
    ticks_completed, latent_rows = _run_counts(summary, overlay, ticks)
    primary_rule = _as_str((manifest or {}).get("saaq_primary_rule")) or saaq_rule
    return CorinthCanalRun(
        experiment_id=experiment_id,
        artifact_version=SAAQ_ARTIFACT_VERSION,
        routing_entropy=overlay.routing_entropy_mean,
        spike_density=overlay.activity_pressure_mean,
        event_rate=overlay.firing_rate_mean,
        firing_rate=overlay.firing_rate_mean,
        membrane_pressure=overlay.membrane_pressure_mean,
        membrane_dv_dt=overlay.membrane_dv_dt_mean,
        saaq_delta_q=overlay.delta_q_mean,
        saaq_delta_q_last=overlay.delta_q_last,
        saaq_delta_q_legacy=overlay.delta_q_legacy_mean,
        saaq_delta_q_v15=overlay.delta_q_v15_mean,
        saaq_delta_q_trajectory=overlay.delta_q_trajectory,
        saaq_delta_q_legacy_trajectory=overlay.delta_q_legacy_trajectory,
        saaq_delta_q_v15_trajectory=overlay.delta_q_v15_trajectory,
        saaq_rule=saaq_rule,
        saaq_primary_rule=primary_rule,
        saaq_dual_emit=dual_emit,
        model_family=family,
        model_slug=slug,
        projection_mode=projection,
        ticks_completed=ticks_completed,
        latent_rows=latent_rows,
        raw_path=raw_path,
        summary=summary,
        run_manifest=manifest,
        skipped=skipped,
    )


def _summary_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    metrics = summary.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object in {path}")
    return raw


def _cell(row: dict[str, str | None], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


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
    if "tick" not in fields:
        return None
    return _parse_int(fields.get("best_walker")), _parse_float(fields.get("elapsed_us"))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _float_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[float] = []
    for item in value:
        parsed = _as_float(item)
        if parsed is not None:
            out.append(parsed)
    return tuple(out)
