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

from benchmarks.corinth_canal_csv import (
    LATENT_CSV_HEADER,
    LatentTelemetrySeries,
    TickTelemetrySeries,
    activity_pressure,
    membrane_pressure,
    parse_latent_telemetry_csv,
    parse_tick_telemetry,
)
from benchmarks.corinth_canal_values import (
    _as_bool,
    _as_float,
    _as_int,
    _as_str,
    _first_float,
    _first_str,
    _float_tuple,
)

SAAQ_ARTIFACT_VERSION = "corinth-canal.saaq.v1"

__all__ = (
    "LATENT_CSV_HEADER",
    "LatentTelemetrySeries",
    "TickTelemetrySeries",
    "activity_pressure",
    "membrane_pressure",
    "parse_latent_telemetry_csv",
    "parse_tick_telemetry",
    "CorinthCanalRun",
    "try_load_corinth_canal",
    "load_corinth_canal",
    "load_run_directory",
    "load_json_artifact",
    "classify_json_artifact",
    "parse_summary",
    "parse_run_manifest",
    "SAAQ_ARTIFACT_VERSION",
)

CANONICAL_LATENT_CSV = "latent_telemetry.csv"
CANONICAL_SUMMARY = "summary.json"
CANONICAL_MANIFEST = "run_manifest.json"
CANONICAL_TICKS = "tick_telemetry.txt"

ArtifactKind = Literal["summary", "manifest", "legacy"]


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


_LOAD_ERRORS = (OSError, csv.Error, ValueError)


def try_load_corinth_canal(path: Path) -> CorinthCanalRun | None:
    """Load a run directory or file; return None when absent or unreadable."""

    if not path.exists():
        return None
    try:
        return load_corinth_canal(path)
    except _LOAD_ERRORS:
        return None


def load_corinth_canal(path: Path) -> CorinthCanalRun:
    """Load a corinth-canal artifact path (run directory, CSV, or JSON)."""

    resolved = path.expanduser().resolve()

    if resolved.is_dir():
        return load_run_directory(resolved)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return _run_from_series(
            parse_latent_telemetry_csv(resolved), resolved.stem, str(resolved)
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

    series = _optional_parse(csv_path, parse_latent_telemetry_csv, skipped)
    summary = _optional_json(summary_path, skipped)
    manifest = _optional_json(manifest_path, skipped)
    ticks = _optional_parse(tick_path, parse_tick_telemetry, skipped)

    return _merge_run(
        _MergeInputs(
            series=series,
            summary=summary,
            manifest=manifest,
            ticks=ticks,
            experiment_id_fallback=path.name,
            raw_path=str(path),
            skipped=tuple(skipped),
        )
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


def parse_summary(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


def parse_run_manifest(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


# ---------------------------------------------------------------------------
# Internal assembly
# ---------------------------------------------------------------------------

def _dir_looks_like_run(path: Path) -> bool:
    return (path / CANONICAL_LATENT_CSV).exists() or (path / CANONICAL_SUMMARY).exists()


@dataclass(frozen=True)
class _MergeInputs:
    series: LatentTelemetrySeries | None = None
    summary: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    ticks: TickTelemetrySeries | None = None
    experiment_id_fallback: str = ""
    raw_path: str = ""
    skipped: tuple[str, ...] = ()


def _optional_json(path: Path, skipped: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        skipped.append(path.name)
        return None
    try:
        return _read_json_object(path)
    except _LOAD_ERRORS:
        skipped.append(path.name)
        return None


def _optional_parse(path: Path, parser: Any, skipped: list[str]) -> Any:
    if not path.exists():
        skipped.append(path.name)
        return None
    try:
        return parser(path)
    except _LOAD_ERRORS:
        skipped.append(path.name)
        return None


def _run_from_series(
    series: LatentTelemetrySeries, experiment_id: str, raw_path: str
) -> CorinthCanalRun:
    return _merge_run(
        _MergeInputs(series=series, experiment_id_fallback=experiment_id, raw_path=raw_path)
    )


def _run_from_summary(raw: dict[str, Any], *, raw_path: str) -> CorinthCanalRun:
    return _merge_run(
        _MergeInputs(
            summary=raw,
            experiment_id_fallback=str(raw.get("run_id") or Path(raw_path).stem),
            raw_path=raw_path,
        )
    )


def _run_from_manifest(raw: dict[str, Any], *, raw_path: str) -> CorinthCanalRun:
    return _merge_run(
        _MergeInputs(
            manifest=raw,
            experiment_id_fallback=str(raw.get("run_id") or Path(raw_path).stem),
            raw_path=raw_path,
        )
    )


def _run_from_legacy(raw: dict[str, Any], *, raw_path: str) -> CorinthCanalRun:
    trajectory = _float_tuple(raw.get("delta_q_trajectory") or raw.get("saaq_delta_q_trajectory"))
    legacy_traj = _float_tuple(
        raw.get("delta_q_legacy_trajectory") or raw.get("saaq_delta_q_legacy_trajectory")
    )
    v15_traj = _float_tuple(
        raw.get("delta_q_v15_trajectory") or raw.get("saaq_delta_q_v15_trajectory")
    )
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
        model_family=_first_str(raw.get("model_family"), raw.get("saaq_model_family")),
        model_slug=_as_str(raw.get("model_slug")),
        ticks_completed=_as_int(raw.get("ticks_completed")),
        latent_rows=_as_int(raw.get("latent_rows")),
        raw_path=str(raw.get("raw_path") or raw_path),
        skipped=(),
    )


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
    saaq_rule = _first_str(
        summary.get("saaq_rule"),
        manifest.get("saaq_rule"),
        manifest.get("saaq_primary_rule"),
        summary.get("saaq_primary_rule"),
    )
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


def _merge_run(parts: _MergeInputs) -> CorinthCanalRun:
    overlay = _overlay_from_series(parts.series)
    experiment_id, saaq_rule, family, slug, projection, dual_emit = _run_identity(
        parts.summary, parts.manifest, parts.experiment_id_fallback
    )
    ticks_completed, latent_rows = _run_counts(parts.summary, overlay, parts.ticks)
    primary_rule = _as_str((parts.manifest or {}).get("saaq_primary_rule")) or saaq_rule
    return CorinthCanalRun(
        experiment_id=experiment_id,
        artifact_version=SAAQ_ARTIFACT_VERSION,
        routing_entropy=overlay.routing_entropy_mean,
        spike_density=None,
        event_rate=None,
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
        raw_path=parts.raw_path,
        summary=parts.summary,
        run_manifest=parts.manifest,
        skipped=parts.skipped,
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
