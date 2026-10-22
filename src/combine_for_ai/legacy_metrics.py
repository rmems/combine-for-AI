"""Build unqualified rendering records from normalized GOZ1 rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from combine_for_ai.goz_import import SCHEMA_MULTIBLOCK_V1
from combine_for_ai.legacy_schema import GATE_ALIASES, GOZ_METRICS
from combine_for_ai.legacy_values import require_finite_metric
from combine_for_ai.report_models import CONTEXT_KEYS, EvidenceRecord, PreparedMetric


def legacy_record(row: dict[str, Any], raw: dict[str, Any], path: Path) -> EvidenceRecord:
    return EvidenceRecord(
        workload_track="compression-analysis",
        execution_kind="import",
        input_provenance=_input_provenance(raw),
        metric_provenance="unknown",
        series=_series(row),
        context=_legacy_context(row, raw, path),
        seed_policy=_seed_policy(row),
        block_index=row.get("block_index"),
        metrics=_legacy_metrics(row, raw),
    )


def _input_provenance(raw: dict[str, Any]) -> str:
    if _marked_fixture(raw):
        return "fixture"
    return "unknown"


def _marked_fixture(raw: dict[str, Any]) -> bool:
    provenance = raw.get("provenance") or {}
    certification = raw.get("certification") or {}
    if isinstance(provenance, dict) and provenance.get("agent") == "fixture":
        return True
    return isinstance(certification, dict) and certification.get("basis") == "fixture"


def _series(row: dict[str, Any]) -> str:
    arm = row.get("arm")
    if isinstance(arm, str) and arm.strip():
        return arm
    raise ValueError("legacy result requires a non-empty arm")


def _seed_policy(row: dict[str, Any]) -> str:
    # Tokens are not independent sample or seed counts.
    return f"single imported observation; seed={row.get('seed', 'unknown')}"


def _legacy_context(row: dict[str, Any], raw: dict[str, Any], path: Path) -> dict[str, Any]:
    context = {
        "model_family": row.get("model_family"),
        "tokens": row.get("tokens"),
        "seed": row.get("seed"),
        "schema": _schema_name(row, raw),
        "source_path": _context_source_path(row, path),
        "declared_context": row.get("context"),
    }
    for key in CONTEXT_KEYS:
        if key in row:
            context[key] = row[key]
    return context


def _schema_name(row: dict[str, Any], raw: dict[str, Any]) -> Any:
    run = raw.get("run")
    if not isinstance(run, dict):
        return row.get("schema")
    return row.get("schema", run.get("schema"))


def _context_source_path(row: dict[str, Any], path: Path) -> str:
    """Provenance label for cohort identity. This value is never opened."""
    supplied = row.get("source_path")
    if not isinstance(supplied, str) or not supplied.strip():
        supplied = str(path)
    return supplied.replace("\x00", "").replace("\r", " ").replace("\n", " ")


def _legacy_metrics(row: dict[str, Any], raw: dict[str, Any]) -> list[PreparedMetric]:
    prepared = {
        "schema": _schema_name(row, raw),
        "statuses": _status_lookup(raw),
        "execution_status": row.get("status"),
    }
    return [_one_metric(row, name, spec, prepared) for name, spec in GOZ_METRICS.items()]


def _status_lookup(raw: dict[str, Any]) -> dict[Any, Any]:
    summary = raw.get("summary") or []
    return {
        item.get("name"): item.get("status")
        for item in summary
        if isinstance(item, dict)
    }


def _one_metric(
    row: dict[str, Any],
    name: str,
    spec: tuple[str, str, str],
    prepared: dict[str, Any],
) -> PreparedMetric:
    unit, definition, scope = spec
    unit, definition = _metric_wording(name, unit, definition, prepared["schema"])
    status, value = _legacy_status(
        prepared["execution_status"], _optional_metric(row, name)
    )
    return PreparedMetric(
        name=name,
        unit=unit,
        definition=definition,
        scope=scope,
        value=value,
        status=status,
        notes=_metric_notes(name, prepared["statuses"], prepared["execution_status"]),
    )


def _metric_wording(
    name: str, unit: str, definition: str, schema: Any
) -> tuple[str, str]:
    if name == "expert_load_js" and schema != SCHEMA_MULTIBLOCK_V1:
        note = " (log base undeclared by legacy route schema)"
        return "unknown", definition + note
    return unit, definition


def _optional_metric(row: dict[str, Any], name: str) -> float | None:
    if name not in row or row[name] is None:
        return None
    return require_finite_metric(row[name], name)


def _legacy_status(execution_status: Any, value: float | None) -> tuple[Any, float | None]:
    if execution_status not in (None, "ok"):
        return execution_status, None
    if value is None:
        return "missing", None
    return "ok", value


def _metric_notes(name: str, statuses: dict[Any, Any], execution_status: Any) -> str:
    gate = statuses.get(GATE_ALIASES.get(name, name))
    if gate:
        notes = f"Upstream gate status: {gate}"
    else:
        notes = "Upstream gate status: unknown"
    if execution_status:
        return f"{notes}; upstream execution status: {execution_status}"
    return notes
