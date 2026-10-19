"""Rendering inputs, not a qualification engine or an evaluator (#45/#49).

Prepared summaries retain upstream definitions and intervals. Legacy GOZ1
observations use the existing importer and are explicitly unqualified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from combine_for_ai.legacy_evidence import legacy_records
from combine_for_ai.report_models import (
    CONTEXT_KEYS,
    EvidenceArchive,
    EvidenceRecord,
    RenderingModel,
)

__all__ = [
    "CONTEXT_KEYS",
    "EvidenceArchive",
    "EvidenceRecord",
    "RenderingModel",
    "load_report_evidence",
    "read_json",
]


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    """Hash the exact bytes parsed, so source identity cannot race a second read."""
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: input root must be a JSON object")
    return raw, hashlib.sha256(data).hexdigest()


def load_report_evidence(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one archived input is required")
    sources: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        source, batch = _load_archive(Path(path), seen)
        sources.append(source)
        cohorts.extend(batch)
    return sources, cohorts


def _load_archive(
    path: Path, seen: set[Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"duplicate archived input: {path}")
    seen.add(resolved)
    raw, digest = read_json(path)
    legacy = _is_legacy(raw)
    records = _parse_records(raw, path, legacy)
    source = {
        "path": str(path),
        "sha256": digest,
        "run": raw.get("run", {}),
        "provenance": raw.get("provenance", {}),
        "legacy": legacy,
    }
    return source, _cohorts(records, str(path), legacy=legacy)


def _is_legacy(raw: dict[str, Any]) -> bool:
    return "records" not in raw and raw.get("schema_version") != "combine.plot-evidence.v1"


def _parse_records(raw: dict[str, Any], path: Path, legacy: bool) -> list[EvidenceRecord]:
    try:
        if legacy:
            return legacy_records(raw, path)
        return list(EvidenceArchive.model_validate(raw).records)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid archive {path}: {exc}") from exc


def _known(value: Any) -> bool:
    if _blank(value):
        return False
    if isinstance(value, dict):
        return _all_known(value.values())
    if isinstance(value, list):
        return _all_known(value)
    return True


def _blank(value: Any) -> bool:
    return value is None or value == "unknown" or value == ""


def _all_known(values: Any) -> bool:
    items = list(values)
    return bool(items) and all(_known(item) for item in items)


def _cohorts(
    records: list[EvidenceRecord], source: str, *, legacy: bool
) -> list[dict[str, Any]]:
    cohorts: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        _place_record(cohorts, record, index, source, legacy)
    return list(cohorts.values())


def _place_record(
    cohorts: dict[str, dict[str, Any]],
    record: EvidenceRecord,
    index: int,
    source: str,
    legacy: bool,
) -> None:
    fields = record.model_dump(
        exclude={"series", "metrics", "block_index", "source_artifacts"}
    )
    complete = _context_complete(record)
    key = json.dumps(fields, sort_keys=True, allow_nan=False)
    if not legacy and not complete:
        key = f"{key}:{index}"
    policy = _comparison_policy(legacy, complete)
    cohort = _cohort_for(cohorts, fields, source, key, policy)
    for metric in record.metrics:
        _add_metric(cohort, record, metric, source)


def _context_complete(record: EvidenceRecord) -> bool:
    if not _known(record.context):
        return False
    if any(not _known(record.context.get(key)) for key in CONTEXT_KEYS):
        return False
    if record.sample_count is None or record.repeat_count is None:
        return False
    return record.seed_policy != "unknown"


def _comparison_policy(legacy: bool, complete: bool) -> str:
    if legacy:
        return "legacy observations: unqualified"
    if complete:
        return "compatible prepared summaries"
    return "isolated: incomplete context"


def _cohort_for(
    cohorts: dict[str, dict[str, Any]],
    fields: dict[str, Any],
    source: str,
    key: str,
    policy: str,
) -> dict[str, Any]:
    cohort = cohorts.get(key)
    if cohort is None:
        cohort = _new_cohort(fields, source, key, policy)
        cohorts[key] = cohort
    return cohort


def _new_cohort(
    fields: dict[str, Any], source: str, key: str, policy: str
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{source}:{key}".encode()).hexdigest()
    return {**fields, "id": digest[:16], "source": source, "comparison_policy": policy, "metrics": []}


def _add_metric(cohort: dict[str, Any], record: EvidenceRecord, metric: Any, source: str) -> None:
    bucket = _metric_bucket(cohort, metric)
    if _duplicate(bucket["observations"], record):
        raise ValueError(
            f"{source}: duplicate observation for {record.series}/{metric.name}"
            f"/block={record.block_index}; prepare repeats upstream"
        )
    bucket["observations"].append(_observation(record, metric))


def _metric_bucket(cohort: dict[str, Any], metric: Any) -> dict[str, Any]:
    definition = metric.model_dump(include={"name", "unit", "definition", "scope"})
    for existing in cohort["metrics"]:
        if _same_definition(existing, definition):
            return existing
    created = {**definition, "observations": []}
    cohort["metrics"].append(created)
    return created


def _same_definition(existing: dict[str, Any], definition: dict[str, Any]) -> bool:
    return all(existing[key] == value for key, value in definition.items())


def _duplicate(observations: list[dict[str, Any]], record: EvidenceRecord) -> bool:
    return any(
        obs["series"] == record.series and obs["block_index"] == record.block_index
        for obs in observations
    )


def _observation(record: EvidenceRecord, metric: Any) -> dict[str, Any]:
    observation = metric.model_dump(exclude={"name", "unit", "definition", "scope"})
    observation.update(
        series=record.series,
        block_index=record.block_index,
        source_artifacts=record.source_artifacts,
    )
    return observation
