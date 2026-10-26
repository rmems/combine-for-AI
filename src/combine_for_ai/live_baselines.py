"""SWE-bench-Live baseline hygiene: ingest, flags, and fair rankings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from combine_for_ai.live_errors import LiveBaselineError
from combine_for_ai.live_gold import (
    byte_identical_gold_src,
    contains_gold_test_patch,
    git_blob_sha1,
    is_test_path,
    load_base_files,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES_PATH = PACKAGE_ROOT / "data" / "swe_bench_live" / "baselines.json"

_LEADERBOARD_LANGUAGE_KEYS = frozenset({"lite", "java", "go", "rust", "python", "javascript", "typescript"})
_AGGREGATE_ONLY_KEYS = frozenset(
    {
        "resolved",
        "total",
        "success",
        "failure",
        "error",
        "success_rate",
        "pass_at_1",
        "per_language",
    }
)


class BaselineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    submission: str
    pr: int | None = None
    subset: str | None = None
    usable_as_baseline: bool
    reason: str


class BaselineRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    ingest_policy: dict[str, Any] = Field(default_factory=dict)
    entries: list[BaselineEntry]

    @model_validator(mode="after")
    def _unique_ids(self) -> BaselineRegistry:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.id in seen:
                msg = f"duplicate baseline entry id {entry.id!r}"
                raise ValueError(msg)
            seen.add(entry.id)
        return self


class TaskResultRow(BaseModel):
    """One evaluated SWE-bench-Live instance derived from a submission or harness row."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    resolved: bool | None = None
    model_patch: str | None = None
    byte_identical_gold_src: bool | None = None
    contains_gold_test_patch: bool | None = None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: str
    instance_ids: tuple[str, ...]
    rows: tuple[TaskResultRow, ...]


def load_baseline_registry(path: Path | None = None) -> BaselineRegistry:
    registry_path = path or DEFAULT_BASELINES_PATH
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        return BaselineRegistry.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise LiveBaselineError(f"failed to load baseline registry at {registry_path}") from exc


def flagged_baseline_ids(registry: BaselineRegistry) -> frozenset[str]:
    return frozenset(
        entry.id for entry in registry.entries if not entry.usable_as_baseline
    )


_SUBMISSION_ID_KEYS = (
    "success_ids",
    "submitted_ids",
    "resolved_ids",
    "unresolved_ids",
)


def is_submission_results_payload(payload: Mapping[str, Any]) -> bool:
    """Detect SWE-bench-Live submission ``results.json`` (per-instance ID lists)."""
    return any(key in payload for key in _SUBMISSION_ID_KEYS)


def _is_per_task_payload(payload: Mapping[str, Any]) -> bool:
    return (
        "instance_id" in payload
        or is_submission_results_payload(payload)
        or isinstance(payload.get("rows"), list)
    )


def _looks_like_language_totals(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("per_language"), Mapping):
        return True
    language_hits = _LEADERBOARD_LANGUAGE_KEYS.intersection(payload.keys())
    if not language_hits:
        return False
    sample = payload[next(iter(language_hits))]
    return isinstance(sample, Mapping) and bool(_AGGREGATE_ONLY_KEYS.intersection(sample.keys()))


def _is_count_only_payload(payload: Mapping[str, Any]) -> bool:
    return _AGGREGATE_ONLY_KEYS.issuperset(payload.keys()) and not any(
        key in payload for key in ("success_ids", "resolved_ids")
    )


def is_leaderboard_aggregate_payload(payload: Mapping[str, Any]) -> bool:
    """Detect leaderboard-style per-language totals (not per-task rows)."""
    if _is_per_task_payload(payload):
        return False
    return _looks_like_language_totals(payload) or _is_count_only_payload(payload)


def _coerce_task_row(raw: Mapping[str, Any], *, source: Path | str) -> TaskResultRow:
    if is_leaderboard_aggregate_payload(raw):
        raise LiveBaselineError(
            f"refusing leaderboard aggregate at {source}; ingest per-task result.json rows only"
        )
    if "instance_id" not in raw:
        raise LiveBaselineError(f"missing instance_id in per-task result at {source}")
    if not isinstance(raw.get("resolved"), bool):
        raise LiveBaselineError(
            f"missing boolean resolved in per-task result at {source}"
        )
    return TaskResultRow.model_validate(raw)


def load_per_task_result_json(path: Path) -> TaskResultRow:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveBaselineError(f"failed to read result.json at {path}") from exc
    if not isinstance(payload, Mapping):
        raise LiveBaselineError(f"expected JSON object in {path}")
    row = _coerce_task_row(payload, source=path)
    if not row.instance_id.strip():
        raise LiveBaselineError(f"empty instance_id in {path}")
    return row


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveBaselineError(f"failed to read JSON at {path}") from exc
    if not isinstance(payload, dict):
        raise LiveBaselineError(f"expected JSON object at {path}")
    return payload


_RESULT_ID_BUCKETS = (
    "submitted_ids",
    "success_ids",
    "resolved_ids",
    "failure_ids",
    "unresolved_ids",
    "error_ids",
    "incomplete_ids",
    "empty_patch_ids",
)


def _ids_from_buckets(results: Mapping[str, Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for bucket in _RESULT_ID_BUCKETS:
        ids = results.get(bucket)
        if not isinstance(ids, list):
            continue
        for item in ids:
            instance_id = str(item)
            if instance_id in seen:
                continue
            seen.add(instance_id)
            ordered.append(instance_id)
    return ordered


def _instance_ids_from_submission_results(results: Mapping[str, Any]) -> list[str]:
    if is_leaderboard_aggregate_payload(results):
        raise LiveBaselineError(
            "refusing leaderboard per-language totals; ingest submission results.json"
        )
    if not is_submission_results_payload(results):
        raise LiveBaselineError("not a SWE-bench-Live submission results.json")
    submitted = results.get("submitted_ids")
    if isinstance(submitted, list) and submitted:
        ordered = [str(item) for item in submitted]
    else:
        ordered = _ids_from_buckets(results)
    if not ordered:
        raise LiveBaselineError("submission results.json lists no instance ids")
    return ordered


def _id_list(results: Mapping[str, Any], key: str) -> list[str] | None:
    value = results.get(key)
    if key not in results:
        return None
    if not isinstance(value, list):
        raise LiveBaselineError(f"submission results.json field {key!r} must be a list")
    return [str(item) for item in value]


def _resolved_id_set(results: Mapping[str, Any], *, source: str) -> set[str]:
    """Require an explicit resolved-id list. Missing keys must not look like zero resolved."""
    resolved = _id_list(results, "resolved_ids")
    success = _id_list(results, "success_ids")
    if resolved is None and success is None:
        raise LiveBaselineError(
            f"unknown submission results format at {source}: "
            "need resolved_ids or success_ids (refusing silent zero-resolved ingest)"
        )
    ids: set[str] = set()
    if resolved is not None:
        ids.update(resolved)
    if success is not None:
        ids.update(success)
    return ids


def _model_patch_from_preds_entry(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        patch = entry.get("model_patch")
        if isinstance(patch, str):
            return patch
    return None


def rows_from_submission_bundle(
    results: Mapping[str, Any],
    preds: Mapping[str, Any],
    *,
    source: str,
) -> list[TaskResultRow]:
    """Expand submission ``results.json`` + ``preds.json`` into per-task rows."""
    resolved_ids = _resolved_id_set(results, source=source)
    rows: list[TaskResultRow] = []
    for instance_id in _instance_ids_from_submission_results(results):
        pred_entry = preds.get(instance_id)
        rows.append(
            TaskResultRow(
                instance_id=instance_id,
                resolved=instance_id in resolved_ids,
                model_patch=_model_patch_from_preds_entry(pred_entry),
            )
        )
    _assert_unique_instance_ids(rows, context=source)
    return rows


def ingest_submission_directory(run_dir: Path, *, run_id: str | None = None) -> RunSnapshot:
    if not run_dir.is_dir():
        raise LiveBaselineError(f"run directory not found: {run_dir}")
    resolved_run_id = run_id or run_dir.name
    results_path = run_dir / "results.json"
    preds_path = run_dir / "preds.json"
    if not results_path.is_file() or not preds_path.is_file():
        raise LiveBaselineError(
            f"submission folder {run_dir} must contain results.json and preds.json"
        )
    results = _load_json_object(results_path)
    preds = _load_json_object(preds_path)
    rows = rows_from_submission_bundle(results, preds, source=str(run_dir))
    instance_ids = tuple(row.instance_id for row in rows)
    return RunSnapshot(run_id=resolved_run_id, instance_ids=instance_ids, rows=tuple(rows))


def ingest_run_directory(run_dir: Path, *, run_id: str | None = None) -> RunSnapshot:
    if not run_dir.is_dir():
        raise LiveBaselineError(f"run directory not found: {run_dir}")
    resolved_run_id = run_id or run_dir.name
    if (run_dir / "results.json").is_file() and (run_dir / "preds.json").is_file():
        return ingest_submission_directory(run_dir, run_id=resolved_run_id)
    rows: list[TaskResultRow] = []
    for path in sorted(run_dir.rglob("result.json")):
        rows.append(load_per_task_result_json(path))
    if not rows:
        raise LiveBaselineError(
            f"no submission results.json+preds.json or per-task result.json under {run_dir}"
        )
    _assert_unique_instance_ids(rows, context=run_dir)
    instance_ids = tuple(row.instance_id for row in rows)
    return RunSnapshot(run_id=resolved_run_id, instance_ids=instance_ids, rows=tuple(rows))


def _reject_non_task_ingest(payload: Mapping[str, Any], source: str) -> None:
    if not payload:
        raise LiveBaselineError(f"empty per-task ingest payload at {source}")
    if is_submission_results_payload(payload):
        raise LiveBaselineError(
            f"refusing bare submission results.json at {source}; "
            "use ingest_submission_directory with preds.json"
        )
    if is_leaderboard_aggregate_payload(payload):
        raise LiveBaselineError(
            f"refusing leaderboard aggregate at {source}; ingest per-task result.json rows only"
        )


def _raw_rows_from_payload(payload: Mapping[str, Any]) -> list[Any]:
    if "rows" in payload and isinstance(payload["rows"], list):
        return payload["rows"]
    if all(isinstance(value, Mapping) for value in payload.values()):
        return list(payload.values())
    return [payload]


def ingest_result_rows(payload: Mapping[str, Any], *, source: str = "<payload>") -> list[TaskResultRow]:
    """Ingest an explicit list of per-task rows (never leaderboard totals)."""
    _reject_non_task_ingest(payload, source)
    rows = [_coerce_task_row(item, source=source) for item in _raw_rows_from_payload(payload)]
    _assert_unique_instance_ids(rows, context=source)
    return rows


def annotate_row_hygiene(
    row: TaskResultRow,
    *,
    gold_patch: str,
    test_patch: str | None,
    base_files: Mapping[str, str],
) -> TaskResultRow:
    model_patch = row.model_patch or ""
    data = row.model_dump()
    data["byte_identical_gold_src"] = byte_identical_gold_src(
        base_files=base_files,
        gold_patch=gold_patch,
        model_patch=model_patch,
        test_patch=test_patch,
    )
    data["contains_gold_test_patch"] = (
        contains_gold_test_patch(model_patch=model_patch, test_patch=test_patch or "")
        if test_patch
        else False
    )
    return TaskResultRow.model_validate(data)


def _assert_unique_instance_ids(rows: Sequence[TaskResultRow], *, context: Path | str) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.instance_id in seen:
            raise LiveBaselineError(
                f"duplicate instance_id {row.instance_id!r} while ingesting {context}"
            )
        seen.add(row.instance_id)


__all__ = [
    "BaselineEntry",
    "BaselineRegistry",
    "LiveBaselineError",
    "RunSnapshot",
    "TaskResultRow",
    "annotate_row_hygiene",
    "byte_identical_gold_src",
    "contains_gold_test_patch",
    "flagged_baseline_ids",
    "git_blob_sha1",
    "ingest_result_rows",
    "ingest_run_directory",
    "ingest_submission_directory",
    "is_leaderboard_aggregate_payload",
    "is_test_path",
    "is_submission_results_payload",
    "rows_from_submission_bundle",
    "load_base_files",
    "load_baseline_registry",
    "load_per_task_result_json",
]
