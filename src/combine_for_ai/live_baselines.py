"""SWE-bench-Live baseline hygiene: ingest, flags, and fair rankings."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


class LiveBaselineError(ValueError):
    """Invalid ingest input or baseline registry."""


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


@dataclass(frozen=True, slots=True)
class RankedRun:
    run_id: str
    resolved_count: int
    evaluated_count: int
    resolve_rate: float


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


def is_submission_results_payload(payload: Mapping[str, Any]) -> bool:
    """Detect SWE-bench-Live submission ``results.json`` (per-instance ID lists)."""
    return "success_ids" in payload or "submitted_ids" in payload


def is_leaderboard_aggregate_payload(payload: Mapping[str, Any]) -> bool:
    """Detect leaderboard-style per-language totals (not per-task rows)."""
    if "instance_id" in payload:
        return False
    if is_submission_results_payload(payload):
        return False
    if "rows" in payload and isinstance(payload["rows"], list):
        return False
    language_hits = _LEADERBOARD_LANGUAGE_KEYS.intersection(payload.keys())
    if language_hits:
        sample = payload[next(iter(language_hits))]
        if isinstance(sample, Mapping) and _AGGREGATE_ONLY_KEYS.intersection(sample.keys()):
            return True
    if "per_language" in payload and isinstance(payload["per_language"], Mapping):
        return True
    if _AGGREGATE_ONLY_KEYS.issuperset(payload.keys()) and "success_ids" not in payload:
        return True
    return False


def _coerce_task_row(raw: Mapping[str, Any], *, source: Path | str) -> TaskResultRow:
    if is_leaderboard_aggregate_payload(raw):
        raise LiveBaselineError(
            f"refusing leaderboard aggregate at {source}; ingest per-task result.json rows only"
        )
    if "instance_id" not in raw:
        raise LiveBaselineError(f"missing instance_id in per-task result at {source}")
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


def _instance_ids_from_submission_results(results: Mapping[str, Any]) -> list[str]:
    if is_leaderboard_aggregate_payload(results):
        raise LiveBaselineError(
            "refusing leaderboard per-language totals; ingest submission results.json"
        )
    if not is_submission_results_payload(results):
        raise LiveBaselineError("not a SWE-bench-Live submission results.json")
    if isinstance(results.get("submitted_ids"), list) and results["submitted_ids"]:
        ordered = [str(item) for item in results["submitted_ids"]]
    else:
        buckets = (
            "success_ids",
            "failure_ids",
            "error_ids",
            "incomplete_ids",
            "empty_patch_ids",
        )
        seen: set[str] = set()
        ordered: list[str] = []
        for bucket in buckets:
            ids = results.get(bucket)
            if not isinstance(ids, list):
                continue
            for item in ids:
                instance_id = str(item)
                if instance_id not in seen:
                    seen.add(instance_id)
                    ordered.append(instance_id)
    if not ordered:
        raise LiveBaselineError("submission results.json lists no instance ids")
    return ordered


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
    raw_success = results.get("success_ids")
    if isinstance(raw_success, list):
        success_ids = {str(item) for item in raw_success}
    else:
        success_ids = set()
    rows: list[TaskResultRow] = []
    for instance_id in _instance_ids_from_submission_results(results):
        pred_entry = preds.get(instance_id)
        rows.append(
            TaskResultRow(
                instance_id=instance_id,
                resolved=instance_id in success_ids,
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


def ingest_result_rows(payload: Mapping[str, Any], *, source: str = "<payload>") -> list[TaskResultRow]:
    """Ingest an explicit list of per-task rows (never leaderboard totals)."""
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
    if "rows" in payload and isinstance(payload["rows"], list):
        raw_rows = payload["rows"]
    elif payload and all(isinstance(value, Mapping) for value in payload.values()):
        raw_rows = list(payload.values())
    else:
        raw_rows = [payload]
    rows = [_coerce_task_row(item, source=source) for item in raw_rows]
    _assert_unique_instance_ids(rows, context=source)
    return rows


def shared_instance_ids(*runs: RunSnapshot) -> frozenset[str]:
    if not runs:
        return frozenset()
    sets = [frozenset(run.instance_ids) for run in runs]
    shared = sets[0]
    for other in sets[1:]:
        shared &= other
    return shared


def rank_runs_on_shared_rows(
    runs: Sequence[RunSnapshot],
    *,
    exclude_run_ids: Iterable[str] = (),
) -> list[RankedRun]:
    excluded = frozenset(exclude_run_ids)
    eligible = [run for run in runs if run.run_id not in excluded]
    if not eligible:
        return []
    common = shared_instance_ids(*eligible)
    if not common:
        raise LiveBaselineError("compared runs share no instance_id intersection")
    ranked: list[RankedRun] = []
    for run in eligible:
        resolved = sum(
            1
            for row in run.rows
            if row.instance_id in common and row.resolved is True
        )
        evaluated = len(common)
        ranked.append(
            RankedRun(
                run_id=run.run_id,
                resolved_count=resolved,
                evaluated_count=evaluated,
                resolve_rate=resolved / evaluated,
            )
        )
    ranked.sort(key=lambda item: (-item.resolve_rate, -item.resolved_count, item.run_id))
    return ranked


def filter_usable_runs(
    runs: Sequence[RunSnapshot],
    registry: BaselineRegistry,
    *,
    baseline_id_for_run: Mapping[str, str],
) -> list[RunSnapshot]:
    """Drop runs mapped to registry entries marked unusable_as_baseline."""
    blocked = flagged_baseline_ids(registry)
    kept: list[RunSnapshot] = []
    for run in runs:
        baseline_id = baseline_id_for_run.get(run.run_id)
        if baseline_id is not None and baseline_id in blocked:
            continue
        kept.append(run)
    return kept


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


_DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<path>.+?) b/(?P<path2>.+?)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _split_patch_files(patch: str) -> dict[str, str]:
    if not patch.strip():
        return {}
    lines = patch.splitlines()
    files: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        match = _DIFF_FILE_RE.match(line)
        if match:
            current = match.group("path")
            files[current] = [line]
            continue
        if current is not None:
            files[current].append(line)
    return {path: "\n".join(chunk) + "\n" for path, chunk in files.items()}


def _apply_unified_hunks(base: str, file_patch: str) -> str:
    if not file_patch.strip():
        return base
    base_lines = base.splitlines(keepends=True)
    if base == "":
        base_lines = []
    out: list[str] = []
    idx = 0
    in_hunk = False
    for line in file_patch.splitlines():
        if line.startswith(("diff --git", "index ", "---", "+++")):
            continue
        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match:
            in_hunk = True
            target_idx = int(hunk_match.group(1)) - 1
            if target_idx < idx:
                raise LiveBaselineError("patch hunks overlap or are out of order")
            out.extend(base_lines[idx:target_idx])
            idx = target_idx
            continue
        if not in_hunk:
            continue
        if line.startswith(" "):
            if idx >= len(base_lines):
                raise LiveBaselineError("patch context exceeds base file length")
            if base_lines[idx].rstrip("\r\n") != line[1:]:
                raise LiveBaselineError("patch context does not match base file")
            out.append(base_lines[idx])
            idx += 1
        elif line.startswith("-"):
            if idx >= len(base_lines):
                raise LiveBaselineError("patch deletion exceeds base file length")
            if base_lines[idx].rstrip("\r\n") != line[1:]:
                raise LiveBaselineError("patch deletion does not match base file")
            idx += 1
        elif line.startswith("+"):
            addition = line[1:]
            if not addition.endswith("\n"):
                addition += "\n"
            out.append(addition)
        elif line == "\\ No newline at end of file":
            continue
        else:
            raise LiveBaselineError(f"unsupported hunk line: {line!r}")
    out.extend(base_lines[idx:])
    return "".join(out)


def is_test_path(path: str, *, test_patch: str | None = None) -> bool:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    filename = lower.rsplit("/", 1)[-1]
    segments = lower.split("/")
    if (
        "tests" in segments
        or "__tests__" in segments
        or "/tests/" in lower
        or "/test/" in lower
        or lower.startswith("test/")
        or "/spec/" in lower
        or "src/test" in lower
    ):
        return True
    if (
        filename.startswith(("test_", "spec_"))
        or filename.endswith(
            (
                "_test.py",
                "_test.rs",
                "_test.go",
                "_test.java",
                "_spec.rb",
                ".spec.ts",
                ".spec.js",
                ".test.ts",
                ".test.js",
            )
        )
        or ".test." in filename
    ):
        return True
    if test_patch and path in _split_patch_files(test_patch):
        return True
    return False


def byte_identical_gold_src(
    *,
    base_files: Mapping[str, str],
    gold_patch: str,
    model_patch: str,
    test_patch: str | None = None,
) -> bool:
    """True when non-test files touched by gold match gold blob hashes after each patch."""
    gold_files = _split_patch_files(gold_patch)
    model_files = _split_patch_files(model_patch)
    compared = 0
    for path in gold_files:
        if is_test_path(path, test_patch=test_patch):
            continue
        compared += 1
        base = base_files.get(path, "")
        try:
            gold_bytes = _apply_unified_hunks(base, gold_files[path]).encode("utf-8")
            if path not in model_files:
                return False
            model_bytes = _apply_unified_hunks(base, model_files[path]).encode("utf-8")
        except LiveBaselineError:
            return False
        if git_blob_sha1(gold_bytes) != git_blob_sha1(model_bytes):
            return False
    return compared > 0


def contains_gold_test_patch(*, model_patch: str, test_patch: str) -> bool:
    if not test_patch.strip():
        return False
    normalized_model = _normalize_patch(model_patch)
    normalized_test = _normalize_patch(test_patch)
    if normalized_test in normalized_model:
        return True
    for path, chunk in _split_patch_files(test_patch).items():
        if _normalize_patch(chunk) in normalized_model:
            return True
        if path in _split_patch_files(model_patch) and _normalize_patch(
            _split_patch_files(model_patch)[path]
        ) == _normalize_patch(chunk):
            return True
    return False


def _normalize_patch(patch: str) -> str:
    lines = [line.rstrip() for line in patch.splitlines()]
    return "\n".join(line for line in lines if line)


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
    "RankedRun",
    "RunSnapshot",
    "TaskResultRow",
    "annotate_row_hygiene",
    "byte_identical_gold_src",
    "contains_gold_test_patch",
    "filter_usable_runs",
    "flagged_baseline_ids",
    "git_blob_sha1",
    "ingest_result_rows",
    "ingest_run_directory",
    "ingest_submission_directory",
    "is_leaderboard_aggregate_payload",
    "is_submission_results_payload",
    "rows_from_submission_bundle",
    "load_baseline_registry",
    "load_per_task_result_json",
    "rank_runs_on_shared_rows",
    "shared_instance_ids",
]
