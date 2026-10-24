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
PINNED_RUST_INSTANCE_IDS_PATH = (
    PACKAGE_ROOT / "data" / "swe_bench_live" / "multi_swe_bench_rust_239.json"
)
MULTI_SWE_BENCH_RUST_TASK_COUNT = 239

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
    missing_count: int


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


def _looks_like_language_totals(payload: Mapping[str, Any]) -> bool:
    language_hits = _LEADERBOARD_LANGUAGE_KEYS.intersection(payload.keys())
    if not language_hits:
        return False
    sample = payload[next(iter(language_hits))]
    return isinstance(sample, Mapping) and bool(_AGGREGATE_ONLY_KEYS.intersection(sample.keys()))


def is_leaderboard_aggregate_payload(payload: Mapping[str, Any]) -> bool:
    """Detect leaderboard-style per-language totals (not per-task rows)."""
    if "instance_id" in payload or is_submission_results_payload(payload):
        return False
    if "rows" in payload and isinstance(payload["rows"], list):
        return False
    if _looks_like_language_totals(payload):
        return True
    if "per_language" in payload and isinstance(payload["per_language"], Mapping):
        return True
    return _AGGREGATE_ONLY_KEYS.issuperset(payload.keys()) and not any(
        key in payload for key in ("success_ids", "resolved_ids")
    )


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


def load_pinned_instance_ids(path: Path | None = None) -> tuple[str, ...]:
    """Load the pinned Multi-SWE-bench Rust task list (HF revision 56ff018c)."""
    pin_path = path or PINNED_RUST_INSTANCE_IDS_PATH
    try:
        payload = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveBaselineError(f"failed to load pinned instance ids at {pin_path}") from exc
    raw = payload.get("instance_ids") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise LiveBaselineError(f"pinned instance id file {pin_path} has no instance_ids")
    ids = tuple(str(item) for item in raw)
    if len(ids) != MULTI_SWE_BENCH_RUST_TASK_COUNT or len(set(ids)) != len(ids):
        raise LiveBaselineError(
            f"expected {MULTI_SWE_BENCH_RUST_TASK_COUNT} unique pinned ids, got {len(ids)}"
        )
    return ids


def shared_instance_ids(*runs: RunSnapshot) -> frozenset[str]:
    if not runs:
        return frozenset()
    sets = [frozenset(run.instance_ids) for run in runs]
    shared = sets[0]
    for other in sets[1:]:
        shared &= other
    return shared


def _eligible_runs(
    runs: Sequence[RunSnapshot], exclude_run_ids: Iterable[str]
) -> list[RunSnapshot]:
    excluded = frozenset(exclude_run_ids)
    return [run for run in runs if run.run_id not in excluded]


def _score_run_on_ids(run: RunSnapshot, universe: Sequence[str]) -> RankedRun:
    by_id = {row.instance_id: row for row in run.rows}
    resolved = 0
    missing = 0
    for instance_id in universe:
        row = by_id.get(instance_id)
        if row is None:
            missing += 1
            continue
        if row.resolved is None:
            raise LiveBaselineError(
                f"missing boolean resolved for {instance_id!r} in run {run.run_id!r}"
            )
        if row.resolved is True:
            resolved += 1
    evaluated = len(universe)
    return RankedRun(
        run_id=run.run_id,
        resolved_count=resolved,
        evaluated_count=evaluated,
        resolve_rate=resolved / evaluated,
        missing_count=missing,
    )


def _out_of_pin_ids(runs: Sequence[RunSnapshot], pin: Sequence[str]) -> list[str]:
    allowed = frozenset(pin)
    extra: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for instance_id in run.instance_ids:
            if instance_id in allowed or instance_id in seen:
                continue
            seen.add(instance_id)
            extra.append(instance_id)
    return extra


def _sort_ranked(ranked: list[RankedRun]) -> list[RankedRun]:
    ranked.sort(key=lambda item: (-item.resolve_rate, -item.resolved_count, item.run_id))
    return ranked


def rank_on_pinned_instance_ids(
    runs: Sequence[RunSnapshot],
    pin: Sequence[str],
    *,
    exclude_run_ids: Iterable[str] = (),
) -> list[RankedRun]:
    """Score every run against ``pin``. Ids outside the pin raise; missing ids are unresolved."""
    if not pin:
        raise LiveBaselineError("pinned instance id list is empty")
    eligible = _eligible_runs(runs, exclude_run_ids)
    if not eligible:
        return []
    extra = _out_of_pin_ids(eligible, pin)
    if extra:
        raise LiveBaselineError(
            "instance ids not in rmems-eval-v0.1 pin: " + ", ".join(extra)
        )
    return _sort_ranked([_score_run_on_ids(run, pin) for run in eligible])


def rank_runs_on_shared_rows(
    runs: Sequence[RunSnapshot],
    *,
    exclude_run_ids: Iterable[str] = (),
) -> list[RankedRun]:
    """Explicit third-party leaderboard mode: rank on the intersection of reported ids."""
    eligible = _eligible_runs(runs, exclude_run_ids)
    if not eligible:
        return []
    common = shared_instance_ids(*eligible)
    if not common:
        raise LiveBaselineError("compared runs share no instance_id intersection")
    universe = tuple(sorted(common))
    return _sort_ranked([_score_run_on_ids(run, universe) for run in eligible])


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
    # Git blob IDs are SHA-1 by definition (not a security MAC).
    digest = hashlib.sha1(header + content, usedforsecurity=False)  # nosec B324
    return digest.hexdigest()


_DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<path>.+?) b/(?P<path2>.+?)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


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


def _hunk_target_index(old_start: int, old_count: int | None) -> int:
    """Map unified-diff old range to a 0-based index into ``base_lines``.

    ``@@ -0,0`` is a new file. ``@@ -N,0`` is an insertion after line N.
    """
    if old_start == 0:
        return 0
    count = 1 if old_count is None else old_count
    if count == 0:
        return old_start
    return old_start - 1


def _seek_hunk(
    out: list[str],
    base_lines: list[str],
    idx: int,
    old_start: int,
    old_count: int | None,
) -> int:
    target_idx = _hunk_target_index(old_start, old_count)
    if target_idx < idx:
        raise LiveBaselineError("patch hunks overlap or are out of order")
    out.extend(base_lines[idx:target_idx])
    return target_idx


def _require_base_line(base_lines: list[str], idx: int, expected: str, kind: str) -> None:
    if idx >= len(base_lines):
        raise LiveBaselineError(f"patch {kind} exceeds base file length")
    if base_lines[idx].rstrip("\r\n") != expected:
        raise LiveBaselineError(f"patch {kind} does not match base file")


def _apply_hunk_body_line(
    line: str, *, base_lines: list[str], out: list[str], idx: int
) -> int:
    if line.startswith(" "):
        _require_base_line(base_lines, idx, line[1:], "context")
        out.append(base_lines[idx])
        return idx + 1
    if line.startswith("-"):
        _require_base_line(base_lines, idx, line[1:], "deletion")
        return idx + 1
    if line.startswith("+"):
        addition = line[1:]
        if not addition.endswith("\n"):
            addition += "\n"
        out.append(addition)
        return idx
    if line == "\\ No newline at end of file":
        return idx
    raise LiveBaselineError(f"unsupported hunk line: {line!r}")


def _start_hunk(
    line: str, out: list[str], base_lines: list[str], idx: int
) -> tuple[bool, int]:
    hunk_match = _HUNK_HEADER_RE.match(line)
    if not hunk_match:
        return False, idx
    old_count = int(hunk_match.group(2)) if hunk_match.group(2) is not None else None
    return True, _seek_hunk(out, base_lines, idx, int(hunk_match.group(1)), old_count)


def _apply_unified_hunks(base: str, file_patch: str) -> str:
    if not file_patch.strip():
        return base
    base_lines = [] if base == "" else base.splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    in_hunk = False
    for line in file_patch.splitlines():
        if line.startswith("diff --git"):
            in_hunk = False
            continue
        if not in_hunk and line.startswith(("index ", "---", "+++")):
            continue
        started, idx = _start_hunk(line, out, base_lines, idx)
        if started:
            in_hunk = True
            continue
        if in_hunk:
            idx = _apply_hunk_body_line(line, base_lines=base_lines, out=out, idx=idx)
    out.extend(base_lines[idx:])
    return "".join(out)


_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__", "spec"})
_TEST_FILE_PREFIXES = ("test_", "spec_")
_TEST_FILE_SUFFIXES = (
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


def _is_test_directory(path_lower: str) -> bool:
    segments = path_lower.split("/")
    return any(segment in _TEST_DIR_NAMES for segment in segments[:-1]) or path_lower.startswith(
        "test/"
    )


def _is_test_filename(filename: str) -> bool:
    return (
        filename.startswith(_TEST_FILE_PREFIXES)
        or filename.endswith(_TEST_FILE_SUFFIXES)
        or ".test." in filename
    )


def is_test_path(path: str, *, test_patch: str | None = None) -> bool:
    lower = path.replace("\\", "/").lower()
    filename = lower.rsplit("/", 1)[-1]
    if _is_test_directory(lower) or _is_test_filename(filename):
        return True
    return bool(test_patch and path in _split_patch_files(test_patch))


def load_base_files(root: Path, paths: Iterable[str]) -> dict[str, str]:
    """Load workspace snapshots for gold-blob comparison. Missing paths stay absent."""
    loaded: dict[str, str] = {}
    for path in paths:
        candidate = root / path
        if candidate.is_file():
            loaded[path] = candidate.read_text(encoding="utf-8")
    return loaded


def _is_new_file_patch(file_patch: str) -> bool:
    if "--- /dev/null" in file_patch:
        return True
    match = _HUNK_HEADER_RE.search(file_patch)
    return bool(match and int(match.group(1)) == 0)


def _base_text_for_gold_path(path: str, file_patch: str, base_files: Mapping[str, str]) -> str | None:
    if path in base_files:
        return base_files[path]
    if _is_new_file_patch(file_patch):
        return ""
    return None


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
        base = _base_text_for_gold_path(path, gold_files[path], base_files)
        if base is None:
            return False
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
    "load_base_files",
    "load_baseline_registry",
    "load_per_task_result_json",
    "load_pinned_instance_ids",
    "rank_on_pinned_instance_ids",
    "rank_runs_on_shared_rows",
    "shared_instance_ids",
]
