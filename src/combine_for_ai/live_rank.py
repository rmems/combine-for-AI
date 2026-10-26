"""Ranking modes: pinned rmems-eval-v0.1 vs explicit third-party intersection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from combine_for_ai.live_errors import LiveBaselineError
from combine_for_ai.live_baselines import (
    BaselineRegistry,
    RunSnapshot,
    flagged_baseline_ids,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PINNED_RUST_INSTANCE_IDS_PATH = (
    PACKAGE_ROOT / "data" / "swe_bench_live" / "multi_swe_bench_rust_239.json"
)
MULTI_SWE_BENCH_RUST_TASK_COUNT = 239


@dataclass(frozen=True, slots=True)
class RankedRun:
    run_id: str
    resolved_count: int
    evaluated_count: int
    resolve_rate: float
    missing_count: int


def _load_pin_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveBaselineError(f"failed to read JSON at {path}") from exc
    if not isinstance(payload, dict):
        raise LiveBaselineError(f"expected JSON object at {path}")
    return payload


def _raw_pinned_ids(payload: Mapping[str, Any], pin_path: Path) -> tuple[str, ...]:
    raw = payload.get("instance_ids")
    if not isinstance(raw, list) or not raw:
        raise LiveBaselineError(f"pinned instance id file {pin_path} has no instance_ids")
    return tuple(str(item) for item in raw)


def _require_unique_pin_count(ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(ids) != MULTI_SWE_BENCH_RUST_TASK_COUNT:
        raise LiveBaselineError(
            f"expected {MULTI_SWE_BENCH_RUST_TASK_COUNT} unique pinned ids, got {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise LiveBaselineError(
            f"expected {MULTI_SWE_BENCH_RUST_TASK_COUNT} unique pinned ids, got {len(ids)}"
        )
    return ids


def load_pinned_instance_ids(path: Path | None = None) -> tuple[str, ...]:
    """Load the pinned Multi-SWE-bench Rust task list (HF revision 56ff018c)."""
    pin_path = path or PINNED_RUST_INSTANCE_IDS_PATH
    return _require_unique_pin_count(_raw_pinned_ids(_load_pin_payload(pin_path), pin_path))


def shared_instance_ids(*runs: RunSnapshot) -> frozenset[str]:
    if not runs:
        return frozenset()
    shared = frozenset(runs[0].instance_ids)
    for other in runs[1:]:
        shared &= frozenset(other.instance_ids)
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


__all__ = [
    "MULTI_SWE_BENCH_RUST_TASK_COUNT",
    "PINNED_RUST_INSTANCE_IDS_PATH",
    "RankedRun",
    "filter_usable_runs",
    "load_pinned_instance_ids",
    "rank_on_pinned_instance_ids",
    "rank_runs_on_shared_rows",
    "shared_instance_ids",
]
