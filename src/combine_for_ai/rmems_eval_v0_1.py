"""rmems-eval-v0.1 scoring: always the 239-id Multi-SWE-bench Rust pin."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from combine_for_ai.live_baselines import BaselineRegistry, RunSnapshot, load_baseline_registry
from combine_for_ai.live_rank import (
    PINNED_RUST_INSTANCE_IDS_PATH,
    RankedRun,
    load_pinned_instance_ids,
    rank_on_pinned_instance_ids,
)

PIN_PATH = PINNED_RUST_INSTANCE_IDS_PATH


def rank_runs(
    runs: Sequence[RunSnapshot],
    *,
    exclude_run_ids: Iterable[str] = (),
    registry: BaselineRegistry | None = None,
    baseline_id_for_run: Mapping[str, str] | None = None,
) -> list[RankedRun]:
    """Rank runs on the pinned 239-id list. Never uses shared-id intersection."""
    pin = load_pinned_instance_ids(PIN_PATH)
    if registry is None:
        registry = load_baseline_registry()
    if baseline_id_for_run is None:
        # Default: a run is filtered only when its run_id is itself a blocked baseline id.
        baseline_id_for_run = {run.run_id: run.run_id for run in runs}
    return rank_on_pinned_instance_ids(
        runs,
        pin,
        exclude_run_ids=exclude_run_ids,
        registry=registry,
        baseline_id_for_run=baseline_id_for_run,
    )


__all__ = ["PIN_PATH", "rank_runs"]
