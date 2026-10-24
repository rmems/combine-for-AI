"""rmems-eval-v0.1 scoring: always the 239-id Multi-SWE-bench Rust pin."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from combine_for_ai.live_baselines import (
    PINNED_RUST_INSTANCE_IDS_PATH,
    RankedRun,
    RunSnapshot,
    load_pinned_instance_ids,
    rank_on_pinned_instance_ids,
)

PIN_PATH = PINNED_RUST_INSTANCE_IDS_PATH


def rank_runs(
    runs: Sequence[RunSnapshot],
    *,
    exclude_run_ids: Iterable[str] = (),
) -> list[RankedRun]:
    """Rank runs on the pinned 239-id list. Never uses shared-id intersection."""
    pin = load_pinned_instance_ids(PIN_PATH)
    return rank_on_pinned_instance_ids(runs, pin, exclude_run_ids=exclude_run_ids)


__all__ = ["PIN_PATH", "rank_runs"]
