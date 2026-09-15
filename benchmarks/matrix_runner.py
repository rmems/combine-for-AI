"""Resume-aware matrix runner: journal, validate, execute, aggregate.

This implements the resumability slice required by RM-105 / RM-1319. It does
not build the rest of the cross-model comparison stack. Each cell is executed
with the existing mock-capable adapters; only validated successful artifacts
are reused after a restart.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.journal import ResumeJournal
from benchmarks.matrix import MatrixDefinition, MatrixError, RetryPolicy
from benchmarks.matrix_session import SessionContext, execute_session
from benchmarks.matrix_types import (
    CellStatus,
    CrashPlan,
    MatrixHooks,
    MatrixRunResult,
    SimulatedInterrupt,
)

__all__ = [
    "CellStatus",
    "CrashPlan",
    "MatrixHooks",
    "MatrixRunResult",
    "SimulatedInterrupt",
    "load_matrix_config",
    "run_matrix",
]


def load_matrix_config(path: Path) -> MatrixDefinition:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise MatrixError("matrix config must be a JSON object")
    return MatrixDefinition.from_dict(raw)


def run_matrix(
    config_path: Path,
    output_dir: Path,
    *,
    retry: RetryPolicy | None = None,
    hooks: MatrixHooks | None = None,
) -> MatrixRunResult:
    definition = load_matrix_config(config_path)
    policy = retry if retry is not None else definition.retry
    hooks = hooks or MatrixHooks()
    matrix_dir = output_dir / definition.name
    journal_path = matrix_dir / "journal.jsonl"
    aggregate_path = matrix_dir / "aggregate.json"
    journal = ResumeJournal(journal_path)
    replay = journal.open()
    try:
        statuses, aggregate = execute_session(
            SessionContext(
                definition=definition,
                policy=policy,
                hooks=hooks,
                matrix_dir=matrix_dir,
                journal=journal,
                config_dir=config_path.parent,
                aggregate_path=aggregate_path,
            ),
            replay.records,
            replay.truncated_tail,
        )
    finally:
        journal.close()
    return MatrixRunResult(
        definition=definition,
        statuses=statuses,
        aggregate=aggregate,
        journal_path=journal_path,
        aggregate_path=aggregate_path,
        truncated_tail=replay.truncated_tail,
    )
