"""Shared types for the resume-aware matrix runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Never

from benchmarks.journal import CellState, CommitPoint
from benchmarks.matrix import CellIdentity, MatrixDefinition, MatrixError


class SimulatedInterrupt(Exception):
    """Raised by tests after a documented commit point, simulating a kill."""

    def __init__(self, point: CommitPoint, cell_id: str | None = None) -> None:
        self.point = point
        self.cell_id = cell_id
        super().__init__(f"simulated interrupt at {point} cell={cell_id}")


@dataclass(frozen=True)
class CrashPlan:
    point: CommitPoint
    cell_id: str | None = None
    state: CellState | None = None
    occurrence: int = 1


class Action(StrEnum):
    RUN = "run"
    SKIP = "skip"
    KEEP = "keep"


@dataclass
class CellStatus:
    identity: CellIdentity
    state: CellState
    attempt: int
    artifact_checksum: str | None = None
    fingerprint: str | None = None
    artifact_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def retry_count(self) -> int:
        return max(0, self.attempt - 1)

    @property
    def disposition(self) -> str:
        if self.state is CellState.SKIPPED:
            return CellState.SKIPPED.value
        if self.state is CellState.SUCCEEDED:
            return CellState.SUCCEEDED.value
        if self.state is CellState.FAILED:
            return CellState.FAILED.value
        if self.state is CellState.RUNNING:
            return "interrupted"
        if self.state is CellState.PENDING:
            return CellState.PENDING.value
        never: Never = self.state
        raise MatrixError(f"unhandled cell state: {never}")


@dataclass(frozen=True)
class MatrixRunResult:
    definition: MatrixDefinition
    statuses: dict[str, CellStatus]
    aggregate: dict[str, Any]
    journal_path: Path
    aggregate_path: Path
    truncated_tail: bool


def utc_now() -> str:
    stamp = datetime.now(timezone.utc).replace(microsecond=0)
    return stamp.isoformat().replace("+00:00", "Z")


class MatrixHooks:
    """Test seams for crash injection and forced cell failures."""

    def __init__(
        self,
        crash: CrashPlan | None = None,
        fail_cell_ids: frozenset[str] | None = None,
        fail_on_attempt: Mapping[str, int] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.crash = crash
        self.fail_cell_ids = fail_cell_ids or frozenset()
        self.fail_on_attempt = dict(fail_on_attempt or {})
        self.clock = clock or utc_now
        self._counts: dict[tuple[CommitPoint, str | None, CellState | None], int] = {}

    def _matches_crash(
        self,
        point: CommitPoint,
        cell_id: str | None,
        state: CellState | None,
    ) -> bool:
        crash = self.crash
        if crash is None or crash.point is not point:
            return False
        if crash.cell_id is not None and crash.cell_id != cell_id:
            return False
        return crash.state is None or crash.state is state

    def on_commit(
        self,
        point: CommitPoint,
        cell_id: str | None = None,
        state: CellState | None = None,
    ) -> None:
        crash = self.crash
        if crash is None or not self._matches_crash(point, cell_id, state):
            return
        key = (point, cell_id, crash.state)
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._counts[key] == crash.occurrence:
            raise SimulatedInterrupt(point, cell_id)
