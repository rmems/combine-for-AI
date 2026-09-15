"""Resume-aware matrix runner: journal, validate, execute, aggregate.

This implements the resumability slice required by RM-105 / RM-1319. It does
not build the rest of the cross-model comparison stack. Each cell is executed
with the existing mock-capable adapters; only validated successful artifacts
are reused after a restart.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Never

from benchmarks.datasets import DatasetSpec, LoadedDataset, default_dataset_registry
from benchmarks.journal import (
    CellState,
    CommitPoint,
    JOURNAL_SCHEMA,
    JournalError,
    ResumeJournal,
)
from benchmarks.jsonio import ensure_dir, write_json
from benchmarks.matrix import (
    CellIdentity,
    IncompatibleMatrixError,
    MatrixDefinition,
    MatrixError,
    RetryPolicy,
    assert_compatible,
    sha256_hex,
)
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import (
    build_model_adapter,
    default_quantization_registry,
    scoped_seed,
)


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


def load_matrix_config(path: Path) -> MatrixDefinition:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise MatrixError("matrix config must be a JSON object")
    return MatrixDefinition.from_dict(raw)


def _journal_header(definition: MatrixDefinition, created_at: str) -> dict[str, Any]:
    return {
        "cell_ids": [cell.cell_id() for cell in definition.cells],
        "config_revision": definition.config_revision,
        "created_at": created_at,
        "definition_fingerprint": definition.fingerprint,
        "kind": "header",
        "matrix_name": definition.name,
        "schema": JOURNAL_SCHEMA,
    }


def _transition(
    identity: CellIdentity,
    state: CellState,
    attempt: int,
    ts: str,
    *,
    artifact_checksum: str | None = None,
    fingerprint: str | None = None,
    artifact_path: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "artifact_checksum": artifact_checksum,
        "artifact_path": artifact_path,
        "attempt": attempt,
        "cell_id": identity.cell_id(),
        "error": error,
        "fingerprint": fingerprint if fingerprint is not None else identity.fingerprint(),
        "identity": identity.to_canonical_dict(),
        "kind": "transition",
        "state": state.value,
        "ts": ts,
    }


def _optional_str(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else None


def _stamp_transition(status: CellStatus, state: CellState, ts: str | None) -> None:
    if state is CellState.RUNNING:
        status.started_at = ts
        status.finished_at = None
        return
    if state is CellState.PENDING:
        return
    status.finished_at = ts


def _copy_present(status: CellStatus, record: Mapping[str, Any], field: str) -> None:
    value = record.get(field)
    if value is not None:
        setattr(status, field, value)


def _apply_transition(status: CellStatus, record: Mapping[str, Any]) -> None:
    try:
        state = CellState(str(record["state"]))
        attempt = int(record.get("attempt") or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise JournalError("malformed journal transition") from exc
    status.state = state
    status.attempt = attempt
    status.error = record.get("error")
    _copy_present(status, record, "artifact_checksum")
    _copy_present(status, record, "fingerprint")
    _copy_present(status, record, "artifact_path")
    _stamp_transition(status, state, _optional_str(record, "ts"))


def _statuses_from_records(
    definition: MatrixDefinition,
    records: tuple[dict[str, Any], ...],
) -> dict[str, CellStatus]:
    statuses: dict[str, CellStatus] = {
        identity.cell_id(): CellStatus(identity=identity, state=CellState.PENDING, attempt=0)
        for identity in definition.cells
    }
    for record in records:
        kind = record.get("kind")
        if kind == "header":
            continue
        if kind != "transition":
            continue
        try:
            cell_id = str(record["cell_id"])
        except KeyError as exc:
            raise JournalError("malformed journal transition") from exc
        identity = definition.cell_by_id.get(cell_id)
        if identity is None:
            raise IncompatibleMatrixError(
                f"journal references unknown cell {cell_id} for the current matrix"
            )
        _apply_transition(statuses[cell_id], record)
    return statuses


def cell_artifact_path(output_dir: Path, cell_id: str) -> Path:
    return output_dir / "cells" / cell_id / "result.json"


def _file_checksum(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    write_json(tmp, dict(payload))
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def artifact_is_valid(status: CellStatus, output_dir: Path) -> bool:
    """A success may be skipped only when checksum and fingerprint both match."""

    if status.artifact_checksum is None or status.fingerprint is None:
        return False
    if status.fingerprint != status.identity.fingerprint():
        return False
    path = cell_artifact_path(output_dir, status.identity.cell_id())
    if not path.is_file():
        return False
    return _file_checksum(path) == status.artifact_checksum


def decide_action(status: CellStatus, policy: RetryPolicy, output_dir: Path) -> Action:
    state = status.state
    if state in {CellState.PENDING, CellState.RUNNING}:
        return Action.RUN
    if state is CellState.FAILED:
        return Action.RUN if policy.allows_retry(status.attempt) else Action.KEEP
    if state not in {CellState.SUCCEEDED, CellState.SKIPPED}:
        never: Never = state
        raise MatrixError(f"unhandled cell state: {never}")
    if artifact_is_valid(status, output_dir):
        return Action.SKIP if state is CellState.SUCCEEDED else Action.KEEP
    return Action.RUN


def resolve_dataset(spec: DatasetSpec, base_path: Path) -> DatasetSpec:
    if not spec.path:
        return spec
    path = Path(spec.path)
    if path.is_absolute():
        return spec
    return replace(spec, path=str((base_path / path).resolve()))


def execute_cell(
    identity: CellIdentity,
    dataset: LoadedDataset,
) -> dict[str, Any]:
    registry = default_quantization_registry()
    profile = registry.get(identity.quantization)
    adapter = build_model_adapter(identity.model, profile)
    scoped = scoped_seed(
        identity.seed,
        identity.model.name,
        identity.quantization,
        dataset.spec.name,
    )
    rng = random.Random(scoped)  # nosec B311: deterministic eval RNG, not crypto
    accumulator = MetricsAccumulator()
    for record in dataset.records:
        prediction = adapter.predict(record, rng)
        accumulator.add(record, prediction)
    total_time = accumulator.token_count / profile.speed_tps if profile.speed_tps else 0.0
    metrics = accumulator.summary(total_time, profile.vram_gb)
    return {
        "cell_id": identity.cell_id(),
        "fingerprint": identity.fingerprint(),
        "identity": identity.to_canonical_dict(),
        "metrics": asdict(metrics),
        "model": {
            "backend": identity.model.backend,
            "name": identity.model.name,
            "revision": identity.model.revision,
        },
        "quantization": identity.quantization,
        "dataset": dataset.spec.name,
        "sample_count": len(dataset.records),
        "seed": identity.seed,
        "split": dataset.spec.split,
    }


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


def _load_datasets(
    definition: MatrixDefinition,
    base_path: Path,
) -> dict[tuple[str, str, str | None], LoadedDataset]:
    registry = default_dataset_registry()
    loaded: dict[tuple[str, str, str | None], LoadedDataset] = {}
    for spec in definition.datasets:
        resolved = resolve_dataset(spec, base_path)
        key = (spec.name, spec.split, spec.path)
        if key not in loaded:
            loaded[key] = registry.loader_for(resolved.source).load(resolved)
    return loaded


def _dataset_for(
    identity: CellIdentity,
    loaded: dict[tuple[str, str, str | None], LoadedDataset],
) -> LoadedDataset:
    key = (identity.dataset.name, identity.dataset.split, identity.dataset.path)
    try:
        return loaded[key]
    except KeyError as exc:
        raise MatrixError(
            f"dataset {identity.dataset.name}/{identity.dataset.split} was not loaded"
        ) from exc


def _journaled_cell_ids(records: tuple[dict[str, Any], ...]) -> set[str]:
    return {
        str(record["cell_id"])
        for record in records
        if record.get("kind") == "transition" and "cell_id" in record
    }


def _aggregate_payload(
    definition: MatrixDefinition,
    statuses: Mapping[str, CellStatus],
    truncated_tail: bool,
) -> dict[str, Any]:
    cells = []
    counts = {state.value: 0 for state in CellState}
    for identity in definition.cells:
        status = statuses[identity.cell_id()]
        counts[status.state.value] += 1
        cells.append(
            {
                **identity.summary(),
                "attempt": status.attempt,
                "disposition": status.disposition,
                "error": status.error,
                "retry_count": status.retry_count,
                "state": status.state.value,
                "artifact_checksum": status.artifact_checksum,
                "finished_at": status.finished_at,
                "started_at": status.started_at,
            }
        )
    return {
        "cells": cells,
        "counts": counts,
        "definition_fingerprint": definition.fingerprint,
        "matrix_name": definition.name,
        "schema": "combine.matrix.aggregate.v1",
        "truncated_tail_recovered": truncated_tail,
    }


def _ensure_journal_header(
    journal: ResumeJournal,
    replay_records: tuple[dict[str, Any], ...],
    definition: MatrixDefinition,
    hooks: MatrixHooks,
) -> None:
    header = next((record for record in replay_records if record.get("kind") == "header"), None)
    if header is None:
        if replay_records:
            raise IncompatibleMatrixError("journal has transitions but no header")
        journal.append(_journal_header(definition, hooks.clock()))
        hooks.on_commit(CommitPoint.JOURNAL_WRITE, None)
        return
    if header.get("schema") != JOURNAL_SCHEMA:
        raise IncompatibleMatrixError(f"unsupported journal schema {header.get('schema')!r}")
    assert_compatible(str(header["definition_fingerprint"]), definition)


def _journal_pending_cells(
    definition: MatrixDefinition,
    statuses: dict[str, CellStatus],
    replay_records: tuple[dict[str, Any], ...],
    journal: ResumeJournal,
    hooks: MatrixHooks,
) -> None:
    already_journaled = _journaled_cell_ids(replay_records)
    for identity in definition.cells:
        cell_id = identity.cell_id()
        if cell_id in already_journaled:
            continue
        ts = hooks.clock()
        journal.append(_transition(identity, CellState.PENDING, 0, ts))
        hooks.on_commit(CommitPoint.JOURNAL_WRITE, cell_id, CellState.PENDING)
        statuses[cell_id].state = CellState.PENDING


def _record_skip(
    identity: CellIdentity,
    status: CellStatus,
    journal: ResumeJournal,
    hooks: MatrixHooks,
) -> None:
    ts = hooks.clock()
    journal.append(
        _transition(
            identity,
            CellState.SKIPPED,
            status.attempt,
            ts,
            artifact_checksum=status.artifact_checksum,
            fingerprint=status.fingerprint,
            artifact_path=status.artifact_path,
        )
    )
    hooks.on_commit(CommitPoint.JOURNAL_WRITE, identity.cell_id(), CellState.SKIPPED)
    _apply_transition(
        status,
        {
            "state": CellState.SKIPPED.value,
            "attempt": status.attempt,
            "artifact_checksum": status.artifact_checksum,
            "fingerprint": status.fingerprint,
            "artifact_path": status.artifact_path,
            "ts": ts,
        },
    )


def _fail_cell(
    identity: CellIdentity,
    status: CellStatus,
    attempt: int,
    error: str,
    journal: ResumeJournal,
    hooks: MatrixHooks,
) -> None:
    ts = hooks.clock()
    journal.append(
        _transition(
            identity,
            CellState.FAILED,
            attempt,
            ts,
            fingerprint=identity.fingerprint(),
            error=error,
        )
    )
    hooks.on_commit(CommitPoint.JOURNAL_WRITE, identity.cell_id(), CellState.FAILED)
    _apply_transition(
        status,
        {"state": CellState.FAILED.value, "attempt": attempt, "error": error, "ts": ts},
    )


def _succeed_cell(
    identity: CellIdentity,
    status: CellStatus,
    attempt: int,
    artifact: Path,
    journal: ResumeJournal,
    hooks: MatrixHooks,
) -> None:
    checksum = _file_checksum(artifact)
    ts = hooks.clock()
    journal.append(
        _transition(
            identity,
            CellState.SUCCEEDED,
            attempt,
            ts,
            artifact_checksum=checksum,
            fingerprint=identity.fingerprint(),
            artifact_path=str(artifact),
        )
    )
    hooks.on_commit(CommitPoint.JOURNAL_WRITE, identity.cell_id(), CellState.SUCCEEDED)
    _apply_transition(
        status,
        {
            "state": CellState.SUCCEEDED.value,
            "attempt": attempt,
            "artifact_checksum": checksum,
            "fingerprint": identity.fingerprint(),
            "artifact_path": str(artifact),
            "ts": ts,
        },
    )


def _run_cell(
    identity: CellIdentity,
    status: CellStatus,
    matrix_dir: Path,
    journal: ResumeJournal,
    hooks: MatrixHooks,
    loaded: dict[tuple[str, str, str | None], LoadedDataset],
) -> None:
    cell_id = identity.cell_id()
    attempt = status.attempt + 1
    ts = hooks.clock()
    journal.append(_transition(identity, CellState.RUNNING, attempt, ts))
    hooks.on_commit(CommitPoint.JOURNAL_WRITE, cell_id, CellState.RUNNING)
    status.state = CellState.RUNNING
    status.attempt = attempt
    status.started_at = ts
    status.finished_at = None
    status.error = None
    artifact = cell_artifact_path(matrix_dir, cell_id)
    try:
        if cell_id in hooks.fail_cell_ids or attempt == hooks.fail_on_attempt.get(cell_id):
            raise RuntimeError("injected cell failure")
        payload = execute_cell(identity, _dataset_for(identity, loaded))
        _atomic_write_json(artifact, payload)
        hooks.on_commit(CommitPoint.RESULT_COMMIT, cell_id)
        _succeed_cell(identity, status, attempt, artifact, journal, hooks)
    except SimulatedInterrupt:
        raise
    except Exception as exc:
        _fail_cell(identity, status, attempt, str(exc), journal, hooks)


def _process_cell(
    identity: CellIdentity,
    status: CellStatus,
    policy: RetryPolicy,
    matrix_dir: Path,
    journal: ResumeJournal,
    hooks: MatrixHooks,
    loaded: dict[tuple[str, str, str | None], LoadedDataset],
) -> None:
    action = decide_action(status, policy, matrix_dir)
    if action is Action.KEEP:
        return
    if action is Action.SKIP:
        _record_skip(identity, status, journal, hooks)
        return
    if action is Action.RUN:
        _run_cell(identity, status, matrix_dir, journal, hooks, loaded)
        return
    never: Never = action
    raise MatrixError(f"unhandled resume action: {never}")


def _execute_session(
    definition: MatrixDefinition,
    policy: RetryPolicy,
    hooks: MatrixHooks,
    matrix_dir: Path,
    journal: ResumeJournal,
    replay_records: tuple[dict[str, Any], ...],
    truncated_tail: bool,
    config_dir: Path,
    aggregate_path: Path,
) -> tuple[dict[str, CellStatus], dict[str, Any]]:
    _ensure_journal_header(journal, replay_records, definition, hooks)
    statuses = _statuses_from_records(definition, replay_records)
    loaded = _load_datasets(definition, config_dir)
    _journal_pending_cells(definition, statuses, replay_records, journal, hooks)
    for identity in definition.cells:
        _process_cell(
            identity, statuses[identity.cell_id()], policy, matrix_dir, journal, hooks, loaded
        )
    aggregate = _aggregate_payload(definition, statuses, truncated_tail)
    _atomic_write_json(aggregate_path, aggregate)
    hooks.on_commit(CommitPoint.AGGREGATE_UPDATE, None)
    return statuses, aggregate


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
        statuses, aggregate = _execute_session(
            definition,
            policy,
            hooks,
            matrix_dir,
            journal,
            replay.records,
            replay.truncated_tail,
            config_path.parent,
            aggregate_path,
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
