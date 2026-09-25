"""Journal replay, cell execution, and aggregate writes for a matrix session."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Never

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
    inputs_checksum,
    sha256_hex,
)
from benchmarks.matrix_types import (
    Action,
    CellStatus,
    MatrixHooks,
    SimulatedInterrupt,
)
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import (
    build_model_adapter,
    default_quantization_registry,
    scoped_seed,
)


@dataclass(frozen=True)
class SessionContext:
    definition: MatrixDefinition
    policy: RetryPolicy
    hooks: MatrixHooks
    matrix_dir: Path
    journal: ResumeJournal
    config_dir: Path
    aggregate_path: Path


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


@dataclass(frozen=True)
class TransitionFields:
    artifact_checksum: str | None = None
    artifact_path: str | None = None
    input_digest: str | None = None
    fingerprint: str | None = None
    error: str | None = None


def _transition(
    identity: CellIdentity,
    state: CellState,
    attempt: int,
    ts: str,
    fields: TransitionFields | None = None,
) -> dict[str, Any]:
    extra = fields or TransitionFields()
    fingerprint = extra.fingerprint
    if fingerprint is None:
        fingerprint = identity.fingerprint()
    return {
        "artifact_checksum": extra.artifact_checksum,
        "artifact_path": extra.artifact_path,
        "inputs_checksum": extra.input_digest,
        "attempt": attempt,
        "cell_id": identity.cell_id(),
        "error": extra.error,
        "fingerprint": fingerprint,
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
    _copy_present(status, record, "inputs_checksum")
    if state is CellState.FAILED:
        status.artifact_checksum = None
        status.artifact_path = None
        status.inputs_checksum = None
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
    if status.inputs_checksum != inputs_checksum(status.identity.dataset):
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


def _dataset_for(
    identity: CellIdentity,
    loaded: dict[DatasetSpec, LoadedDataset],
    base_path: Path,
) -> LoadedDataset:
    spec = identity.dataset
    cached = loaded.get(spec)
    if cached is not None:
        return cached
    try:
        resolved = resolve_dataset(spec, base_path)
        loaded[spec] = default_dataset_registry().loader_for(resolved.source).load(resolved)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise MatrixError(
            f"dataset {identity.dataset.name}/{identity.dataset.split} was not loaded"
        ) from exc
    return loaded[spec]


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
    ctx: SessionContext,
    replay_records: tuple[dict[str, Any], ...],
) -> None:
    header = next((record for record in replay_records if record.get("kind") == "header"), None)
    if header is None:
        if replay_records:
            raise IncompatibleMatrixError("journal has transitions but no header")
        ctx.journal.append(_journal_header(ctx.definition, ctx.hooks.clock()))
        ctx.hooks.on_commit(CommitPoint.JOURNAL_WRITE, None)
        return
    if header.get("schema") != JOURNAL_SCHEMA:
        raise IncompatibleMatrixError(f"unsupported journal schema {header.get('schema')!r}")
    assert_compatible(str(header["definition_fingerprint"]), ctx.definition)


def _journal_pending_cells(
    ctx: SessionContext,
    statuses: dict[str, CellStatus],
    replay_records: tuple[dict[str, Any], ...],
) -> None:
    already_journaled = _journaled_cell_ids(replay_records)
    for identity in ctx.definition.cells:
        cell_id = identity.cell_id()
        if cell_id in already_journaled:
            continue
        ts = ctx.hooks.clock()
        ctx.journal.append(_transition(identity, CellState.PENDING, 0, ts))
        ctx.hooks.on_commit(CommitPoint.JOURNAL_WRITE, cell_id, CellState.PENDING)
        statuses[cell_id].state = CellState.PENDING


def _record_skip(ctx: SessionContext, identity: CellIdentity, status: CellStatus) -> None:
    ts = ctx.hooks.clock()
    ctx.journal.append(
        _transition(
            identity,
            CellState.SKIPPED,
            status.attempt,
            ts,
            TransitionFields(
                artifact_checksum=status.artifact_checksum,
                fingerprint=status.fingerprint,
                artifact_path=status.artifact_path,
                input_digest=status.inputs_checksum,
            ),
        )
    )
    ctx.hooks.on_commit(CommitPoint.JOURNAL_WRITE, identity.cell_id(), CellState.SKIPPED)
    _apply_transition(
        status,
        {
            "state": CellState.SKIPPED.value,
            "attempt": status.attempt,
            "artifact_checksum": status.artifact_checksum,
            "fingerprint": status.fingerprint,
            "artifact_path": status.artifact_path,
            "inputs_checksum": status.inputs_checksum,
            "ts": ts,
        },
    )


def _fail_cell(
    ctx: SessionContext,
    identity: CellIdentity,
    status: CellStatus,
    attempt: int,
    error: str,
) -> None:
    ts = ctx.hooks.clock()
    ctx.journal.append(
        _transition(
            identity,
            CellState.FAILED,
            attempt,
            ts,
            TransitionFields(fingerprint=identity.fingerprint(), error=error),
        )
    )
    ctx.hooks.on_commit(CommitPoint.JOURNAL_WRITE, identity.cell_id(), CellState.FAILED)
    _apply_transition(
        status,
        {"state": CellState.FAILED.value, "attempt": attempt, "error": error, "ts": ts},
    )


def _succeed_cell(
    ctx: SessionContext,
    identity: CellIdentity,
    status: CellStatus,
    attempt: int,
    artifact: Path,
) -> None:
    checksum = _file_checksum(artifact)
    digest = inputs_checksum(identity.dataset)
    ts = ctx.hooks.clock()
    ctx.journal.append(
        _transition(
            identity,
            CellState.SUCCEEDED,
            attempt,
            ts,
            TransitionFields(
                artifact_checksum=checksum,
                fingerprint=identity.fingerprint(),
                artifact_path=str(artifact),
                input_digest=digest,
            ),
        )
    )
    ctx.hooks.on_commit(CommitPoint.JOURNAL_WRITE, identity.cell_id(), CellState.SUCCEEDED)
    _apply_transition(
        status,
        {
            "state": CellState.SUCCEEDED.value,
            "attempt": attempt,
            "artifact_checksum": checksum,
            "fingerprint": identity.fingerprint(),
            "artifact_path": str(artifact),
            "inputs_checksum": digest,
            "ts": ts,
        },
    )


def _run_cell(
    ctx: SessionContext,
    identity: CellIdentity,
    status: CellStatus,
    loaded: dict[DatasetSpec, LoadedDataset],
) -> None:
    cell_id = identity.cell_id()
    attempt = status.attempt + 1
    ts = ctx.hooks.clock()
    ctx.journal.append(_transition(identity, CellState.RUNNING, attempt, ts))
    ctx.hooks.on_commit(CommitPoint.JOURNAL_WRITE, cell_id, CellState.RUNNING)
    status.state = CellState.RUNNING
    status.attempt = attempt
    status.started_at = ts
    status.finished_at = None
    status.error = None
    artifact = cell_artifact_path(ctx.matrix_dir, cell_id)
    try:
        if cell_id in ctx.hooks.fail_cell_ids or attempt == ctx.hooks.fail_on_attempt.get(cell_id):
            raise RuntimeError("injected cell failure")
        payload = execute_cell(identity, _dataset_for(identity, loaded, ctx.config_dir))
        _atomic_write_json(artifact, payload)
        ctx.hooks.on_commit(CommitPoint.RESULT_COMMIT, cell_id)
        _succeed_cell(ctx, identity, status, attempt, artifact)
    except SimulatedInterrupt:
        raise
    except Exception as exc:
        _fail_cell(ctx, identity, status, attempt, str(exc))


def _process_cell(
    ctx: SessionContext,
    identity: CellIdentity,
    status: CellStatus,
    loaded: dict[DatasetSpec, LoadedDataset],
) -> None:
    action = decide_action(status, ctx.policy, ctx.matrix_dir)
    if action is Action.KEEP:
        return
    if action is Action.SKIP:
        _record_skip(ctx, identity, status)
        return
    if action is Action.RUN:
        _run_cell(ctx, identity, status, loaded)
        return
    never: Never = action
    raise MatrixError(f"unhandled resume action: {never}")


def execute_session(
    ctx: SessionContext,
    replay_records: tuple[dict[str, Any], ...],
    truncated_tail: bool,
) -> tuple[dict[str, CellStatus], dict[str, Any]]:
    _ensure_journal_header(ctx, replay_records)
    statuses = _statuses_from_records(ctx.definition, replay_records)
    loaded: dict[DatasetSpec, LoadedDataset] = {}
    _journal_pending_cells(ctx, statuses, replay_records)
    for identity in ctx.definition.cells:
        _process_cell(ctx, identity, statuses[identity.cell_id()], loaded)
    aggregate = _aggregate_payload(ctx.definition, statuses, truncated_tail)
    _atomic_write_json(ctx.aggregate_path, aggregate)
    ctx.hooks.on_commit(CommitPoint.AGGREGATE_UPDATE, None)
    return statuses, aggregate
