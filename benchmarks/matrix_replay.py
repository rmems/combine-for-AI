"""Journal header/transition records and replay into cell statuses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from benchmarks.journal import JOURNAL_SCHEMA, CellState, JournalError
from benchmarks.matrix import CellIdentity, IncompatibleMatrixError, MatrixDefinition
from benchmarks.matrix_types import CellStatus


@dataclass(frozen=True)
class TransitionFields:
    artifact_checksum: str | None = None
    artifact_path: str | None = None
    input_digest: str | None = None
    fingerprint: str | None = None
    error: str | None = None


def journal_header(definition: MatrixDefinition, created_at: str) -> dict[str, Any]:
    return {
        "cell_ids": [cell.cell_id() for cell in definition.cells],
        "config_revision": definition.config_revision,
        "created_at": created_at,
        "definition_fingerprint": definition.fingerprint,
        "kind": "header",
        "matrix_name": definition.name,
        "schema": JOURNAL_SCHEMA,
    }


def transition_record(
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


def apply_transition(status: CellStatus, record: Mapping[str, Any]) -> None:
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


def statuses_from_records(
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
        apply_transition(statuses[cell_id], record)
    return statuses


def journaled_cell_ids(records: tuple[dict[str, Any], ...]) -> set[str]:
    return {
        str(record["cell_id"])
        for record in records
        if record.get("kind") == "transition" and "cell_id" in record
    }


def aggregate_payload(
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
