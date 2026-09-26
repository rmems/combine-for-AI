"""JSONL parsing and canonical record helpers shared by mapped loaders."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from benchmarks.dataset_types import DatasetRecord

RowMapper = Callable[[dict[str, Any]], DatasetRecord | None]


def canonical_record(row: dict[str, Any]) -> DatasetRecord | None:
    if "prompt" not in row or row["prompt"] is None:
        return None
    choices = row.get("choices")
    if choices is not None:
        choices = [str(choice) for choice in choices]
    answer_index = row.get("answer_index")
    if answer_index is not None:
        answer_index = int(answer_index)
    reference = row.get("reference")
    return DatasetRecord(
        prompt=str(row["prompt"]),
        reference=None if reference is None else str(reference),
        choices=choices,
        answer_index=answer_index,
    )


def row_as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    raise TypeError(f"cannot convert dataset row of type {type(row)!r} to dict")


def apply_max_samples(
    records: list[DatasetRecord], max_samples: int | None
) -> list[DatasetRecord]:
    if max_samples is None:
        return records
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    return records[:max_samples]


def iter_jsonl_payloads(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json_loads_object(line, line_number, path)
            except ValueError:
                raise
            yield line_number, payload


def json_loads_object(line: str, line_number: int, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json on line {line_number} in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"jsonl object on line {line_number} in {path} must be a mapping"
        )
    return payload


def require_canonical_record(row: dict[str, Any]) -> DatasetRecord:
    record = canonical_record(row)
    if record is None:
        raise ValueError("missing required field 'prompt'")
    return record


def _map_jsonl_row(
    map_row: RowMapper,
    payload: dict[str, Any],
    *,
    line_number: int,
    path: Path,
) -> DatasetRecord | None:
    try:
        return map_row(payload)
    except ValueError as exc:
        raise ValueError(f"{exc} on line {line_number} in {path}") from exc


def _normalized_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None:
        return None
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    return max_samples


def records_from_jsonl(
    path: Path,
    map_row: RowMapper,
    max_samples: int | None,
) -> list[DatasetRecord]:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    limit = _normalized_max_samples(max_samples)
    if limit == 0:
        return []

    records: list[DatasetRecord] = []
    for line_number, payload in iter_jsonl_payloads(path):
        record = _map_jsonl_row(
            map_row, payload, line_number=line_number, path=path
        )
        if record is None:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records
