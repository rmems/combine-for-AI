from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmarks.cases import (
    REPO_ROOT,
    DatasetCase,
    DatasetCaseError,
    FamilySpec,
    family_spec,
    parse_case,
    resolve_example_id,
    validate_case_batch,
)
from benchmarks.datasets import DatasetSpec, JsonlDatasetLoader, LoadedDataset


def _row_to_case(
    payload: Mapping[str, Any],
    *,
    spec: FamilySpec,
    split: str,
    index: int,
    path: Path,
) -> DatasetCase:
    explicit = payload.get("example_id") or payload.get("id")
    expected = payload.get("expected", payload.get("reference"))
    row_split = str(payload.get("split", split))
    extra = payload.get("metadata")
    extra_meta = dict(extra) if isinstance(extra, Mapping) else {}
    metadata = {
        **extra_meta,
        "path": str(path),
        "row_index": index,
        "hf_id": spec.hf_id,
        "hf_subset": spec.hf_subset,
    }
    return parse_case(
        {
            "example_id": resolve_example_id(
                spec.name,
                row_split,
                index,
                str(explicit) if explicit else None,
            ),
            "dataset": spec.name,
            "split": row_split,
            "prompt": payload["prompt"],
            "task": spec.task,
            "expected": expected,
            "choices": payload.get("choices"),
            "answer_index": payload.get("answer_index"),
            "source": "jsonl",
            "metadata": metadata,
        }
    )


def _parse_jsonl_object(
    line: str,
    *,
    line_number: int,
    path: Path,
    dataset: str,
) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except ValueError as exc:
        raise DatasetCaseError(
            f"invalid json on line {line_number} in {path}",
            dataset=dataset,
            example_id=None,
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetCaseError(
            f"jsonl row must be an object (line {line_number} in {path})",
            dataset=dataset,
            example_id=None,
            expected="object",
            observed=type(payload).__name__,
        )
    if "prompt" not in payload:
        raise DatasetCaseError(
            f"missing prompt (line {line_number} in {path})",
            dataset=dataset,
            example_id=payload.get("example_id") or payload.get("id"),
            expected="prompt",
            observed=None,
        )
    return payload


def _read_sample_rows(
    handle: Iterable[str],
    *,
    spec: FamilySpec,
    split: str,
    path: Path,
    max_samples: int | None,
) -> list[DatasetCase]:
    cases: list[DatasetCase] = []
    for line_number, line in enumerate(handle, start=1):
        if max_samples is not None and len(cases) >= max_samples:
            break
        payload = _parse_jsonl_object(
            line,
            line_number=line_number,
            path=path,
            dataset=spec.name,
        )
        if payload is None:
            continue
        cases.append(
            _row_to_case(
                payload,
                spec=spec,
                split=split,
                index=len(cases),
                path=path,
            )
        )
    return cases


def load_sample_cases(
    family: str,
    *,
    split: str = "validation",
    path: str | Path | None = None,
    max_samples: int | None = None,
) -> list[DatasetCase]:
    """Load committed local fixtures. Never downloads and never opens the network."""
    spec = family_spec(family)
    jsonl_path = Path(path) if path is not None else REPO_ROOT / spec.sample_relpath
    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset sample file not found: {jsonl_path}")
    if max_samples is not None and max_samples < 0:
        raise DatasetCaseError(
            "max_samples must be non-negative",
            dataset=spec.name,
            example_id=None,
            expected=">= 0",
            observed=max_samples,
        )

    with jsonl_path.open("r", encoding="utf-8") as handle:
        cases = _read_sample_rows(
            handle,
            spec=spec,
            split=split,
            path=jsonl_path,
            max_samples=max_samples,
        )
    return validate_case_batch(cases)


def load_family_records(
    family: str,
    *,
    split: str = "validation",
    path: str | Path | None = None,
    max_samples: int | None = None,
) -> LoadedDataset:
    """Load the same local fixture through the existing JSONL loader."""
    spec = family_spec(family)
    jsonl_path = Path(path) if path is not None else REPO_ROOT / spec.sample_relpath
    dataset_spec = DatasetSpec(
        name=spec.name,
        split=split,
        source="jsonl",
        path=str(jsonl_path),
        hf_id=spec.hf_id,
        hf_subset=spec.hf_subset,
        max_samples=max_samples,
    )
    return JsonlDatasetLoader().load(dataset_spec)
