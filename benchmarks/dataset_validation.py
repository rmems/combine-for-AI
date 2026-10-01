"""Schema checks for loaded benchmark datasets."""

from __future__ import annotations

from typing import Never

from benchmarks.dataset_types import (
    CatalogEntry,
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
    TaskKind,
)


def _assert_never(value: Never) -> Never:
    raise ValueError(f"unhandled task kind: {value!r}")


def _validate_multiple_choice(record: DatasetRecord, label: str) -> None:
    if not record.choices or len(record.choices) < 2:
        raise ValueError(f"{label} must include at least two choices")
    if any(not str(choice).strip() for choice in record.choices):
        raise ValueError(f"{label} has an empty choice")
    if record.answer_index is None:
        raise ValueError(f"{label} is missing answer_index")
    if not 0 <= record.answer_index < len(record.choices):
        raise ValueError(
            f"{label} answer_index {record.answer_index} is out of range "
            f"for {len(record.choices)} choices"
        )


def _validate_reference(record: DatasetRecord, label: str) -> None:
    if record.reference is None or not str(record.reference).strip():
        raise ValueError(f"{label} is missing a reference")


def _validate_generic(record: DatasetRecord, label: str) -> None:
    if record.choices is not None or record.answer_index is not None:
        _validate_multiple_choice(record, label)


def _validate_record(record: DatasetRecord, task: TaskKind, label: str) -> None:
    if task is not TaskKind.LANGUAGE_MODELING and not str(record.prompt).strip():
        raise ValueError(f"{label} has an empty prompt")
    match task:
        case TaskKind.CLOZE | TaskKind.MATH:
            _validate_reference(record, label)
        case TaskKind.MULTIPLE_CHOICE:
            _validate_multiple_choice(record, label)
        case TaskKind.LANGUAGE_MODELING:
            return
        case TaskKind.GENERIC:
            _validate_generic(record, label)
        case _:
            _assert_never(task)


def _required_min_samples(spec: DatasetSpec, entry: CatalogEntry | None) -> int:
    if spec.max_samples == 0:
        return 0
    if spec.min_samples is not None:
        return spec.min_samples
    if entry is None:
        return 0
    return entry.min_samples


def validate_loaded(loaded: LoadedDataset, entry: CatalogEntry | None) -> None:
    spec = loaded.spec
    if spec.max_samples is not None and spec.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if spec.min_samples is not None and spec.min_samples < 0:
        raise ValueError("min_samples must be non-negative")

    task = entry.task if entry is not None else TaskKind.GENERIC
    min_samples = _required_min_samples(spec, entry)
    if len(loaded.records) < min_samples:
        raise ValueError(
            f"{spec.name}: expected at least {min_samples} samples, "
            f"got {len(loaded.records)}"
        )
    for index, record in enumerate(loaded.records):
        _validate_record(record, task, f"{spec.name} record {index}")


def multiple_choice_accuracy(
    predictions: list[int], answer_indices: list[int]
) -> float:
    if len(predictions) != len(answer_indices):
        raise ValueError("predictions and answer_indices must have the same length")
    if not predictions:
        return 0.0
    correct = sum(
        int(pred == answer)
        for pred, answer in zip(predictions, answer_indices, strict=True)
    )
    return correct / len(predictions)
