from __future__ import annotations

from typing import Any

from benchmarks.dataset_support import (
    MappedDatasetLoader,
    canonical_record,
    multiple_choice_accuracy,
    register_row_mapper,
)
from benchmarks.dataset_types import DatasetRecord


def map_hellaswag_row(row: dict[str, Any]) -> DatasetRecord | None:
    record = canonical_record(row)
    if record is not None:
        return record

    ctx = row.get("ctx")
    if not ctx:
        ctx_a = str(row.get("ctx_a") or "").strip()
        ctx_b = str(row.get("ctx_b") or "").strip()
        ctx = f"{ctx_a} {ctx_b}".strip()
    endings = row.get("endings")
    label = row.get("label")
    if not ctx or endings is None or label is None:
        return None
    choices = [str(ending) for ending in endings]
    if len(choices) < 2:
        return None
    return DatasetRecord(
        prompt=str(ctx).strip(),
        choices=choices,
        answer_index=int(label),
    )


register_row_mapper("hellaswag", map_hellaswag_row)


class HellaSwagLoader(MappedDatasetLoader):
    """Load HellaSwag as a multiple-choice commonsense completion task."""

    catalog_name = "hellaswag"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_hellaswag_row(row)


def score_hellaswag(
    predictions: list[int], answer_indices: list[int]
) -> float:
    """Accuracy of selected ending indices against gold labels."""
    return multiple_choice_accuracy(predictions, answer_indices)
