from __future__ import annotations

from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import (
    MappedDatasetLoader,
    multiple_choice_accuracy,
    register_row_mapper,
)
from benchmarks.dataset_types import DatasetRecord


def _hellaswag_prompt(row: dict[str, Any]) -> str:
    ctx = row.get("ctx")
    if ctx:
        return str(ctx).strip()
    ctx_a = str(row.get("ctx_a") or "").strip()
    ctx_b = str(row.get("ctx_b") or "").strip()
    return f"{ctx_a} {ctx_b}".strip()


def map_hellaswag_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map HellaSwag using lm-eval ``hellaswag`` fields: ctx/endings/label."""
    record = canonical_record(row)
    if record is not None:
        return record

    prompt = _hellaswag_prompt(row)
    endings = row.get("endings")
    label = row.get("label")
    if not prompt or endings is None or label is None:
        return None
    choices = [str(ending) for ending in endings]
    if len(choices) < 2:
        return None
    return DatasetRecord(
        prompt=prompt,
        choices=choices,
        answer_index=int(label),
    )


register_row_mapper("hellaswag", map_hellaswag_row)


class HellaSwagLoader(MappedDatasetLoader):
    """Load HellaSwag (lm-eval ``hellaswag``) as a multiple-choice task."""

    catalog_name = "hellaswag"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_hellaswag_row(row)


def score_hellaswag(
    predictions: list[int], answer_indices: list[int]
) -> float:
    """Accuracy of selected ending indices against gold labels."""
    return multiple_choice_accuracy(predictions, answer_indices)
