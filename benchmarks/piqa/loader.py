from __future__ import annotations

from typing import Any

from benchmarks.dataset_support import (
    MappedDatasetLoader,
    canonical_record,
    multiple_choice_accuracy,
    register_row_mapper,
)
from benchmarks.dataset_types import DatasetRecord


def map_piqa_row(row: dict[str, Any]) -> DatasetRecord | None:
    record = canonical_record(row)
    if record is not None:
        return record

    goal = row.get("goal")
    sol1 = row.get("sol1")
    sol2 = row.get("sol2")
    label = row.get("label")
    if goal is None or sol1 is None or sol2 is None or label is None:
        return None
    return DatasetRecord(
        prompt=str(goal).strip(),
        choices=[str(sol1), str(sol2)],
        answer_index=int(label),
    )


register_row_mapper("piqa", map_piqa_row)


class PIQALoader(MappedDatasetLoader):
    """Load PIQA as a two-choice physical commonsense task."""

    catalog_name = "piqa"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_piqa_row(row)


def score_piqa(predictions: list[int], answer_indices: list[int]) -> float:
    return multiple_choice_accuracy(predictions, answer_indices)
