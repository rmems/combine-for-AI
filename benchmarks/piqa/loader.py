from __future__ import annotations

from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import MappedDatasetLoader, register_row_mapper
from benchmarks.dataset_validation import multiple_choice_accuracy
from benchmarks.dataset_types import DatasetRecord


def _piqa_answer_index(label: Any) -> int | None:
    if label is None:
        return None
    if isinstance(label, str) and not label.strip():
        return None
    if isinstance(label, bool) or not isinstance(label, (int, str)):
        raise ValueError("PIQA label must be 0 or 1")
    answer_index = int(label)
    if answer_index == -1:
        return None
    if answer_index not in (0, 1):
        raise ValueError("PIQA label must be 0 or 1")
    return answer_index


def map_piqa_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map PIQA using lm-eval ``piqa`` fields: goal/sol1/sol2/label."""
    record = canonical_record(row)
    if record is not None:
        return record

    goal = row.get("goal")
    sol1 = row.get("sol1")
    sol2 = row.get("sol2")
    answer_index = _piqa_answer_index(row.get("label"))
    if goal is None or sol1 is None or sol2 is None or answer_index is None:
        return None
    return DatasetRecord(
        prompt=str(goal).strip(),
        choices=[str(sol1), str(sol2)],
        answer_index=answer_index,
    )


register_row_mapper("piqa", map_piqa_row)


class PIQALoader(MappedDatasetLoader):
    """Load PIQA (lm-eval ``piqa``) as a two-choice physical-reasoning task."""

    catalog_name = "piqa"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_piqa_row(row)


def score_piqa(predictions: list[int], answer_indices: list[int]) -> float:
    return multiple_choice_accuracy(predictions, answer_indices)
