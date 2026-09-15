from __future__ import annotations

from typing import Any

from benchmarks.dataset_support import (
    MappedDatasetLoader,
    canonical_record,
    multiple_choice_accuracy,
    register_row_mapper,
)
from benchmarks.dataset_types import DatasetRecord


def _arc_answer_index(answer_key: Any, labels: list[str]) -> int:
    key = str(answer_key).strip()
    if key in labels:
        return labels.index(key)
    upper = [label.upper() for label in labels]
    if key.upper() in upper:
        return upper.index(key.upper())
    if key.isdigit():
        number = int(key)
        as_text = str(number)
        if as_text in labels:
            return labels.index(as_text)
        if 1 <= number <= len(labels):
            return number - 1
        if 0 <= number < len(labels):
            return number
    raise ValueError(f"cannot resolve ARC answerKey {answer_key!r} against {labels}")


def map_arc_easy_row(row: dict[str, Any]) -> DatasetRecord | None:
    record = canonical_record(row)
    if record is not None:
        return record

    question = row.get("question")
    choices = row.get("choices")
    answer_key = row.get("answerKey")
    if question is None or choices is None or answer_key is None:
        return None
    if not isinstance(choices, dict):
        return None
    texts = [str(text) for text in choices.get("text") or []]
    labels = [str(label) for label in choices.get("label") or []]
    if len(texts) < 2:
        return None
    if not labels:
        labels = [str(index) for index in range(len(texts))]
    return DatasetRecord(
        prompt=str(question).strip(),
        choices=texts,
        answer_index=_arc_answer_index(answer_key, labels),
    )


register_row_mapper("arc_easy", map_arc_easy_row)


class ARCEasyLoader(MappedDatasetLoader):
    """Load ARC-Easy science questions as multiple-choice records."""

    catalog_name = "arc_easy"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_arc_easy_row(row)


def score_arc_easy(
    predictions: list[int], answer_indices: list[int]
) -> float:
    return multiple_choice_accuracy(predictions, answer_indices)
