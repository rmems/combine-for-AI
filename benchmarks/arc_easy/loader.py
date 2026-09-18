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


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in value or []]


def _aligned_arc_labels(texts: list[str], labels: list[str]) -> list[str] | None:
    if not labels:
        return [str(index) for index in range(len(texts))]
    if len(labels) != len(texts):
        return None
    return labels


def _arc_choice_lists(choices: Any) -> tuple[list[str], list[str]] | None:
    if not isinstance(choices, dict):
        return None
    texts = _as_str_list(choices.get("text"))
    if len(texts) < 2:
        return None
    labels = _aligned_arc_labels(texts, _as_str_list(choices.get("label")))
    if labels is None:
        return None
    return texts, labels


def map_arc_easy_row(row: dict[str, Any]) -> DatasetRecord | None:
    record = canonical_record(row)
    if record is not None:
        return record

    question = row.get("question")
    parsed = _arc_choice_lists(row.get("choices"))
    answer_key = row.get("answerKey")
    if question is None or parsed is None or answer_key is None:
        return None
    texts, labels = parsed
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
