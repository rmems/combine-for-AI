from __future__ import annotations

from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import (
    MappedDatasetLoader,
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


def _aligned_arc_labels(texts: list[str], labels: list[str]) -> list[str]:
    if not labels:
        return [str(index) for index in range(len(texts))]
    if len(labels) != len(texts):
        raise ValueError(
            f"ARC choices.label length {len(labels)} != choices.text length "
            f"{len(texts)}"
        )
    return labels


def _arc_choice_lists(choices: Any) -> tuple[list[str], list[str]] | None:
    if not isinstance(choices, dict):
        return None
    texts = _as_str_list(choices.get("text"))
    if len(texts) < 2:
        return None
    labels = _aligned_arc_labels(texts, _as_str_list(choices.get("label")))
    return texts, labels


def map_arc_easy_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map ARC-Easy using lm-eval ``arc_easy`` question/choices/answerKey."""
    record = canonical_record(row)
    if record is not None:
        return record

    question = row.get("question")
    parsed = _arc_choice_lists(row.get("choices"))
    answer_key = row.get("answerKey")
    if question is None or parsed is None or answer_key is None:
        return None
    texts, labels = parsed
    answer_index = _arc_answer_index(answer_key, labels)
    if not 0 <= answer_index < len(texts):
        raise ValueError(
            f"ARC answer index {answer_index} is out of range for "
            f"{len(texts)} choices"
        )
    return DatasetRecord(
        prompt=str(question).strip(),
        choices=texts,
        answer_index=answer_index,
    )


register_row_mapper("arc_easy", map_arc_easy_row)


class ARCEasyLoader(MappedDatasetLoader):
    """Load ARC-Easy (lm-eval ``arc_easy``) science multiple-choice items."""

    catalog_name = "arc_easy"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_arc_easy_row(row)


def score_arc_easy(
    predictions: list[int], answer_indices: list[int]
) -> float:
    return multiple_choice_accuracy(predictions, answer_indices)
