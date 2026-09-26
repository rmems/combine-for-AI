from __future__ import annotations

import pytest

from benchmarks.arc_easy.loader import (
    ARCEasyLoader,
    map_arc_easy_row,
    score_arc_easy,
)
from benchmarks.dataset_support import CATALOG, sample_path_for
from benchmarks.datasets import DatasetSpec


def test_arc_easy_sample_jsonl() -> None:
    dataset = ARCEasyLoader().load(
        DatasetSpec(
            name="arc_easy",
            source="arc_easy",
            path=str(sample_path_for(CATALOG["arc_easy"])),
        )
    )
    assert len(dataset.records) == 2
    record = dataset.records[0]
    assert record.is_multiple_choice
    assert record.choices is not None
    assert record.choices[0] == "the Sun"
    assert record.answer_index == 0


def test_map_arc_easy_native_letter_key() -> None:
    record = map_arc_easy_row(
        {
            "question": "What is the main source of energy for Earth's climate?",
            "choices": {
                "text": ["the Sun", "the Moon", "volcanoes", "tides"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        }
    )
    assert record is not None
    assert record.answer_index == 0


def test_map_arc_easy_numeric_key() -> None:
    record = map_arc_easy_row(
        {
            "question": "Which object is best for measuring temperature?",
            "choices": {
                "text": ["thermometer", "ruler", "scale", "stopwatch"],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "1",
        }
    )
    assert record is not None
    assert record.answer_index == 0


def test_map_arc_easy_unknown_key() -> None:
    with pytest.raises(ValueError, match="cannot resolve ARC answerKey"):
        map_arc_easy_row(
            {
                "question": "Q?",
                "choices": {"text": ["a", "b"], "label": ["A", "B"]},
                "answerKey": "Z",
            }
        )


def test_map_arc_easy_rejects_mismatched_choice_lengths() -> None:
    with pytest.raises(ValueError, match="choices.label length 2 != choices.text length 4"):
        map_arc_easy_row(
            {
                "question": "Q?",
                "choices": {
                    "text": ["a", "b", "c", "d"],
                    "label": ["A", "B"],
                },
                "answerKey": "2",
            }
        )


def test_score_arc_easy() -> None:
    assert score_arc_easy([0, 2], [0, 2]) == 1.0
