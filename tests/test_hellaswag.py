from __future__ import annotations

from benchmarks.dataset_support import CATALOG, sample_path_for
from benchmarks.datasets import DatasetSpec
from benchmarks.hellaswag.loader import (
    HellaSwagLoader,
    map_hellaswag_row,
    score_hellaswag,
)


def test_hellaswag_sample_jsonl() -> None:
    dataset = HellaSwagLoader().load(
        DatasetSpec(
            name="hellaswag",
            source="hellaswag",
            path=str(sample_path_for(CATALOG["hellaswag"])),
        )
    )
    assert len(dataset.records) == 2
    record = dataset.records[0]
    assert record.is_multiple_choice
    assert record.choices is not None
    assert record.answer_index == 0
    assert "adds sauce" in record.choices[0]


def test_map_hellaswag_native_fields() -> None:
    record = map_hellaswag_row(
        {
            "ctx": "A woman is cooking pasta. She drains the pot and then",
            "endings": [
                "adds sauce and serves dinner.",
                "puts it back on the stove to freeze.",
                "throws the pot out the window.",
                "waits for the pasta to freeze.",
            ],
            "label": "0",
        }
    )
    assert record is not None
    assert record.answer_index == 0
    assert len(record.choices or []) == 4


def test_map_hellaswag_ctx_a_b() -> None:
    record = map_hellaswag_row(
        {
            "ctx_a": "A cyclist reaches a red light.",
            "ctx_b": "They",
            "endings": ["stop and wait for green.", "keep pedaling through traffic."],
            "label": 0,
        }
    )
    assert record is not None
    assert record.prompt == "A cyclist reaches a red light. They"


def test_score_hellaswag() -> None:
    assert score_hellaswag([0, 1, 0], [0, 1, 1]) == 2 / 3
    assert score_hellaswag([], []) == 0.0
