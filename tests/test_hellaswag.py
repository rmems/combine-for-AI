from __future__ import annotations

import math

import pytest

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


@pytest.mark.parametrize("endings", ["yes", ["yes", 2]])
def test_map_hellaswag_rejects_non_string_endings(endings) -> None:
    with pytest.raises(ValueError, match="endings must be a list of strings"):
        map_hellaswag_row(
            {"ctx": "Question", "endings": endings, "label": 0}
        )


def test_score_hellaswag() -> None:
    assert math.isclose(score_hellaswag([0, 1, 0], [0, 1, 1]), 2 / 3)
    assert math.isclose(score_hellaswag([], []), 0.0)


def test_hellaswag_test_split_is_not_scored(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("unlabeled test split must not be fetched")

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", boom)
    with pytest.raises(ValueError, match="validation"):
        HellaSwagLoader().load(
            DatasetSpec(name="hellaswag", source="hellaswag", split="test")
        )


def test_hellaswag_labeled_local_test_split_is_scored(tmp_path) -> None:
    path = tmp_path / "hellaswag-test.jsonl"
    path.write_text(
        '{"prompt":"Question","choices":["a","b"],"answer_index":1}\n',
        encoding="utf-8",
    )
    loaded = HellaSwagLoader().load(
        DatasetSpec(
            name="hellaswag",
            source="hellaswag",
            split="test",
            path=str(path),
        )
    )
    assert loaded.records[0].answer_index == 1
