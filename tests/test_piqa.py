from __future__ import annotations

import math

import pytest

from benchmarks.dataset_support import CATALOG, sample_path_for
from benchmarks.datasets import DatasetSpec
from benchmarks.piqa.loader import PIQALoader, map_piqa_row, score_piqa


def test_piqa_sample_jsonl() -> None:
    dataset = PIQALoader().load(
        DatasetSpec(
            name="piqa",
            source="piqa",
            path=str(sample_path_for(CATALOG["piqa"])),
        )
    )
    assert len(dataset.records) == 2
    record = dataset.records[0]
    assert record.is_multiple_choice
    assert record.choices is not None
    assert len(record.choices) == 2
    assert record.answer_index == 0


def test_map_piqa_native_fields() -> None:
    record = map_piqa_row(
        {
            "goal": "To keep a plant healthy you should",
            "sol1": "water it regularly.",
            "sol2": "store it in a freezer.",
            "label": 0,
        }
    )
    assert record is not None
    assert record.choices == ["water it regularly.", "store it in a freezer."]
    assert record.answer_index == 0


def test_score_piqa() -> None:
    assert math.isclose(score_piqa([0, 1], [0, 0]), 0.5)


@pytest.mark.parametrize("label", [None, "", "  ", -1, "-1", " -1 "])
def test_piqa_skips_unlabeled_rows(label) -> None:
    assert map_piqa_row({"goal": "Question", "sol1": "a", "sol2": "b", "label": label}) is None


@pytest.mark.parametrize("label", [0, 1, "0", "1"])
def test_piqa_converts_valid_labels(label) -> None:
    record = map_piqa_row({"goal": "Question", "sol1": "a", "sol2": "b", "label": label})
    assert record is not None
    assert record.answer_index == int(label)


@pytest.mark.parametrize("label", [-2, 2])
def test_piqa_rejects_out_of_range_labels(label) -> None:
    with pytest.raises(ValueError, match="label"):
        map_piqa_row({"goal": "Question", "sol1": "a", "sol2": "b", "label": label})


@pytest.mark.parametrize("label", [True, False, 1.0, 1.9])
def test_piqa_rejects_non_integer_label_types(label) -> None:
    with pytest.raises(ValueError, match="label"):
        map_piqa_row(
            {"goal": "Question", "sol1": "a", "sol2": "b", "label": label}
        )
