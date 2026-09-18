from __future__ import annotations

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
    assert score_piqa([0, 1], [0, 0]) == 0.5
