from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.cases import (
    FAMILY_SPECS,
    DatasetCaseError,
    TaskKind,
    available_families,
    case_to_record,
    cases_from_loaded,
    family_spec,
    load_family_records,
    load_sample_cases,
    normalize_family_name,
    parse_case,
    record_to_case,
)
from benchmarks.datasets import DatasetRecord, DatasetSpec, JsonlDatasetLoader
from benchmarks.lambada.loader import load_lambada_sample
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import Prediction

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_families_are_registered() -> None:
    names = available_families()
    assert set(names) == {
        "lambada",
        "hellaswag",
        "wikitext2",
        "gsm8k",
        "piqa",
        "arc_easy",
    }


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("LAMBADA", "lambada"),
        ("HellaSwag", "hellaswag"),
        ("WikiText-2", "wikitext2"),
        ("wikitext_2", "wikitext2"),
        ("GSM8K", "gsm8k"),
        ("PIQA", "piqa"),
        ("ARC-Easy", "arc_easy"),
        ("arc", "arc_easy"),
    ],
)
def test_family_aliases(alias: str, canonical: str) -> None:
    assert normalize_family_name(alias) == canonical


@pytest.mark.parametrize("family", list(FAMILY_SPECS))
def test_local_fixtures_produce_canonical_cases(family: str) -> None:
    spec = family_spec(family)
    cases = load_sample_cases(family)

    assert len(cases) >= 2
    ids = [case.example_id for case in cases]
    assert len(ids) == len(set(ids))
    for index, case in enumerate(cases):
        assert case.dataset == spec.name
        assert case.split == "validation"
        assert case.task is spec.task
        assert case.prompt
        assert case.source == "jsonl"
        assert case.metadata["hf_id"] == spec.hf_id
        assert case.example_id == f"{spec.name}:validation:{index:04d}"
        assert case.target
        record = case_to_record(case)
        assert record.prompt == case.prompt
        if spec.task is TaskKind.CLASSIFICATION:
            assert record.is_multiple_choice
            assert record.choices == case.choices
        else:
            assert record.reference == case.expected
            assert not record.is_multiple_choice


@pytest.mark.parametrize("family", list(FAMILY_SPECS))
def test_jsonl_loader_still_accepts_sample_fixtures(family: str) -> None:
    loaded = load_family_records(family)
    assert len(loaded.records) >= 2
    for record in loaded.records:
        assert record.prompt
        if family_spec(family).task is TaskKind.CLASSIFICATION:
            assert record.is_multiple_choice
        else:
            assert record.reference


@pytest.mark.parametrize("family", list(FAMILY_SPECS))
def test_adapter_from_loaded_records(family: str) -> None:
    loaded = load_family_records(family)
    cases = cases_from_loaded(loaded)
    direct = load_sample_cases(family)
    assert [case.example_id for case in cases] == [case.example_id for case in direct]
    assert [case.task for case in cases] == [case.task for case in direct]
    assert [case.prompt for case in cases] == [case.prompt for case in direct]


def test_record_to_case_and_back_keeps_metrics_compatible() -> None:
    record = DatasetRecord(
        prompt="To keep a plant healthy you should",
        choices=["water it regularly.", "store it in a freezer."],
        answer_index=0,
    )
    case = record_to_case(record, dataset="piqa", split="validation", index=0)
    adapted = case_to_record(case)
    accumulator = MetricsAccumulator()
    accumulator.add(adapted, Prediction(output=0, logprob=-0.5, tokens=4))
    summary = accumulator.summary(total_time_s=1.0, vram_gb=1.0)
    assert summary.accuracy == 1.0


def test_synthetic_lambada_records_adapt_to_cloze_cases() -> None:
    records = load_lambada_sample(sample_size=3, seed=42)
    cases = [
        record_to_case(
            record,
            dataset="lambada",
            split="validation",
            index=index,
            source="synthetic_cloze_sample",
        )
        for index, record in enumerate(records)
    ]
    assert all(case.task is TaskKind.CLOZE for case in cases)
    assert all(
        case.expected == record.reference
        for case, record in zip(cases, records, strict=True)
    )
    assert [case_to_record(case).prompt for case in cases] == [
        record.prompt for record in records
    ]


def test_jsonl_loader_ignores_example_id_field() -> None:
    path = REPO_ROOT / "configs" / "datasets" / "lambada.sample.jsonl"
    loaded = JsonlDatasetLoader().load(
        DatasetSpec(name="lambada", source="jsonl", path=str(path))
    )
    assert loaded.records[0].prompt.startswith("The quick brown fox")
    assert loaded.records[0].reference == "dog"
    assert loaded.records[0].__dataclass_fields__.keys() >= {
        "prompt",
        "reference",
        "choices",
        "answer_index",
    }
    assert "example_id" not in loaded.records[0].__dataclass_fields__


def test_unknown_family_fails_clearly() -> None:
    with pytest.raises(DatasetCaseError, match="unknown dataset family"):
        load_sample_cases("not-a-dataset")


def test_max_samples_zero_returns_empty() -> None:
    assert load_sample_cases("lambada", max_samples=0) == []


def test_max_samples_negative_fails() -> None:
    with pytest.raises(DatasetCaseError, match="non-negative"):
        load_sample_cases("lambada", max_samples=-1)


def test_whitespace_identity_is_rejected() -> None:
    with pytest.raises(DatasetCaseError, match="example_id"):
        parse_case(
            {
                "example_id": "   ",
                "dataset": "lambada",
                "split": "validation",
                "prompt": "The quick brown fox jumps over the lazy",
                "task": "cloze",
                "expected": "dog",
            }
        )
