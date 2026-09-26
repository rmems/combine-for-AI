from __future__ import annotations

import pytest

from benchmarks.dataset_support import CATALOG, sample_path_for
from benchmarks.datasets import DatasetRecord, DatasetSpec, default_dataset_registry
from benchmarks.lambada.loader import (
    LAMBADALoader,
    calculate_cloze_accuracy,
    calculate_perplexity,
    load_lambada_sample,
    map_lambada_row,
)


def test_load_lambada_sample() -> None:
    records = load_lambada_sample(sample_size=10, seed=42)

    assert len(records) == 2
    assert all(isinstance(record.prompt, str) for record in records)
    assert all(isinstance(record.reference, str) for record in records)
    for record in records:
        assert record.prompt
        assert record.reference
        assert not record.is_multiple_choice


def test_lambada_loader_from_sample_jsonl() -> None:
    loader = LAMBADALoader()
    spec = DatasetSpec(
        name="lambada_test",
        source="lambada",
        path=str(sample_path_for(CATALOG["lambada"])),
        max_samples=2,
    )
    dataset = loader.load(spec)

    assert dataset.spec == spec
    assert len(dataset.records) == 2
    assert dataset.metadata["source"] == "jsonl"
    assert dataset.records[0].reference == "dog"


def test_lambada_loader_offline_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", None)
    loader = LAMBADALoader()
    dataset = loader.load(DatasetSpec(name="lambada", source="lambada"))

    assert len(dataset.records) == 2
    assert dataset.metadata["source"] == "jsonl"
    assert dataset.metadata.get("fallback") is True


def test_lambada_loader_with_max_samples() -> None:
    loader = LAMBADALoader()
    spec = DatasetSpec(
        name="lambada_limited",
        source="lambada",
        path=str(sample_path_for(CATALOG["lambada"])),
        max_samples=1,
    )
    dataset = loader.load(spec)
    assert len(dataset.records) == 1


def test_map_lambada_native_text() -> None:
    record = map_lambada_row({"text": "She walked to the store"})
    assert record is not None
    assert record.prompt == "She walked to the "
    assert record.reference == "store"


def test_map_lambada_skips_empty_text() -> None:
    assert map_lambada_row({"text": "   "}) is None
    assert map_lambada_row({"text": "oneword"}) is None


def test_calculate_cloze_accuracy() -> None:
    predictions = ["store", "chair", "car", "tree", "book"]
    references = ["store", "chair", "bike", "tree", "book"]
    accuracy = calculate_cloze_accuracy(predictions, references)
    assert accuracy == pytest.approx(4 / 5)


def test_calculate_cloze_accuracy_empty() -> None:
    assert calculate_cloze_accuracy([], []) == 0.0


def test_calculate_cloze_accuracy_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_cloze_accuracy(["store", "chair"], ["store"])


def test_calculate_cloze_accuracy_one_sided_empty() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_cloze_accuracy([], ["store"])


def test_calculate_perplexity() -> None:
    log_probs = [-1.0, -2.0, -1.5, -0.5, -1.0]
    token_counts = [3, 2]
    perplexity = calculate_perplexity(log_probs, token_counts)
    expected = 2 ** (6.0 / 5)
    assert perplexity == pytest.approx(expected)


def test_calculate_perplexity_empty() -> None:
    assert calculate_perplexity([], []) == float("inf")


def test_calculate_perplexity_mismatch() -> None:
    with pytest.raises(ValueError, match="must match total tokens"):
        calculate_perplexity([-1.0, -2.0, -1.5], [2, 2])


def test_lambada_in_registry() -> None:
    registry = default_dataset_registry()
    assert "jsonl" in registry.available_sources()
    assert "hf" in registry.available_sources()
    assert "lambada" in registry.available_sources()
    assert isinstance(registry.loader_for("lambada"), LAMBADALoader)


def test_lambada_rejects_negative_max_samples() -> None:
    loader = LAMBADALoader()
    with pytest.raises(ValueError, match="non-negative"):
        loader.load(DatasetSpec(name="bad", source="lambada", max_samples=-1))


def test_calculate_perplexity_rejects_negative_token_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_perplexity([-1.0], [-1, 2])


def test_lambada_max_samples_zero() -> None:
    loader = LAMBADALoader()
    dataset = loader.load(
        DatasetSpec(
            name="empty",
            source="lambada",
            path=str(sample_path_for(CATALOG["lambada"])),
            max_samples=0,
        )
    )
    assert dataset.records == []


def test_dataset_record_cloze_format() -> None:
    record = DatasetRecord(prompt="She walked to the ", reference="store")
    assert not record.is_multiple_choice
    assert record.choices is None
    assert record.answer_index is None
