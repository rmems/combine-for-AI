from __future__ import annotations

import pytest

from benchmarks.datasets import DatasetSpec, default_dataset_registry
from benchmarks.lambada.loader import (
    LAMBADALoader,
    calculate_cloze_accuracy,
    calculate_perplexity,
    load_lambada_sample,
)


def test_load_lambada_sample() -> None:
    """Test that LAMBADA sample loading works."""
    records = load_lambada_sample(sample_size=10, seed=42)
    
    assert len(records) == 10
    assert all(isinstance(record.prompt, str) for record in records)
    assert all(isinstance(record.reference, str) for record in records)
    
    # Check that we have prompts and references
    for record in records:
        assert record.prompt
        assert record.reference
        assert not record.is_multiple_choice


def test_lambada_loader() -> None:
    """Test the LAMBADALoader class."""
    loader = LAMBADALoader(sample_size=5, seed=123)
    spec = DatasetSpec(name="lambada_test", max_samples=5)
    
    dataset = loader.load(spec)
    
    assert dataset.spec == spec
    assert len(dataset.records) == 5
    assert dataset.metadata["source"] == "synthetic_cloze_sample"
    assert dataset.metadata["sample_size"] == 5
    assert dataset.metadata["seed"] == 123


def test_lambada_loader_default() -> None:
    """Test LAMBADALoader with default parameters."""
    loader = LAMBADALoader()
    spec = DatasetSpec(name="lambada_default")
    
    dataset = loader.load(spec)
    
    assert len(dataset.records) == 50  # default sample size
    assert dataset.metadata["sample_size"] == 50


def test_lambada_loader_with_max_samples() -> None:
    """Test LAMBADALoader respects max_samples."""
    loader = LAMBADALoader(sample_size=100, seed=456)
    spec = DatasetSpec(name="lambada_limited", max_samples=3)
    
    dataset = loader.load(spec)
    
    assert len(dataset.records) == 3  # respects max_samples
    assert dataset.metadata["sample_size"] == 3


def test_calculate_cloze_accuracy() -> None:
    """Test cloze accuracy calculation."""
    predictions = ["store", "chair", "car", "tree", "book"]
    references = ["store", "chair", "bike", "tree", "book"]
    
    accuracy = calculate_cloze_accuracy(predictions, references)
    
    # 4 out of 5 correct (car vs bike is wrong)
    expected = 4 / 5
    assert accuracy == pytest.approx(expected)


def test_calculate_cloze_accuracy_empty() -> None:
    """Test cloze accuracy with empty lists."""
    accuracy = calculate_cloze_accuracy([], [])
    assert accuracy == 0.0


def test_calculate_cloze_accuracy_mismatch() -> None:
    """Test cloze accuracy with mismatched lengths."""
    predictions = ["store", "chair"]
    references = ["store"]
    
    with pytest.raises(ValueError, match="same length"):
        calculate_cloze_accuracy(predictions, references)


def test_calculate_cloze_accuracy_one_sided_empty() -> None:
    with pytest.raises(ValueError, match="same length"):
        calculate_cloze_accuracy([], ["store"])


def test_calculate_perplexity() -> None:
    """Test perplexity calculation."""
    # Example: 2 examples, first has 3 tokens, second has 2 tokens
    # Log probabilities for each token (base 2)
    log_probs = [-1.0, -2.0, -1.5, -0.5, -1.0]  # 5 tokens total
    token_counts = [3, 2]  # 3 tokens in first example, 2 in second
    
    perplexity = calculate_perplexity(log_probs, token_counts)
    
    # Calculate manually:
    total_log_prob = sum(log_probs)  # -1.0 -2.0 -1.5 -0.5 -1.0 = -6.0
    total_tokens = sum(token_counts)  # 5
    avg_log_prob = total_log_prob / total_tokens  # -6.0 / 5 = -1.2
    expected = 2 ** (-avg_log_prob)  # 2 ** 1.2 ≈ 2.297
    
    assert perplexity == pytest.approx(expected)


def test_calculate_perplexity_empty() -> None:
    """Test perplexity calculation with empty lists."""
    perplexity = calculate_perplexity([], [])
    assert perplexity == float("inf")


def test_calculate_perplexity_mismatch() -> None:
    """Test perplexity calculation with mismatched lengths."""
    log_probs = [-1.0, -2.0, -1.5]
    token_counts = [2, 2]  # Expects 4 tokens total, but only have 3
    
    with pytest.raises(ValueError, match="must match total tokens"):
        calculate_perplexity(log_probs, token_counts)


def test_lambada_in_registry() -> None:
    """Test that LAMBADA loader is registered by default."""
    registry = default_dataset_registry()

    assert "jsonl" in registry.available_sources()
    assert "hf" in registry.available_sources()
    assert "lambada" in registry.available_sources()
    assert isinstance(registry.loader_for("lambada"), LAMBADALoader)


def test_lambada_rejects_negative_max_samples() -> None:
    loader = LAMBADALoader()
    with pytest.raises(ValueError, match="non-negative"):
        loader.load(DatasetSpec(name="bad", max_samples=-1))


def test_lambada_rejects_negative_constructor_sample_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LAMBADALoader(sample_size=-1)


def test_calculate_perplexity_rejects_negative_token_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_perplexity([-1.0], [-1, 2])


def test_lambada_max_samples_zero() -> None:
    loader = LAMBADALoader(sample_size=50)
    dataset = loader.load(DatasetSpec(name="empty", max_samples=0))
    assert dataset.records == []


def test_dataset_record_cloze_format() -> None:
    """Test that LAMBADA records are in correct format for cloze tasks."""
    from benchmarks.datasets import DatasetRecord
    
    record = DatasetRecord(
        prompt="She walked to the ",
        reference="store",
    )
    
    assert not record.is_multiple_choice
    assert record.choices is None
    assert record.answer_index is None