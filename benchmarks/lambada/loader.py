from __future__ import annotations

from typing import Any

from benchmarks.dataset_jsonl import canonical_record
from benchmarks.dataset_support import (
    CATALOG,
    MappedDatasetLoader,
    register_row_mapper,
    sample_path_for,
)
from benchmarks.dataset_types import DatasetRecord, DatasetSpec

# LAMBADA is last-word cloze: the final whitespace-delimited token is the target.


def map_lambada_row(row: dict[str, Any]) -> DatasetRecord | None:
    """Map a LAMBADA row to a cloze pair.

    Field mapping follows EleutherAI lm-evaluation-harness ``lambada_openai``:
    Hub rows expose ``text``; the last whitespace-delimited token is the target.
    Canonical ``prompt`` / ``reference`` JSONL is passed through unchanged.
    """
    record = canonical_record(row)
    if record is not None:
        return record

    text = row.get("text")
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    parts = stripped.rsplit(None, 1)
    if len(parts) != 2:
        return None
    prefix, target = parts
    target = target.strip(".,!?;:\"'")
    if not target:
        return None
    return DatasetRecord(prompt=f"{prefix} ", reference=target)


register_row_mapper("lambada", map_lambada_row)


def load_lambada_sample(
    sample_size: int | None = None, seed: int = 42
) -> list[DatasetRecord]:
    """Load cloze records from the bundled LAMBADA sample JSONL."""
    _ = seed  # kept for call-site compatibility with the old synthetic helper
    entry = CATALOG["lambada"]
    spec = DatasetSpec(
        name="lambada",
        source="lambada",
        path=str(sample_path_for(entry)),
        max_samples=sample_size,
    )
    return LAMBADALoader().load(spec).records


class LAMBADALoader(MappedDatasetLoader):
    """Load LAMBADA (lm-eval ``lambada_openai``) as last-word cloze pairs."""

    catalog_name = "lambada"

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        return map_lambada_row(row)


def calculate_cloze_accuracy(
    predictions: list[str], references: list[str]
) -> float:
    """Calculate cloze accuracy; empty+empty is 0.0, length mismatch raises."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0

    correct = sum(
        1 for pred, ref in zip(predictions, references, strict=True) if pred == ref
    )
    return correct / len(predictions)


def calculate_perplexity(
    log_probabilities: list[float],
    token_counts: list[int],
) -> float:
    """Calculate perplexity from token log-probs; validate counts before fallback."""
    if any(count < 0 for count in token_counts):
        raise ValueError("token counts must be non-negative")
    total_tokens = sum(token_counts)
    if len(log_probabilities) != total_tokens:
        raise ValueError(
            f"Log probabilities ({len(log_probabilities)}) must match "
            f"total tokens ({total_tokens})"
        )
    if total_tokens == 0:
        return float("inf")

    avg_log_prob = sum(log_probabilities) / total_tokens
    return float(2 ** (-avg_log_prob))
