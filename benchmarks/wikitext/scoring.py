"""Token-level perplexity scoring for WikiText-style language-modeling rows."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks.dataset_types import DatasetRecord
from benchmarks.models import ModelAdapter, Prediction


def document_text(record: DatasetRecord) -> str:
    if record.reference is None:
        return record.prompt
    if not record.prompt or not record.reference:
        return record.prompt + record.reference
    separator = ""
    if not record.prompt[-1].isspace() and not record.reference[0].isspace():
        separator = " "
    return record.prompt + separator + record.reference


def _prediction_from_text(adapter: ModelAdapter, text: str) -> Prediction:
    logprobs = adapter.token_logprobs(text)
    if not logprobs:
        return Prediction(output="", logprob=0.0, tokens=0, token_logprobs=())
    mean = sum(logprobs) / len(logprobs)
    return Prediction(
        output="",
        logprob=mean,
        tokens=len(logprobs),
        token_logprobs=logprobs,
    )


def language_model_prediction(adapter: ModelAdapter, record: DatasetRecord) -> Prediction:
    """Score the document with the adapter's per-token natural-log likelihoods.

    The runner uses this for language-modeling datasets instead of exact-matching
    a generated string against ``reference``.
    """
    return _prediction_from_text(adapter, document_text(record))


def language_model_corpus_prediction(
    adapter: ModelAdapter, records: Sequence[DatasetRecord]
) -> Prediction:
    """Score records as one continuous corpus, preserving their boundaries."""
    return _prediction_from_text(
        adapter, "".join(document_text(record) for record in records)
    )
