"""Token-level perplexity scoring for WikiText-style language-modeling rows."""

from __future__ import annotations

from benchmarks.dataset_types import DatasetRecord
from benchmarks.models import ModelAdapter, Prediction


def document_text(record: DatasetRecord) -> str:
    if record.reference:
        return f"{record.prompt} {record.reference}".strip()
    return record.prompt


def language_model_prediction(adapter: ModelAdapter, record: DatasetRecord) -> Prediction:
    """Score the document with the adapter's per-token natural-log likelihoods.

    The runner uses this for language-modeling datasets instead of exact-matching
    a generated string against ``reference``.
    """
    logprobs = adapter.token_logprobs(document_text(record))
    if not logprobs:
        return Prediction(output="", logprob=0.0, tokens=0, token_logprobs=())
    mean = sum(logprobs) / len(logprobs)
    return Prediction(
        output="",
        logprob=mean,
        tokens=len(logprobs),
        token_logprobs=logprobs,
    )
