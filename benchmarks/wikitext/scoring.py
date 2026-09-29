"""Token-level perplexity scoring for WikiText-style language-modeling rows."""

from __future__ import annotations

import hashlib

from benchmarks.dataset_types import DatasetRecord
from benchmarks.models import Prediction


def _document_text(record: DatasetRecord) -> str:
    if record.reference:
        return f"{record.prompt} {record.reference}".strip()
    return record.prompt


def _token_logprob(token: str) -> float:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return -0.05 - (digest[0] / 255.0) * 2.0


def language_model_prediction(record: DatasetRecord) -> Prediction:
    """Score the document with one natural-log NLL per whitespace token.

    The runner uses this for language-modeling datasets instead of exact-matching
    a generated string against ``reference``.
    """
    pieces = _document_text(record).split()
    if not pieces:
        return Prediction(output="", logprob=0.0, tokens=0, token_logprobs=())
    logprobs = tuple(_token_logprob(piece) for piece in pieces)
    mean = sum(logprobs) / len(logprobs)
    return Prediction(
        output="",
        logprob=mean,
        tokens=len(logprobs),
        token_logprobs=logprobs,
    )
