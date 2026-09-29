from __future__ import annotations

import math

from benchmarks.dataset_support import CATALOG, HuggingFaceDatasetLoader, sample_path_for
from benchmarks.datasets import DatasetSpec
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import Prediction
from benchmarks.wikitext.loader import (
    WikiText2Loader,
    concatenate_documents,
    map_wikitext_row,
)
from benchmarks.wikitext.scoring import language_model_prediction


def test_wikitext_runner_scores_token_nll_not_exact_match() -> None:
    record = map_wikitext_row({"text": "The capital city of France is Paris."})
    assert record is not None
    assert record.reference is None
    prediction = language_model_prediction(record)
    assert prediction.token_logprobs is not None
    assert len(prediction.token_logprobs) == len(record.prompt.split())
    missed = Prediction(output="not the document", logprob=-9.0, tokens=1)
    exact = MetricsAccumulator()
    exact.add(record, missed)
    assert exact.summary(1.0, 1.0).accuracy == 0.0
    scored = MetricsAccumulator()
    scored.add(record, prediction, language_modeling=True)
    assert prediction.token_logprobs is not None
    expected = math.exp(-sum(prediction.token_logprobs) / len(prediction.token_logprobs))
    summary = scored.summary(1.0, 1.0)
    assert summary.accuracy == 0.0
    assert summary.perplexity == expected


def test_wikitext_sample_jsonl() -> None:
    dataset = WikiText2Loader().load(
        DatasetSpec(
            name="wikitext2",
            source="wikitext2",
            path=str(sample_path_for(CATALOG["wikitext2"])),
        )
    )
    assert len(dataset.records) == 2
    assert "Apollo" in dataset.records[0].prompt
    joined = concatenate_documents(dataset.records)
    assert "France" in joined
    assert dataset.records[1].reference == "Paris."


def test_map_wikitext_native_preserves_raw_text() -> None:
    blank = map_wikitext_row({"text": " \n"})
    assert blank is not None
    assert blank.prompt == " \n"
    assert blank.reference is None
    record = map_wikitext_row({"text": "  The capital city of France is Paris.  "})
    assert record is not None
    assert record.prompt == "  The capital city of France is Paris.  "
    assert record.reference is None


def test_wikitext_hf_rows_preserve_blank_separators(tmp_path, monkeypatch) -> None:
    rows = [
        {"text": ""},
        {"text": "In 1969, the Apollo 11 mission landed on the moon."},
        {"text": "   "},
    ]
    monkeypatch.setattr(
        "benchmarks.dataset_mapped_cache.hf_load_dataset",
        lambda *args, **kwargs: rows,
    )
    dataset = HuggingFaceDatasetLoader().load(
        DatasetSpec(
            name="wikitext2",
            source="hf",
            hf_id="Salesforce/wikitext",
            hf_subset="wikitext-2-raw-v1",
            cache_dir=str(tmp_path),
        )
    )
    assert len(dataset.records) == 3
    assert dataset.metadata["skipped"] == 0
    assert concatenate_documents(dataset.records) == (
        "In 1969, the Apollo 11 mission landed on the moon.   "
    )
