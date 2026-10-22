from __future__ import annotations

import math

import benchmarks.wikitext.scoring as wikitext_scoring
from benchmarks.dataset_support import CATALOG, HuggingFaceDatasetLoader, sample_path_for
from benchmarks.dataset_types import DatasetRecord, LoadedDataset
from benchmarks.datasets import DatasetSpec, default_dataset_registry
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import MockModelAdapter, ModelSpec, QuantizationProfile
from benchmarks.runner import _EvalContext, _evaluate_dataset, _language_modeling
from benchmarks.wikitext.loader import (
    WikiText2Loader,
    concatenate_documents,
    map_wikitext_row,
)
from benchmarks.wikitext.scoring import document_text, language_model_prediction


def test_wikitext_runner_scores_token_nll_not_exact_match() -> None:
    record = map_wikitext_row({"text": "The capital city of France is Paris."})
    assert record is not None
    assert record.reference is None
    fp16 = QuantizationProfile(
        name="fp16",
        precision="fp16",
        format="baseline",
        bits=16,
        supported=True,
        speed_tps=1000.0,
        vram_gb=14.0,
        notes="",
    )
    awq = QuantizationProfile(
        name="awq",
        precision="int4",
        format="awq",
        bits=4,
        supported=True,
        speed_tps=2000.0,
        vram_gb=6.0,
        notes="",
    )
    fp16_prediction = language_model_prediction(
        MockModelAdapter(ModelSpec(backend="mock", name="toy"), fp16), record
    )
    awq_prediction = language_model_prediction(
        MockModelAdapter(ModelSpec(backend="mock", name="toy"), awq), record
    )
    assert fp16_prediction.token_logprobs is not None
    assert awq_prediction.token_logprobs is not None
    assert len(fp16_prediction.token_logprobs) == len(record.prompt.split())
    assert fp16_prediction.token_logprobs != awq_prediction.token_logprobs
    scored = MetricsAccumulator()
    scored.add(record, fp16_prediction, language_modeling=True)
    expected = math.exp(
        -sum(fp16_prediction.token_logprobs) / len(fp16_prediction.token_logprobs)
    )
    summary = scored.summary(1.0, 1.0)
    assert math.isclose(summary.accuracy, 0.0)
    assert math.isclose(summary.perplexity, expected)


def test_document_text_preserves_existing_whitespace_boundary() -> None:
    from benchmarks.dataset_types import DatasetRecord

    assert document_text(DatasetRecord(prompt="hello ", reference="world")) == "hello world"
    assert document_text(DatasetRecord(prompt="hello", reference="world")) == "hello world"
    assert document_text(DatasetRecord(prompt=" \n")) == " \n"


def test_language_model_corpus_is_scored_as_one_continuous_stream() -> None:
    scorer = getattr(wikitext_scoring, "language_model_corpus_prediction", None)
    assert scorer is not None

    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls = []

        def token_logprobs(self, text):
            self.calls.append(text)
            return (-1.0,) if text.strip() else ()

    adapter = RecordingAdapter()
    prediction = scorer(
        adapter,
        [
            DatasetRecord(prompt="first"),
            DatasetRecord(prompt=" \n"),
            DatasetRecord(prompt="second"),
        ],
    )
    assert adapter.calls == ["first \nsecond"]
    assert prediction.tokens == 1


def test_runner_scores_wikitext_as_one_corpus(monkeypatch) -> None:
    calls = []

    def score_corpus(adapter, records):
        from benchmarks.models import Prediction

        calls.append(records)
        return Prediction(
            output="", logprob=-1.0, tokens=2, token_logprobs=(-1.0, -1.0)
        )

    monkeypatch.setattr("benchmarks.runner.language_model_corpus_prediction", score_corpus)
    profile = QuantizationProfile(
        name="fp16",
        precision="fp16",
        format="baseline",
        bits=16,
        supported=True,
        speed_tps=1000.0,
        vram_gb=14.0,
        notes="",
    )
    adapter = MockModelAdapter(ModelSpec(backend="mock", name="toy"), profile)
    records = [DatasetRecord(prompt="first"), DatasetRecord(prompt="second")]
    result = _evaluate_dataset(
        adapter,
        profile,
        LoadedDataset(
            spec=DatasetSpec(name="wikitext2", source="wikitext2"),
            records=records,
            metadata={"task": "language_modeling"},
        ),
        _EvalContext(seed=1, telemetry=None),  # type: ignore[arg-type]
    )

    assert calls == [records]
    assert result.sample_count == 2
    assert math.isclose(result.metrics.perplexity, math.e)


def test_empty_language_model_prediction_has_undefined_perplexity() -> None:
    from benchmarks.dataset_types import DatasetRecord

    scored = MetricsAccumulator()
    scored.add(
        DatasetRecord(prompt=" \n"),
        language_model_prediction(
            MockModelAdapter(
                ModelSpec(backend="mock", name="toy"),
                QuantizationProfile(
                    name="fp16",
                    precision="fp16",
                    format="baseline",
                    bits=16,
                    supported=True,
                    speed_tps=1000.0,
                    vram_gb=14.0,
                    notes="",
                ),
            ),
            DatasetRecord(prompt=" \n"),
        ),
        language_modeling=True,
    )
    assert math.isnan(scored.summary(0.0, 1.0).perplexity)


def test_custom_named_wikitext_source_stays_language_modeling() -> None:
    loaded = default_dataset_registry().loader_for("wikitext2").load(
        DatasetSpec(
            name="wiki_eval",
            source="wikitext2",
            path=str(sample_path_for(CATALOG["wikitext2"])),
        )
    )
    assert loaded.metadata["task"] == "language_modeling"
    assert _language_modeling(loaded) is True


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


def test_mock_token_logprobs_depend_on_document_text() -> None:
    profile = QuantizationProfile(
        name="fp16",
        precision="fp16",
        format="baseline",
        bits=16,
        supported=True,
        speed_tps=1000.0,
        vram_gb=14.0,
        notes="",
    )
    adapter = MockModelAdapter(ModelSpec(backend="mock", name="toy"), profile)
    paris = adapter.token_logprobs("Paris is in France")
    other = adapter.token_logprobs("a completely different document")
    assert len(paris) == len(other) == 4
    assert paris != other
    assert all(value < 0 for value in paris)
    assert all(value < 0 for value in other)


def test_generic_hf_wikitext_keeps_blank_separators(tmp_path, monkeypatch) -> None:
    from benchmarks.datasets import HuggingFaceDatasetLoader as GenericHfLoader

    rows = [{"text": "first"}, {"text": ""}, {"text": "second"}]

    def load_dataset(path, name=None, *, split, revision):
        assert path == "Salesforce/wikitext"
        assert name == "wikitext-2-raw-v1"
        assert revision == "c" * 40
        return rows

    monkeypatch.setattr("benchmarks.dataset_cache_hf.hf_load_dataset", load_dataset)
    monkeypatch.setattr(
        "benchmarks.dataset_cache_hf.resolve_huggingface_revision",
        lambda spec: "c" * 40,
    )
    loaded = GenericHfLoader().load(
        DatasetSpec(
            name="wikitext2",
            source="hf",
            hf_id="Salesforce/wikitext",
            hf_subset="wikitext-2-raw-v1",
            cache_root=str(tmp_path),
            cache_mode="online",
        )
    )
    assert [record.prompt for record in loaded.records] == ["first", "", "second"]
    assert loaded.metadata["resolved_revision"] == "c" * 40
    assert loaded.metadata["source_uri"].endswith("@" + "c" * 40)
