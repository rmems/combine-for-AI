from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from benchmarks.datasets import DatasetRecord, DatasetSpec, JsonlDatasetLoader
from benchmarks.metrics import MetricsAccumulator, entropy_from_counts
from benchmarks.models import MockModelAdapter, ModelSpec, QuantizationProfile
from benchmarks.reporting import metrics_to_row, write_csv, write_json
from benchmarks.runner import run_benchmarks
from collections import Counter


def test_lambada_sample_parsing() -> None:
    loader = JsonlDatasetLoader()
    lines = [
        json.dumps({"prompt": "She walked to the ", "reference": "store"}),
        json.dumps({"prompt": "He sat on the ", "reference": "chair"}),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.writelines(line + "\n" for line in lines)
        tmp_path = Path(tmp.name)

    try:
        spec = DatasetSpec(name="lambada", source="jsonl", path=str(tmp_path))
        loaded = loader.load(spec)
        assert len(loaded.records) == 2
        assert loaded.records[0].prompt == "She walked to the "
        assert loaded.records[0].reference == "store"
        assert not loaded.records[0].is_multiple_choice
    finally:
        tmp_path.unlink()


def test_cloze_accuracy() -> None:
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
    import random

    rng = random.Random(42)

    record = DatasetRecord(prompt="hello", reference="world")
    pred = adapter.predict(record, rng)
    assert isinstance(pred.output, str)

    correct = 0
    for _ in range(100):
        rng = random.Random(42 + _)
        pred = adapter.predict(record, rng)
        if pred.output == "world":
            correct += 1
    # Mock adapter has an 86% correct rate for fp16; allow some variance
    assert 70 < correct <= 100


def test_perplexity_shape() -> None:
    accumulator = MetricsAccumulator()
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
    import random

    rng = random.Random(42)
    for _ in range(10):
        record = DatasetRecord(prompt="hello", reference="world")
        pred = adapter.predict(record, rng)
        accumulator.add(record, pred)

    summary = accumulator.summary(total_time_s=1.0, vram_gb=14.0)
    assert summary.perplexity > 0.0
    assert isinstance(summary.perplexity, float)


def test_csv_output() -> None:
    rows = [
        {"run_id": "r1", "accuracy": 0.9, "perplexity": 12.3},
        {"run_id": "r2", "accuracy": 0.85, "perplexity": 14.1},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.csv"
        write_csv(path, rows)
        assert path.exists()
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert len(read_rows) == 2
        assert read_rows[0]["run_id"] == "r1"


def test_json_output() -> None:
    payload = {"results": [{"accuracy": 0.9}]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        write_json(path, payload)
        assert path.exists()
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["results"][0]["accuracy"] == 0.9


def test_metrics_to_row() -> None:
    from benchmarks.metrics import MetricsSummary

    summary = MetricsSummary(
        accuracy=0.9,
        perplexity=12.0,
        throughput=100.0,
        latency_ms=50.0,
        vram_gb=14.0,
        routing_entropy=1.5,
        spike_density=0.2,
    )
    row = metrics_to_row(summary)
    assert row["accuracy"] == 0.9
    assert row["perplexity"] == 12.0
    assert row["spike_density"] == 0.2


def test_final_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "benchmark.json"
        dataset_path = Path(tmpdir) / "data.jsonl"
        output_dir = Path(tmpdir) / "reports"

        with dataset_path.open("w") as f:
            f.write(json.dumps({"prompt": "hello", "reference": "world"}) + "\n")
            f.write(json.dumps({"prompt": "foo", "reference": "bar"}) + "\n")

        config = {
            "run_name": "smoke-run",
            "seed": 1337,
            "model": {"backend": "mock", "name": "toy-model", "revision": "local"},
            "quantization": ["fp16"],
            "datasets": [
                {
                    "name": "smoke",
                    "source": "jsonl",
                    "path": str(dataset_path),
                    "split": "validation",
                    "max_samples": 2,
                }
            ],
        }
        with config_path.open("w") as f:
            json.dump(config, f)

        metadata = run_benchmarks(
            config_path=config_path,
            output_dir=output_dir,
            formats=["json", "csv"],
            seed_override=42,
        )

        assert metadata.run_name == "smoke-run"
        assert metadata.seed == 42
        assert (output_dir / "json").exists()
        assert (output_dir / "csv").exists()

        json_files = list((output_dir / "json").glob("*.json"))
        csv_files = list((output_dir / "csv").glob("*.csv"))
        assert len(json_files) == 1
        assert len(csv_files) == 1

        with json_files[0].open("r") as f:
            report = json.load(f)
        assert report["run"]["run_name"] == "smoke-run"
        assert len(report["results"]) == 1


def test_entropy_from_counts() -> None:
    counts = Counter({"a": 50, "b": 50})
    entropy = entropy_from_counts(counts)
    assert entropy > 0.0
    counts_empty = Counter()
    assert entropy_from_counts(counts_empty) == 0.0


def test_cli_help() -> None:
    from scripts.run_smoke_benchmark import parse_args as smoke_parse_args

    with pytest.raises(SystemExit) as exc_info:
        smoke_parse_args(["--help"])
    assert exc_info.value.code == 0
