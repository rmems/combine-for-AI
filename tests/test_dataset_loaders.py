from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.dataset_support import (
    CATALOG,
    HuggingFaceDatasetLoader,
    JsonlDatasetLoader,
    sample_path_for,
    validate_loaded,
)
from benchmarks.dataset_types import DatasetRecord, LoadedDataset, TaskKind
from benchmarks.datasets import DatasetSpec, default_dataset_registry
from benchmarks.runner import load_datasets


NAMED_DATASETS = (
    "lambada",
    "hellaswag",
    "wikitext2",
    "gsm8k",
    "piqa",
    "arc_easy",
)


def test_registry_exposes_every_named_dataset() -> None:
    registry = default_dataset_registry()
    names = registry.named_datasets()
    assert names == NAMED_DATASETS
    sources = set(registry.available_sources())
    assert {"jsonl", "hf", "wikitext", "arc-easy"}.issubset(sources)
    for name, loader in registry.iter_named_datasets():
        spec = DatasetSpec(
            name=name,
            source=name,
            path=str(sample_path_for(CATALOG[name])),
        )
        loaded = loader.load(spec)
        assert loaded.records
        assert all(record.prompt.strip() for record in loaded.records)


def test_runner_load_datasets_iterates_registered_samples(tmp_path: Path) -> None:
    registry = default_dataset_registry()
    config = {
        "datasets": [
            {
                "name": name,
                "source": name,
                "path": str(sample_path_for(CATALOG[name])),
                "max_samples": 2,
            }
            for name in registry.named_datasets()
        ]
    }
    loaded = load_datasets(config, tmp_path)
    assert [item.spec.name for item in loaded] == list(NAMED_DATASETS)
    assert all(item.records for item in loaded)


def test_sample_jsonl_files_work_offline() -> None:
    loader = JsonlDatasetLoader()
    for name in NAMED_DATASETS:
        spec = DatasetSpec(
            name=name,
            source="jsonl",
            path=str(sample_path_for(CATALOG[name])),
        )
        dataset = loader.load(spec)
        assert dataset.metadata["source"] == "jsonl"
        assert len(dataset.records) >= 1


def test_jsonl_missing_path() -> None:
    with pytest.raises(ValueError, match="missing a path"):
        JsonlDatasetLoader().load(DatasetSpec(name="demo", source="jsonl"))


def test_jsonl_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError, match="not found"):
        JsonlDatasetLoader().load(
            DatasetSpec(name="demo", source="jsonl", path=str(missing))
        )


def test_jsonl_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid json"):
        JsonlDatasetLoader().load(
            DatasetSpec(name="demo", source="jsonl", path=str(path))
        )


def test_jsonl_missing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "noprompt.jsonl"
    path.write_text(json.dumps({"reference": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        JsonlDatasetLoader().load(
            DatasetSpec(name="demo", source="jsonl", path=str(path))
        )


def test_validation_rejects_too_few_samples() -> None:
    spec = DatasetSpec(
        name="lambada",
        source="lambada",
        path=str(sample_path_for(CATALOG["lambada"])),
        min_samples=50,
    )
    with pytest.raises(ValueError, match="at least 50"):
        default_dataset_registry().loader_for("lambada").load(spec)


def test_validation_rejects_bad_answer_index() -> None:
    loaded = LoadedDataset(
        spec=DatasetSpec(name="hellaswag", source="hellaswag"),
        records=[
            DatasetRecord(
                prompt="ctx",
                choices=["a", "b"],
                answer_index=3,
            )
        ],
        metadata={},
    )
    with pytest.raises(ValueError, match="out of range"):
        validate_loaded(loaded, CATALOG["hellaswag"])


def test_validation_rejects_empty_prompt() -> None:
    loaded = LoadedDataset(
        spec=DatasetSpec(name="lambada", source="lambada"),
        records=[DatasetRecord(prompt="   ", reference="store")],
        metadata={},
    )
    with pytest.raises(ValueError, match="empty prompt"):
        validate_loaded(loaded, CATALOG["lambada"])


def test_hf_cache_avoids_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        return [{"text": "She walked to the store"}]

    monkeypatch.setattr("benchmarks.dataset_support.hf_load_dataset", fake_load)
    spec = DatasetSpec(
        name="lambada",
        source="hf",
        hf_id="EleutherAI/lambada_openai",
        cache_dir=str(tmp_path),
    )
    loader = HuggingFaceDatasetLoader()
    first = loader.load(spec)
    second = loader.load(spec)
    assert calls["n"] == 1
    assert first.records[0].reference == "store"
    assert first.metadata["source"] == "hf"
    assert second.metadata["source"] == "hf_cache"
    assert second.records == first.records


def test_hf_falls_back_to_sample_when_datasets_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmarks.dataset_support.hf_load_dataset", None)
    dataset = HuggingFaceDatasetLoader().load(
        DatasetSpec(name="piqa", source="hf")
    )
    assert dataset.metadata["source"] == "jsonl"
    assert dataset.metadata.get("fallback") is True
    assert len(dataset.records) == 2


def test_hf_without_fallback_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("benchmarks.dataset_support.hf_load_dataset", None)
    with pytest.raises(ImportError, match="datasets is required"):
        HuggingFaceDatasetLoader().load(
            DatasetSpec(
                name="custom",
                source="hf",
                hf_id="org/name",
                path=str(tmp_path / "missing.jsonl"),
            )
        )


def test_dataset_spec_from_dict_reads_optional_fields() -> None:
    spec = DatasetSpec.from_dict(
        {
            "name": "lambada",
            "source": "hf",
            "hf_id": "EleutherAI/lambada_openai",
            "min_samples": 2,
            "max_samples": 8,
            "cache_dir": "/tmp/cache",
        }
    )
    assert spec.min_samples == 2
    assert spec.max_samples == 8
    assert spec.cache_dir == "/tmp/cache"
    assert spec.hf_id == "EleutherAI/lambada_openai"


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="not registered"):
        default_dataset_registry().loader_for("nope")


def test_catalog_task_kinds_cover_all_named_datasets() -> None:
    expected = {
        "lambada": TaskKind.CLOZE,
        "hellaswag": TaskKind.MULTIPLE_CHOICE,
        "wikitext2": TaskKind.LANGUAGE_MODELING,
        "gsm8k": TaskKind.MATH,
        "piqa": TaskKind.MULTIPLE_CHOICE,
        "arc_easy": TaskKind.MULTIPLE_CHOICE,
    }
    for name, task in expected.items():
        assert CATALOG[name].task is task
