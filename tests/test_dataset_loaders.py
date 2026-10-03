from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.dataset_mapped_cache import (
    MappedHfSpec,
    cache_sidecar_path,
    normalized_cache_path,
)
from benchmarks.dataset_cache_models import provenance_from_metadata
from benchmarks.dataset_support import (
    CATALOG,
    HuggingFaceDatasetLoader,
    JsonlDatasetLoader,
    ROW_MAPPERS,
    _register_entry,
    load_with_fallback,
    mapper_for,
    sample_path_for,
    validate_loaded,
)
from benchmarks.dataset_types import (
    CatalogEntry,
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
    TaskKind,
)
from benchmarks.datasets import default_dataset_registry
from benchmarks.runner import load_datasets


NAMED_DATASETS = (
    "lambada",
    "hellaswag",
    "wikitext2",
    "gsm8k",
    "piqa",
    "arc_easy",
)


@pytest.fixture(autouse=True)
def _disable_live_hub_revision_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmarks.dataset_mapped_cache._remote_commit", lambda _spec: None
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


def test_jsonl_mapper_type_errors_include_line_and_path(tmp_path: Path) -> None:
    path = tmp_path / "bad-shape.jsonl"
    path.write_text('{"prompt": "question"}\n', encoding="utf-8")

    def bad_mapper(_row):
        raise TypeError("bad field shape")

    with pytest.raises(
        ValueError, match=rf"bad field shape on line 1 in {path}"
    ):
        from benchmarks.dataset_jsonl import records_from_jsonl

        records_from_jsonl(path, bad_mapper, None)


def test_canonical_choices_reject_non_string_values() -> None:
    from benchmarks.dataset_jsonl import canonical_record

    with pytest.raises(ValueError, match="list of strings"):
        canonical_record(
            {"prompt": "question", "choices": [1, "two"], "answer_index": 0}
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


def test_explicit_min_samples_is_not_bypassed_by_zero_max_samples() -> None:
    loaded = LoadedDataset(
        spec=DatasetSpec(name="demo", min_samples=1, max_samples=0),
        records=[],
        metadata={},
    )
    with pytest.raises(ValueError, match="at least 1"):
        validate_loaded(loaded, None)


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

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
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
    sidecar = cache_sidecar_path(Path(first.metadata["cached_path"]))
    assert sidecar.exists()


def test_prefer_cache_hit_does_not_probe_hub(tmp_path, monkeypatch) -> None:
    from benchmarks import dataset_mapped_cache as cache

    spec = MappedHfSpec(
        name="piqa",
        hf_id="ybisk/piqa",
        hf_subset=None,
        split="validation",
        revision="main",
        cache_dir=tmp_path,
        task=TaskKind.MULTIPLE_CHOICE,
    )
    path = cache.normalized_cache_path(spec)
    cache._persist_mapped_cache(
        path,
        [DatasetRecord(prompt="keep", choices=["a", "b"], answer_index=0)],
        spec,
        "a" * 40,
    )

    def unexpected_probe(_spec):
        raise AssertionError("prefer-cache must use a validated hit before probing")

    monkeypatch.setattr(cache, "_remote_commit", unexpected_probe)
    records, metadata = cache.records_from_hf(
        spec,
        lambda row: DatasetRecord(prompt=str(row)),
        refresh=False,
    )
    assert records[0].prompt == "keep"
    assert metadata["source"] == "hf_cache"


def test_invalid_mapped_rows_are_not_persisted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.dataset_mapped_cache.hf_load_dataset",
        lambda *args, **kwargs: [
            {"goal": "bad", "sol1": "only", "sol2": "", "label": 0}
        ],
    )
    loaded = default_dataset_registry().loader_for("piqa").load(
        DatasetSpec(name="piqa", source="piqa", cache_dir=str(tmp_path))
    )
    assert loaded.metadata["source"] == "jsonl"
    normalized = list((tmp_path / "normalized").rglob("*.jsonl"))
    assert normalized == []


def test_invalid_mapped_cache_is_refetched(tmp_path, monkeypatch) -> None:
    from benchmarks import dataset_mapped_cache as cache
    from benchmarks.piqa.loader import map_piqa_row

    spec = MappedHfSpec(
        name="piqa",
        hf_id="ybisk/piqa",
        hf_subset=None,
        split="validation",
        cache_dir=tmp_path,
        task=TaskKind.MULTIPLE_CHOICE,
    )
    path = cache.normalized_cache_path(spec)
    cache._persist_mapped_cache(
        path,
        [DatasetRecord(prompt="poison", choices=["only"], answer_index=0)],
        spec,
        None,
    )
    monkeypatch.setattr(
        cache,
        "hf_load_dataset",
        lambda *args, **kwargs: [
            {"goal": "good", "sol1": "one", "sol2": "two", "label": 1}
        ],
    )

    records, metadata = cache.records_from_hf(spec, map_piqa_row)

    assert metadata["source"] == "hf"
    assert records[0].prompt == "good"
    assert records[0].answer_index == 1


def test_failed_revision_probe_is_treated_as_unknown(monkeypatch) -> None:
    from benchmarks import dataset_mapped_cache as cache

    monkeypatch.setattr(
        cache,
        "resolve_dataset_revision",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("hub unavailable")),
    )
    assert cache._remote_commit(
        MappedHfSpec(
            name="custom",
            hf_id="org/data",
            hf_subset=None,
            split="validation",
            revision="main",
            cache_dir=Path("cache"),
        )
    ) is None


def test_hf_cache_rejects_tampered_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        return [{"text": "She walked to the store"}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    spec = DatasetSpec(
        name="lambada",
        source="hf",
        hf_id="EleutherAI/lambada_openai",
        cache_dir=str(tmp_path),
    )
    loader = HuggingFaceDatasetLoader()
    first = loader.load(spec)
    sidecar = cache_sidecar_path(Path(first.metadata["cached_path"]))
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["digest"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    second = loader.load(spec)
    assert calls["n"] == 2
    assert second.metadata["source"] == "hf"


def test_hf_cache_rejects_tampered_jsonl_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        return [{"text": "She walked to the store"}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    spec = DatasetSpec(
        name="lambada",
        source="hf",
        hf_id="EleutherAI/lambada_openai",
        cache_dir=str(tmp_path),
    )
    loader = HuggingFaceDatasetLoader()
    first = loader.load(spec)
    path = Path(first.metadata["cached_path"])
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    payload["reference"] = "tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    second = loader.load(spec)
    assert calls["n"] == 2
    assert second.metadata["source"] == "hf"
    assert second.records[0].reference == "store"


def test_hf_falls_back_to_sample_when_datasets_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", None)
    dataset = HuggingFaceDatasetLoader().load(
        DatasetSpec(name="piqa", source="hf")
    )
    assert dataset.metadata["source"] == "jsonl"
    assert dataset.metadata.get("fallback") is True
    assert len(dataset.records) == 2


def test_hf_without_fallback_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", None)
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
            "cache_root": "/tmp/cache-root",
            "cache_mode": "prefer-cache",
            "revision": "main",
            "upstream_license": "cc-by-4.0",
        }
    )
    assert spec.min_samples == 2
    assert spec.max_samples == 8
    assert spec.cache_dir == "/tmp/cache"
    assert spec.cache_root == "/tmp/cache-root"
    assert spec.cache_mode == "prefer-cache"
    assert spec.revision == "main"
    assert spec.upstream_license == "cc-by-4.0"
    assert spec.hf_id == "EleutherAI/lambada_openai"


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="not registered"):
        default_dataset_registry().loader_for("nope")


def test_row_mappers_cover_catalog_aliases() -> None:
    default_dataset_registry()
    ROW_MAPPERS.pop("wikitext", None)
    ROW_MAPPERS.pop("arc-easy", None)
    assert mapper_for("wikitext") is mapper_for("wikitext2")
    assert mapper_for("arc-easy") is mapper_for("arc_easy")


def test_alias_cannot_collide_with_another_primary_name() -> None:
    with pytest.raises(ValueError, match="collides with primary catalog name"):
        _register_entry(
            CatalogEntry(
                name="__tmp_collision__",
                task=TaskKind.GENERIC,
                hf_id="x",
                sample_relpath="x.jsonl",
                aliases=("lambada",),
            )
        )
    assert "__tmp_collision__" not in CATALOG


def test_normalized_cache_path_uses_full_sha256(tmp_path: Path) -> None:
    path = normalized_cache_path(
        MappedHfSpec(
            name="lambada",
            hf_id="org/name",
            hf_subset=None,
            split="test",
            cache_dir=tmp_path,
        )
    )
    assert len(path.stem) == 64
    assert path.suffix == ".jsonl"


def test_min_samples_rejects_bool() -> None:
    with pytest.raises(ValueError, match="min_samples must be an integer"):
        DatasetSpec(name="demo", min_samples=True)  # type: ignore[arg-type]


def test_generic_jsonl_enforces_min_samples(tmp_path: Path) -> None:
    from benchmarks.datasets import JsonlDatasetLoader as GenericJsonl

    path = tmp_path / "row.jsonl"
    path.write_text(json.dumps({"prompt": "q", "reference": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2"):
        GenericJsonl().load(
            DatasetSpec(
                name="demo", source="jsonl", path=str(path), min_samples=2
            )
        )


def test_relative_cache_dir_resolves_against_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    loaded = load_datasets(
        {
            "datasets": [
                {
                    "name": "lambada",
                    "source": "lambada",
                    "path": str(sample_path_for(CATALOG["lambada"])),
                    "cache_dir": "cache",
                }
            ]
        },
        config_dir,
    )
    assert loaded[0].spec.cache_dir == str((config_dir / "cache").resolve())


def test_fallback_metadata_reaches_provenance() -> None:
    from benchmarks.dataset_cache_models import provenance_from_metadata

    provenance = provenance_from_metadata(
        {"source": "jsonl", "fallback": True, "hf_error": "offline", "path": "/tmp/x"}
    )
    assert provenance["dataset_load_source"] == "jsonl"
    assert provenance["dataset_fallback"] is True
    assert provenance["dataset_hf_error"] == "offline"


def test_named_jsonl_preserves_full_provenance(tmp_path: Path) -> None:
    path = tmp_path / "lambada.jsonl"
    path.write_text(
        json.dumps({"prompt": "The answer is ", "reference": "yes"}) + "\n",
        encoding="utf-8",
    )
    loaded = default_dataset_registry().loader_for("lambada").load(
        DatasetSpec(
            name="lambada",
            source="lambada",
            path=str(path),
            upstream_license="MIT",
        )
    )
    assert loaded.metadata["dataset_upstream_license"] == "MIT"
    assert loaded.metadata["source_uri"] == path.resolve().as_uri()
    assert loaded.metadata["row_count"] == 1


def test_generic_jsonl_omitted_split_defaults_to_validation(tmp_path: Path) -> None:
    from benchmarks.datasets import JsonlDatasetLoader as GenericJsonl

    path = tmp_path / "row.jsonl"
    path.write_text(json.dumps({"prompt": "q", "reference": "a"}) + "\n", encoding="utf-8")
    loaded = GenericJsonl().load(
        DatasetSpec(name="demo", source="jsonl", path=str(path))
    )
    assert loaded.spec.split == "validation"


def test_omitted_split_uses_catalog_default() -> None:
    from benchmarks.dataset_support import catalog_entry, resolve_hf_split

    spec = DatasetSpec(name="gsm8k", source="gsm8k")
    assert spec.split is None
    assert resolve_hf_split(spec, catalog_entry("gsm8k")) == "test"


def test_explicit_validation_split_is_preserved() -> None:
    from benchmarks.dataset_support import catalog_entry, resolve_hf_split

    spec = DatasetSpec(name="gsm8k", source="gsm8k", split="validation")
    assert resolve_hf_split(spec, catalog_entry("gsm8k")) == "validation"


def test_catalog_hub_ids_match_case_families() -> None:
    assert CATALOG["piqa"].hf_id == "ybisk/piqa"
    assert CATALOG["wikitext2"].hf_id == "Salesforce/wikitext"


def test_runner_named_source_without_path_fetches_hf(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_hf(spec, map_row, *, allow_fetch=True, refresh=False):
        calls["n"] += 1
        assert allow_fetch is True
        assert refresh is False
        return [DatasetRecord(prompt="hub ", reference="answer")], {"source": "hf"}

    monkeypatch.setattr("benchmarks.dataset_support.records_from_hf", fake_hf)
    loaded = load_datasets(
        {
            "datasets": [
                {
                    "name": "lambada",
                    "source": "lambada",
                    "cache_dir": str(tmp_path),
                }
            ]
        },
        tmp_path,
    )
    assert calls["n"] == 1
    assert loaded[0].metadata["source"] == "hf"


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


@pytest.mark.parametrize("answer_index", [True, False, 1.2, 1.0, [], {}, "", "1.2", "bad"])
def test_canonical_answer_index_rejects_invalid_values(answer_index) -> None:
    from benchmarks.dataset_jsonl import canonical_record

    with pytest.raises(ValueError, match="answer_index"):
        canonical_record({"prompt": "Question", "answer_index": answer_index})


@pytest.mark.parametrize("choices", ["ab", b"ab", 1, {"0": "first", "1": "second"}])
def test_canonical_choices_rejects_non_lists(choices) -> None:
    from benchmarks.dataset_jsonl import canonical_record

    with pytest.raises(ValueError, match="choices must be a list"):
        canonical_record({"prompt": "Question", "choices": choices})


def test_canonical_choices_keeps_list_items() -> None:
    from benchmarks.dataset_jsonl import canonical_record

    record = canonical_record({"prompt": "Question", "choices": ["yes", "no"]})
    assert record is not None
    assert record.choices == ["yes", "no"]


@pytest.mark.parametrize("answer_index", [0, 1, "0", " 1 "])
def test_canonical_answer_index_accepts_integers(answer_index) -> None:
    from benchmarks.dataset_jsonl import canonical_record

    record = canonical_record({"prompt": "Question", "answer_index": answer_index})
    assert record is not None
    assert record.answer_index == int(answer_index)


def _hub_stub_for(name: str) -> tuple[list[DatasetRecord], dict]:
    task = CATALOG[name].task
    if task is TaskKind.MULTIPLE_CHOICE:
        record = DatasetRecord(prompt="hub", choices=["a", "b"], answer_index=0)
    elif task is TaskKind.LANGUAGE_MODELING:
        record = DatasetRecord(prompt="hub document")
    else:
        record = DatasetRecord(prompt="hub ", reference="answer")
    return [record], {"source": "hf", "hf_id": CATALOG[name].hf_id}


@pytest.mark.parametrize("name", NAMED_DATASETS)
def test_named_loader_without_path_fetches_hf(name, monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def fake_hf(*args, **kwargs):
        calls["n"] += 1
        return _hub_stub_for(name)

    monkeypatch.setattr("benchmarks.dataset_support.records_from_hf", fake_hf)
    loaded = default_dataset_registry().loader_for(name).load(
        DatasetSpec(name=name, source=name, cache_dir=str(tmp_path))
    )
    assert calls["n"] == 1
    assert loaded.metadata["source"] == "hf"
    assert loaded.records


@pytest.mark.parametrize("name", NAMED_DATASETS)
def test_named_loader_offline_miss_skips_hf(name, monkeypatch, tmp_path) -> None:
    def unexpected_hf(*args, **kwargs):
        raise AssertionError("offline mode must not call Hugging Face")

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", unexpected_hf)
    loaded = default_dataset_registry().loader_for(name).load(
        DatasetSpec(
            name=name,
            source=name,
            cache_mode="offline",
            cache_dir=str(tmp_path),
        )
    )
    assert loaded.records
    assert loaded.metadata["source"] == "jsonl"
    assert loaded.metadata.get("fallback") is True
    assert "offline cache miss" in loaded.metadata["hf_error"]


def test_named_loader_offline_hit_skips_hf(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        return [{"text": "She walked to the store"}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    spec = DatasetSpec(
        name="lambada",
        source="lambada",
        cache_dir=str(tmp_path),
    )
    first = default_dataset_registry().loader_for("lambada").load(spec)
    assert calls["n"] == 1
    assert first.metadata["source"] == "hf"

    def unexpected_hf(*args, **kwargs):
        raise AssertionError("offline mode must not refetch Hugging Face")

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", unexpected_hf)
    second = default_dataset_registry().loader_for("lambada").load(
        DatasetSpec(
            name="lambada",
            source="lambada",
            cache_mode="offline",
            cache_dir=str(tmp_path),
        )
    )
    assert second.metadata["source"] == "hf_cache"
    assert second.records == first.records


@pytest.mark.parametrize("name", NAMED_DATASETS)
def test_named_loader_without_path_falls_back_to_sample(name, monkeypatch) -> None:
    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", None)
    loaded = default_dataset_registry().loader_for(name).load(
        DatasetSpec(name=name, source=name)
    )
    assert loaded.records
    assert loaded.metadata["source"] == "jsonl"
    assert loaded.metadata.get("fallback") is True


@pytest.mark.parametrize("label", ["bad", []])
def test_hf_mapping_errors_fall_back_to_jsonl(tmp_path, monkeypatch, label) -> None:
    monkeypatch.setattr(
        "benchmarks.dataset_mapped_cache.hf_load_dataset",
        lambda *args, **kwargs: [{"goal": "Question", "sol1": "a", "sol2": "b", "label": label}],
    )
    loaded = HuggingFaceDatasetLoader().load(
        DatasetSpec(name="piqa", source="hf", cache_dir=str(tmp_path))
    )
    assert loaded.metadata["source"] == "jsonl"
    assert loaded.metadata["fallback"] is True
    assert loaded.metadata["hf_error"]
    assert loaded.records


@pytest.mark.parametrize("subset", [None, "subset"])
def test_hf_revisions_are_requested_and_cached_separately(tmp_path, monkeypatch, subset) -> None:
    calls = []

    def fake_load(hf_id, *args, **kwargs):
        calls.append((hf_id, args, kwargs["revision"]))
        return [{"prompt": kwargs["revision"]}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    loader = HuggingFaceDatasetLoader()
    results = []
    for revision in ("first", "second", "first"):
        results.append(loader.load(DatasetSpec(
            name="custom", source="hf", hf_id="org/data", hf_subset=subset,
            revision=revision, cache_dir=str(tmp_path),
        )))
    assert calls == [("org/data", (subset,) if subset else (), rev) for rev in ("first", "second")]
    assert [result.records[0].prompt for result in results] == ["first", "second", "first"]
    assert results[-1].metadata["source"] == "hf_cache"
    assert results[0].metadata["cached_path"] != results[1].metadata["cached_path"]
    path = Path(results[0].metadata["cached_path"])
    sidecar = cache_sidecar_path(path)
    payload = json.loads(sidecar.read_text())
    assert payload["revision"] == "first"
    payload["revision"] = "second"
    sidecar.write_text(json.dumps(payload))
    from benchmarks.dataset_mapped_cache import read_validated_cache

    assert read_validated_cache(
        path,
        MappedHfSpec(
            name="custom",
            hf_id="org/data",
            hf_subset=subset,
            split="validation",
            revision="first",
            cache_dir=tmp_path,
        ),
    ) is None


@pytest.mark.parametrize("damage", ["missing_jsonl", "missing_sidecar", "truncated", "bad_row", "bad_sidecar", "invalid_utf8"])
def test_damaged_mapped_cache_refetches(tmp_path, monkeypatch, damage) -> None:
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(1)
        return [{"prompt": "One"}, {"prompt": "Two"}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    spec = DatasetSpec(name="custom", source="hf", hf_id="org/data", cache_dir=str(tmp_path))
    loader = HuggingFaceDatasetLoader()
    first = loader.load(spec)
    path = Path(first.metadata["cached_path"])
    sidecar = cache_sidecar_path(path)
    if damage == "missing_jsonl":
        path.unlink()
    elif damage == "missing_sidecar":
        sidecar.unlink()
    elif damage == "truncated":
        path.write_text('{"prompt": "One"}\n{"prompt":')
    elif damage == "bad_row":
        path.write_text('{"prompt": "One", "choices": 1}\n')
    elif damage == "invalid_utf8":
        path.write_bytes(b'\xff')
    else:
        sidecar.write_text('{')
    second = loader.load(spec)
    assert len(calls) == 2
    assert second.records == first.records
    assert second.metadata["source"] == "hf"


def test_cache_files_disappearing_during_read_are_misses(tmp_path, monkeypatch) -> None:
    from benchmarks import dataset_mapped_cache as cache

    path = tmp_path / "records.jsonl"
    spec = MappedHfSpec(
        name="custom",
        hf_id="org/data",
        hf_subset=None,
        split="validation",
        cache_dir=tmp_path,
    )
    cache._persist_mapped_cache(path, [DatasetRecord(prompt="One")], spec, None)
    original_read = Path.read_text

    def disappearing_sidecar(self, *args, **kwargs):
        self.unlink()
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", disappearing_sidecar)
    assert cache.read_validated_cache(path, spec) is None


@pytest.mark.parametrize("fail_publish", [None, 0, 1])
def test_mapped_cache_publishes_complete_files_and_sidecar_last(tmp_path, monkeypatch, fail_publish) -> None:
    from benchmarks import dataset_mapped_cache as cache

    path = tmp_path / "records.jsonl"
    sidecar = cache.cache_sidecar_path(path)
    published = []
    replace = cache.os.replace

    def observe_replace(source, destination):
        assert source.parent == destination.parent == tmp_path
        assert source != destination
        json.loads(source.read_text())
        if fail_publish == len(published):
            raise OSError("interrupted publication")
        published.append(destination)
        replace(source, destination)

    monkeypatch.setattr(cache.os, "replace", observe_replace)

    spec = MappedHfSpec(
        name="custom",
        hf_id="org/data",
        hf_subset=None,
        split="validation",
        cache_dir=tmp_path,
    )

    def persist():
        cache._persist_mapped_cache(
            path, [DatasetRecord(prompt="One")], spec, None
        )

    if fail_publish is None:
        persist()
        assert published == [path, sidecar]
    else:
        with pytest.raises(OSError, match="interrupted publication"):
            persist()
        assert cache.read_validated_cache(path, spec) is None
    assert set(tmp_path.iterdir()) == set(published)


def test_auto_prefer_uses_resolved_sample_without_hub(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("auto must not fetch when a local file resolves")

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", boom)
    records, metadata = load_with_fallback(
        DatasetSpec(name="piqa", source="piqa"),
        entry=CATALOG["piqa"],
        map_row=mapper_for("piqa"),
        prefer="auto",
    )
    assert records
    assert metadata["source"] == "jsonl"
    assert metadata.get("fallback") is None


def test_movable_revision_refetches_when_hub_commit_changes(tmp_path, monkeypatch) -> None:
    commits = iter(("a" * 40, "b" * 40))

    def remote(_spec):
        return next(commits)

    def fake_load(hf_id, *args, **kwargs):
        return [{"prompt": kwargs["revision"]}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache._remote_commit", remote)
    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    loader = HuggingFaceDatasetLoader()
    spec = DatasetSpec(
        name="custom",
        source="hf",
        hf_id="org/data",
        revision="main",
        cache_mode="online",
        cache_dir=str(tmp_path),
    )
    first = loader.load(spec)
    second = loader.load(spec)
    assert first.records[0].prompt == "a" * 40
    assert first.metadata["source"] == "hf"
    assert first.metadata["resolved_revision"] == "a" * 40
    assert first.metadata["source_uri"] == f"hf://datasets/org/data@{'a' * 40}"
    provenance = provenance_from_metadata(first.metadata)
    assert provenance["dataset_source_uri"] == first.metadata["source_uri"]
    assert provenance["dataset_resolved_revision"] == "a" * 40
    assert second.metadata["source"] == "hf"
    assert second.records[0].prompt == "b" * 40
    assert second.metadata["resolved_revision"] == "b" * 40


def test_offline_movable_cache_does_not_resolve_hub(tmp_path, monkeypatch) -> None:
    def remote(_spec):
        raise AssertionError("offline mode must not resolve a Hub revision")

    def fake_load(*args, **kwargs):
        return [{"prompt": "cached"}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    spec = DatasetSpec(
        name="custom",
        source="hf",
        hf_id="org/data",
        revision="main",
        cache_dir=str(tmp_path),
    )
    HuggingFaceDatasetLoader().load(spec)
    monkeypatch.setattr("benchmarks.dataset_mapped_cache._remote_commit", remote)
    monkeypatch.setattr(
        "benchmarks.dataset_mapped_cache.hf_load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("refetch")),
    )
    loaded = HuggingFaceDatasetLoader().load(
        DatasetSpec(
            name="custom",
            source="hf",
            hf_id="org/data",
            revision="main",
            cache_mode="offline",
            cache_dir=str(tmp_path),
        )
    )
    assert loaded.metadata["source"] == "hf_cache"
    assert loaded.records[0].prompt == "cached"


def test_failed_refresh_keeps_valid_cache(tmp_path, monkeypatch) -> None:
    from benchmarks import dataset_mapped_cache as cache

    spec = MappedHfSpec(
        name="piqa",
        hf_id="ybisk/piqa",
        hf_subset=None,
        split="validation",
        revision="main",
        cache_dir=tmp_path,
    )
    path = cache.normalized_cache_path(spec)
    cache._persist_mapped_cache(
        path,
        [DatasetRecord(prompt="keep", choices=["a", "b"], answer_index=0)],
        spec,
        "a" * 40,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("hub down")

    monkeypatch.setattr(cache, "hf_load_dataset", boom)
    monkeypatch.setattr(cache, "_remote_commit", lambda _spec: "b" * 40)
    records, metadata = cache.records_from_hf(
        spec, lambda row: DatasetRecord(prompt="new"), refresh=True
    )
    assert records[0].prompt == "keep"
    assert metadata["source"] == "hf_cache"
    assert metadata["stale"] is True
    cached = cache.read_validated_cache(path, spec)
    assert cached is not None
    assert cached[0].prompt == "keep"


def test_lambada_mapper_version_ignores_stale_cache(tmp_path, monkeypatch) -> None:
    from benchmarks.dataset_mapped_cache import records_from_hf
    from benchmarks.lambada.loader import LAMBADALoader, map_lambada_row

    calls = {"n": 0}

    def fake_load(*args, **kwargs):
        calls["n"] += 1
        return [{"text": "She walked home."}]

    monkeypatch.setattr("benchmarks.dataset_mapped_cache.hf_load_dataset", fake_load)
    entry = CATALOG["lambada"]
    records_from_hf(
        MappedHfSpec(
            name="lambada",
            hf_id=entry.hf_id,
            hf_subset=entry.hf_subset,
            split=entry.default_split,
            cache_dir=tmp_path,
            mapper_version="1",
        ),
        map_lambada_row,
    )
    loaded = LAMBADALoader().load(
        DatasetSpec(name="lambada", source="lambada", cache_dir=str(tmp_path))
    )
    assert calls["n"] == 2
    assert loaded.records[0].reference == "home."
    assert loaded.metadata["source"] == "hf"
