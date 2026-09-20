"""Dataset cache: offline-first loads, license manifests, and rejection paths."""

from __future__ import annotations

import errno
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.dataset_cache as dataset_cache_module
import benchmarks.dataset_cache_hf as dataset_cache_hf
import benchmarks.dataset_cache_io as dataset_cache_io
from benchmarks.dataset_cache import (
    DEFAULT_HF_REVISION,
    LICENSE_SCOPE_DATASET_SOURCE,
    LOADER_SCHEMA_VERSION,
    UNKNOWN_LICENSE,
    CacheManifest,
    CacheMissError,
    CacheMode,
    CacheValidationError,
    DatasetCache,
    FetchResult,
    cache_key_for,
    checksum_bytes,
    encode_records,
    fetch_huggingface_dataset,
    parse_cache_mode,
    requested_revision,
)
from benchmarks.dataset_cache_io import open_dir_nofollow
from benchmarks.dataset_types import validate_dataset_record
from benchmarks.datasets import DatasetRecord, DatasetSpec, HuggingFaceDatasetLoader, JsonlDatasetLoader
from benchmarks.runner import _apply_dataset_cache_defaults, run_benchmarks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dataset_cache"
FIXTURE_REVISION = "1" * 40

FIXTURE_SPEC = DatasetSpec(
    name="fixture-cloze",
    source="hf",
    hf_id="local/fixture-cloze",
    split="validation",
    revision=FIXTURE_REVISION,
)


def _network_must_not_run(spec: DatasetSpec) -> FetchResult:
    raise AssertionError(f"network client invoked for {spec.name}")


def _fetch_ok(
    spec: DatasetSpec,
    *,
    license_id: str = "cc-by-4.0",
    revision: str = FIXTURE_REVISION,
) -> FetchResult:
    return FetchResult(
        records=[
            DatasetRecord(prompt="She walked to the ", reference="store"),
            DatasetRecord(prompt="He sat on the ", reference="chair"),
        ],
        source_uri=f"hf://datasets/{spec.hf_id}@{revision}",
        resolved_revision=revision,
        upstream_license=license_id,
    )


def _install_fixture(cache: DatasetCache, spec: DatasetSpec, name: str) -> Path:
    src = FIXTURES / name
    dest = cache.entry_dir(cache.key_for(spec))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return dest


def _offline_cache(tmp_path: Path) -> DatasetCache:
    return DatasetCache(root=tmp_path / "cache", mode=CacheMode.OFFLINE)


def test_cache_key_includes_name_config_split_revision_and_schema() -> None:
    key = cache_key_for(FIXTURE_SPEC)
    assert key.dataset_name == "fixture-cloze"
    assert key.hf_id == "local/fixture-cloze"
    assert key.configuration == ""
    assert key.split == "validation"
    assert key.revision == FIXTURE_REVISION
    assert key.schema_version == LOADER_SCHEMA_VERSION
    again = cache_key_for(FIXTURE_SPEC)
    assert key.digest() == again.digest()

    other = cache_key_for(
        DatasetSpec(
            name="fixture-cloze",
            hf_id="local/fixture-cloze",
            split="validation",
            revision="2" * 40,
        )
    )
    assert other.digest() != key.digest()

    other_source = cache_key_for(
        DatasetSpec(
            name="fixture-cloze",
            hf_id="other/fixture-cloze",
            split="validation",
            revision=FIXTURE_REVISION,
        )
    )
    assert other_source.digest() != key.digest()

    subset = cache_key_for(
        DatasetSpec(
            name="fixture-cloze",
            hf_id="local/fixture-cloze",
            hf_subset="plain_text",
            split="validation",
            revision=FIXTURE_REVISION,
        )
    )
    assert subset.configuration == "plain_text"
    assert subset.digest() != key.digest()


def test_parse_cache_mode_aliases_and_rejects_unknown() -> None:
    assert parse_cache_mode("ONLINE") is CacheMode.ONLINE
    assert parse_cache_mode("prefer_cache") is CacheMode.PREFER_CACHE
    assert parse_cache_mode("offline") is CacheMode.OFFLINE
    with pytest.raises(ValueError, match="unknown cache mode"):
        parse_cache_mode("maybe")


def test_offline_hit_does_not_invoke_network(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    entry = _install_fixture(cache, FIXTURE_SPEC, "hit")

    loaded = cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    assert loaded.cache_hit is True
    assert loaded.cache_path == entry
    assert len(loaded.records) == 2
    assert loaded.records[0].reference == "store"
    assert loaded.manifest.upstream_license == "cc-by-4.0"
    assert loaded.manifest.license_scope == LICENSE_SCOPE_DATASET_SOURCE
    assert loaded.manifest.cache_key_digest == cache.key_for(FIXTURE_SPEC).digest()


def test_offline_miss_does_not_invoke_network(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    expected = cache.entry_dir(cache.key_for(FIXTURE_SPEC))

    with pytest.raises(CacheMissError, match="Offline cache miss") as excinfo:
        cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    error = excinfo.value
    assert error.path == expected
    assert error.reason == "missing-artifact"
    assert str(expected) in str(error)
    assert "No network client was invoked" in str(error)


def test_repeated_loads_reuse_the_same_immutable_entry(tmp_path: Path) -> None:
    cache = DatasetCache(root=tmp_path / "cache", mode=CacheMode.PREFER_CACHE)
    calls: list[str] = []

    def fetch(spec: DatasetSpec) -> FetchResult:
        calls.append(spec.name)
        return _fetch_ok(spec)

    first = cache.load(FIXTURE_SPEC, fetch=fetch)
    records_stat = (first.cache_path / "records.jsonl").stat()
    manifest_stat = (first.cache_path / "manifest.json").stat()
    second = cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    assert calls == ["fixture-cloze"]
    assert first.cache_path == second.cache_path
    assert first.manifest.checksum_sha256 == second.manifest.checksum_sha256
    assert first.manifest.retrieved_at == second.manifest.retrieved_at
    assert (second.cache_path / "records.jsonl").stat().st_mtime_ns == records_stat.st_mtime_ns
    assert (second.cache_path / "manifest.json").stat().st_mtime_ns == manifest_stat.st_mtime_ns


def test_online_mode_keeps_entry_when_fetch_matches(tmp_path: Path) -> None:
    cache = DatasetCache(root=tmp_path / "cache", mode=CacheMode.ONLINE)
    first = cache.load(FIXTURE_SPEC, fetch=_fetch_ok)
    retrieved_at = first.manifest.retrieved_at
    second = cache.load(FIXTURE_SPEC, fetch=_fetch_ok)
    assert second.cache_path == first.cache_path
    assert second.manifest.retrieved_at == retrieved_at
    assert second.cache_hit is True


def test_checksum_mismatch_is_quarantined_not_deleted(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    entry = _install_fixture(cache, FIXTURE_SPEC, "checksum-mismatch")
    original = (entry / "records.jsonl").read_bytes()

    with pytest.raises(CacheValidationError, match="checksum-mismatch") as excinfo:
        cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    error = excinfo.value
    assert error.path == entry
    assert error.reason is not None and "checksum-mismatch" in error.reason
    assert error.quarantined_to is not None
    assert error.quarantined_to.is_dir()
    assert (error.quarantined_to / "records.jsonl").read_bytes() == original
    assert not entry.exists()
    assert "Quarantined to" in str(error)


def test_prefer_cache_refetches_after_quarantining_invalid_entry(tmp_path: Path) -> None:
    cache = DatasetCache(root=tmp_path / "cache", mode=CacheMode.PREFER_CACHE)
    _install_fixture(cache, FIXTURE_SPEC, "checksum-mismatch")
    loaded = cache.load(FIXTURE_SPEC, fetch=_fetch_ok)
    assert loaded.cache_hit is False
    assert len(loaded.records) == 2
    assert loaded.manifest.checksum_sha256 == checksum_bytes(
        encode_records(loaded.records)
    )


def test_stale_schema_is_rejected_with_path_and_reason(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    entry = _install_fixture(cache, FIXTURE_SPEC, "stale-schema")

    with pytest.raises(CacheValidationError, match="stale-schema") as excinfo:
        cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    error = excinfo.value
    assert error.path == entry
    assert error.reason is not None
    assert "stale-schema" in error.reason
    assert f"expected={LOADER_SCHEMA_VERSION}" in error.reason
    assert error.quarantined_to is not None
    assert (error.quarantined_to / "manifest.json").exists()


def test_unknown_license_is_an_explicit_value(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    _install_fixture(cache, FIXTURE_SPEC, "unknown-license")

    loaded = cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)
    assert loaded.manifest.upstream_license == UNKNOWN_LICENSE
    metadata = loaded.metadata(source="hf")
    assert metadata["dataset_upstream_license"] == UNKNOWN_LICENSE
    assert metadata["dataset_license_scope"] == LICENSE_SCOPE_DATASET_SOURCE
    assert "license" not in metadata or metadata["dataset_license_scope"] != "repository"


def test_hf_loader_offline_hit_skips_injected_client(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="fixture-cloze",
        source="hf",
        hf_id="local/fixture-cloze",
        split="validation",
        revision=FIXTURE_REVISION,
        cache_mode="offline",
        cache_root=str(tmp_path / "cache"),
    )
    cache = DatasetCache.for_spec(spec)
    _install_fixture(cache, spec, "hit")
    loader = HuggingFaceDatasetLoader(fetch=_network_must_not_run)

    dataset = loader.load(spec)

    assert len(dataset.records) == 2
    assert dataset.metadata["cache_hit"] is True
    assert dataset.metadata["dataset_upstream_license"] == "cc-by-4.0"
    assert dataset.metadata["dataset_license_scope"] == LICENSE_SCOPE_DATASET_SOURCE


def test_hf_loader_offline_miss_skips_injected_client(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="fixture-cloze",
        source="hf",
        hf_id="local/fixture-cloze",
        split="validation",
        revision=FIXTURE_REVISION,
        cache_mode="offline",
        cache_root=str(tmp_path / "cache"),
    )
    loader = HuggingFaceDatasetLoader(fetch=_network_must_not_run)
    with pytest.raises(CacheMissError, match="Offline cache miss"):
        loader.load(spec)


def test_prefer_cache_miss_fetches_once(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="fixture-cloze",
        source="hf",
        hf_id="local/fixture-cloze",
        split="validation",
        revision=FIXTURE_REVISION,
        cache_mode="prefer-cache",
        cache_root=str(tmp_path / "cache"),
        max_samples=1,
    )
    calls = {"n": 0}

    def fetch(item: DatasetSpec) -> FetchResult:
        calls["n"] += 1
        return _fetch_ok(item)

    loader = HuggingFaceDatasetLoader(fetch=fetch)
    first = loader.load(spec)
    second = loader.load(spec)

    assert calls["n"] == 1
    assert len(first.records) == 1
    assert len(second.records) == 1
    assert second.metadata["cache_hit"] is True
    assert second.metadata["row_count"] == 2


def test_hf_loader_rejects_negative_max_samples() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DatasetSpec(
            name="fixture-cloze",
            source="hf",
            hf_id="local/fixture-cloze",
            max_samples=-1,
        )


def test_dataset_spec_rejects_boolean_max_samples() -> None:
    with pytest.raises(ValueError, match="integer"):
        DatasetSpec(name="fixture-cloze", max_samples=False)


def test_dataset_record_rejects_boolean_answer_index() -> None:
    with pytest.raises(ValueError, match="integer answer_index"):
        validate_dataset_record(
            DatasetRecord(prompt="question", choices=["no", "yes"], answer_index=True)
        )


def test_unpinned_hf_revision_defaults_to_main() -> None:
    spec = DatasetSpec(name="fixture-cloze", hf_id="local/fixture-cloze")
    assert requested_revision(spec) == DEFAULT_HF_REVISION
    assert cache_key_for(spec).revision == DEFAULT_HF_REVISION


def test_jsonl_max_samples_zero(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )
    loaded = JsonlDatasetLoader().load(
        DatasetSpec(name="smoke", source="jsonl", path=str(path), max_samples=0)
    )
    assert loaded.records == []


def test_explicit_license_override_on_cache_hit(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="fixture-cloze",
        source="hf",
        hf_id="local/fixture-cloze",
        split="validation",
        revision=FIXTURE_REVISION,
        cache_mode="offline",
        cache_root=str(tmp_path / "cache"),
        upstream_license="cc-by-4.0",
    )
    cache = DatasetCache.for_spec(spec)
    _install_fixture(cache, spec, "unknown-license")
    loaded = HuggingFaceDatasetLoader(fetch=_network_must_not_run).load(spec)
    assert loaded.metadata["dataset_upstream_license"] == "cc-by-4.0"


def test_relative_dataset_cache_root_is_resolved_against_config_dir(tmp_path: Path) -> None:
    spec = DatasetSpec(name="smoke", cache_root="dataset-cache")
    updated = _apply_dataset_cache_defaults(spec, {}, tmp_path)
    assert updated.cache_root == str((tmp_path / "dataset-cache").resolve())


def test_sample_manifest_fixture_is_machine_readable() -> None:
    raw = json.loads((FIXTURES / "sample_manifest.json").read_text(encoding="utf-8"))
    manifest = CacheManifest.from_dict(raw)
    assert manifest.upstream_license == "cc-by-4.0"
    assert manifest.license_scope == LICENSE_SCOPE_DATASET_SOURCE
    assert manifest.loader_schema_version == LOADER_SCHEMA_VERSION
    assert manifest.row_count == 2
    assert len(manifest.checksum_sha256) == 64
    records = encode_records(
        [
            DatasetRecord(prompt="She walked to the ", reference="store"),
            DatasetRecord(prompt="He sat on the ", reference="chair"),
        ]
    )
    assert checksum_bytes(records) == manifest.checksum_sha256


def test_run_record_surfaces_dataset_license_not_repo_license(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "run_name": "license-evidence",
        "seed": 1,
        "model": {"backend": "mock", "name": "toy-model", "revision": "local"},
        "quantization": ["fp16"],
        "datasets": [
            {
                "name": "smoke",
                "source": "jsonl",
                "path": str(dataset_path),
                "split": "validation",
                "upstream_license": "unknown",
            }
        ],
    }
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "reports"

    run_benchmarks(config_path, output_dir, ["json"], seed_override=1)

    report_path = next((output_dir / "json").glob("*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = report["results"][0]
    assert row["dataset_upstream_license"] == UNKNOWN_LICENSE
    assert row["dataset_license_scope"] == LICENSE_SCOPE_DATASET_SOURCE
    assert "license" not in report["run"]


def test_symlink_entry_is_rejected_without_moving_the_target(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    outside = tmp_path / "outside-secrets"
    shutil.copytree(FIXTURES / "hit", outside)
    marker = outside / "do-not-move.txt"
    marker.write_text("secret\n", encoding="utf-8")

    entry = cache.entry_dir(cache.key_for(FIXTURE_SPEC))
    entry.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(outside, entry)

    with pytest.raises(CacheValidationError, match="unsafe-symlink") as excinfo:
        cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    error = excinfo.value
    assert error.reason == "unsafe-symlink"
    assert not entry.exists()
    assert not entry.is_symlink()
    assert outside.is_dir()
    assert marker.read_text(encoding="utf-8") == "secret\n"
    assert (outside / "records.jsonl").exists()
    assert error.quarantined_to is not None
    assert (error.quarantined_to / "rejected-symlink").read_text(encoding="utf-8").strip() == str(
        outside
    )
    assert not error.quarantined_to.is_symlink()


def test_symlink_records_file_is_rejected_without_following(tmp_path: Path) -> None:
    cache = _offline_cache(tmp_path)
    outside = tmp_path / "outside-records.jsonl"
    outside.write_text("not-dataset-bytes\n", encoding="utf-8")
    entry = _install_fixture(cache, FIXTURE_SPEC, "hit")
    records = entry / "records.jsonl"
    records.unlink()
    os.symlink(outside, records)

    with pytest.raises(CacheValidationError, match="unsafe-symlink") as excinfo:
        cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    assert excinfo.value.reason == "unsafe-symlink"
    assert outside.read_text(encoding="utf-8") == "not-dataset-bytes\n"
    assert not entry.exists()
    assert excinfo.value.quarantined_to is not None
    assert (excinfo.value.quarantined_to / "records.jsonl").is_symlink()


@pytest.mark.parametrize(
    ("error_number", "reason"),
    [
        (errno.ELOOP, "unsafe-symlink"),
        (errno.ENOTDIR, "not-a-directory"),
        (errno.EACCES, "unreadable-entry"),
    ],
)
def test_open_dir_preserves_failure_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int, reason: str
) -> None:
    entry = tmp_path / "entry"
    entry.mkdir()

    def fail_open(*args, **kwargs):
        raise OSError(error_number, os.strerror(error_number))

    monkeypatch.setattr(dataset_cache_io.os, "open", fail_open)
    with pytest.raises(CacheValidationError) as excinfo:
        open_dir_nofollow(entry)
    assert excinfo.value.reason == reason


def test_unreadable_entry_is_quarantined_with_accurate_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _offline_cache(tmp_path)
    entry = _install_fixture(cache, FIXTURE_SPEC, "hit")
    real_open = os.open

    def deny_entry(path, *args, **kwargs):
        if Path(path) == entry:
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(dataset_cache_io.os, "open", deny_entry)
    with pytest.raises(CacheValidationError) as excinfo:
        cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)

    error = excinfo.value
    assert error.reason == "unreadable-entry"
    assert error.quarantined_to is not None
    note = json.loads(
        (error.quarantined_to / "quarantine.json").read_text(encoding="utf-8")
    )
    assert note["reason"] == "unreadable-entry"


def test_symlink_cache_root_can_store(tmp_path: Path) -> None:
    real = tmp_path / "real-cache"
    real.mkdir()
    link = tmp_path / "link-cache"
    os.symlink(real, link)
    cache = DatasetCache(root=link, mode=CacheMode.PREFER_CACHE)

    loaded = cache.load(FIXTURE_SPEC, fetch=_fetch_ok)

    assert loaded.cache_hit is False
    digest = cache.key_for(FIXTURE_SPEC).digest()
    live = cache.entry_dir(cache.key_for(FIXTURE_SPEC))
    assert live.is_dir()
    assert not live.is_symlink()
    assert (live / "records.jsonl").is_file()
    assert (real / digest / "records.jsonl").is_file()


def test_second_store_keeps_existing_live_entry(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = DatasetCache(root=root, mode=CacheMode.PREFER_CACHE)
    first = cache.load(FIXTURE_SPEC, fetch=_fetch_ok)
    loser = FetchResult(
        records=[DatasetRecord(prompt="losing record", reference="loser")],
        source_uri=f"hf://datasets/{FIXTURE_SPEC.hf_id}@{FIXTURE_REVISION}",
        resolved_revision=FIXTURE_REVISION,
        upstream_license="cc-by-4.0",
    )
    loser_payload = encode_records(loser.records)
    again = cache._store(
        cache.key_for(FIXTURE_SPEC),
        loser,
        loser_payload,
        checksum_bytes(loser_payload),
    )
    assert again.cache_path == first.cache_path
    assert again.cache_hit is True
    assert again.records == first.records
    assert again.manifest == first.manifest
    assert first.cache_path.is_dir()
    assert list(root.glob(".*.staging-*")) == []
    notes = list((root / "quarantine").glob("*/quarantine.json"))
    assert notes
    reasons = [json.loads(path.read_text(encoding="utf-8"))["reason"] for path in notes]
    assert "lost-publish-race" in reasons
    hit = cache.load(FIXTURE_SPEC, fetch=_network_must_not_run)
    assert hit.cache_hit is True
    assert hit.manifest.checksum_sha256 == first.manifest.checksum_sha256


def test_publish_race_propagates_invalid_winner(tmp_path: Path) -> None:
    cache = DatasetCache(root=tmp_path / "cache", mode=CacheMode.PREFER_CACHE)
    key = cache.key_for(FIXTURE_SPEC)
    live = cache.entry_dir(key)
    live.mkdir(parents=True)
    (live / "records.jsonl").write_text("corrupt\n", encoding="utf-8")

    fetched = _fetch_ok(FIXTURE_SPEC)
    payload = encode_records(fetched.records)
    with pytest.raises(CacheValidationError, match="incomplete cache entry"):
        cache._store(key, fetched, payload, checksum_bytes(payload))

    assert live.is_dir()
    assert (live / "records.jsonl").read_text(encoding="utf-8") == "corrupt\n"


def test_movable_hf_revision_is_pinned_before_key_and_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_revision = "a" * 40
    spec = DatasetSpec(
        name="movable",
        source="hf",
        hf_id="org/dataset",
        revision="main",
    )
    observed: dict[str, object] = {}

    class FakeDataset(list):
        info = SimpleNamespace(license="mit")

    def resolve(item: DatasetSpec) -> str:
        observed["resolved_spec"] = item
        return resolved_revision

    def load_dataset(dataset_id, subset, *, split, revision):
        observed["load"] = (dataset_id, subset, split, revision)
        return FakeDataset([{"prompt": "question", "reference": "answer"}])

    monkeypatch.setattr(dataset_cache_module, "resolve_huggingface_revision", resolve)
    monkeypatch.setattr(dataset_cache_hf, "hf_load_dataset", load_dataset)
    cache = DatasetCache(root=tmp_path / "cache", mode=CacheMode.PREFER_CACHE)

    loaded = cache.load(spec, fetch=fetch_huggingface_dataset)

    expected_key = cache_key_for(spec, resolved_revision=resolved_revision)
    assert loaded.cache_path == cache.entry_dir(expected_key)
    assert loaded.manifest.cache_key["revision"] == resolved_revision
    assert loaded.manifest.resolved_revision == resolved_revision
    assert loaded.manifest.source_uri == (
        f"hf://datasets/{spec.hf_id}@{resolved_revision}"
    )
    assert observed["load"] == (
        spec.hf_id,
        None,
        spec.split,
        resolved_revision,
    )


def test_offline_legacy_hit_does_not_resolve_movable_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = DatasetSpec(
        name="legacy",
        source="hf",
        hf_id="org/dataset",
        revision="main",
    )
    writer = DatasetCache(root=tmp_path / "cache", mode=CacheMode.PREFER_CACHE)
    fetched = _fetch_ok(spec, revision="main")
    payload = encode_records(fetched.records)
    writer._store(
        writer.key_for(spec), fetched, payload, checksum_bytes(payload)
    )

    def fail_resolution(_spec: DatasetSpec) -> str:
        raise AssertionError("remote revision resolution invoked")

    monkeypatch.setattr(
        dataset_cache_module, "resolve_huggingface_revision", fail_resolution
    )
    cache = DatasetCache(root=writer.root, mode=CacheMode.OFFLINE)
    loaded = cache.load(spec, fetch=_network_must_not_run)

    assert loaded.cache_hit is True
    assert loaded.manifest.resolved_revision == "main"
