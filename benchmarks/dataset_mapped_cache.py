"""Mapped Hugging Face fetch plus a validated sidecar JSONL cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.dataset_hf_pin import (
    dataset_source_uri,
    is_commit_sha,
    resolve_dataset_revision,
)
from benchmarks.dataset_jsonl import (
    apply_max_samples,
    require_canonical_record,
    records_from_jsonl,
    row_as_dict,
)
from benchmarks.dataset_types import DatasetRecord, DatasetSpec

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

# v3 includes mapper_version so a mapping change cannot reuse older rows.
NORMALIZED_CACHE_SCHEMA = "combine.normalized_hf_cache.v3"
RowMapper = Callable[[dict[str, Any]], DatasetRecord | None]
CacheHit = tuple[list[DatasetRecord], dict[str, Any]]
CachedRows = tuple[list[DatasetRecord], dict[str, Any]]


@dataclass(frozen=True)
class MappedHfSpec:
    name: str
    hf_id: str
    hf_subset: str | None
    split: str
    revision: str | None = None
    max_samples: int | None = None
    cache_dir: Path | None = None
    mapper_version: str = "1"


def write_normalized_jsonl(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_payload(), ensure_ascii=False))
            handle.write("\n")


def cache_key_digest(spec: MappedHfSpec) -> str:
    key = json.dumps(
        [
            NORMALIZED_CACHE_SCHEMA,
            spec.hf_id,
            spec.hf_subset,
            spec.split,
            spec.revision,
            spec.mapper_version,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalized_cache_path(spec: MappedHfSpec) -> Path:
    digest = cache_key_digest(spec)
    cache_dir = _require_cache_dir(spec)
    return cache_dir / "normalized" / spec.name / spec.split / f"{digest}.jsonl"


def cache_sidecar_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".meta.json")


def jsonl_content_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cache_sidecar(
    jsonl_path: Path,
    spec: MappedHfSpec,
    row_count: int,
    resolved_revision: str | None,
) -> None:
    payload = _sidecar_payload(jsonl_path, spec, row_count, resolved_revision)
    cache_sidecar_path(jsonl_path).write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def read_validated_cache(
    jsonl_path: Path, spec: MappedHfSpec
) -> list[DatasetRecord] | None:
    loaded = _read_matching_cache(jsonl_path, spec)
    if loaded is None:
        return None
    records, _sidecar = loaded
    return apply_max_samples(records, spec.max_samples)


def records_from_hf(
    spec: MappedHfSpec,
    map_row: RowMapper,
    *,
    allow_fetch: bool = True,
) -> CacheHit:
    _require_fetch_spec(spec)
    path = normalized_cache_path(spec)
    remote = _online_commit(spec, allow_fetch)
    hit = _fresh_hit(path, spec, remote)
    if hit is not None:
        return hit
    if not allow_fetch:
        raise RuntimeError(
            f"offline cache miss for Hugging Face dataset {spec.hf_id}"
        )
    return _fetch_mapped_from_hf(path, spec, map_row, remote)


def _sidecar_payload(
    jsonl_path: Path,
    spec: MappedHfSpec,
    row_count: int,
    resolved_revision: str | None,
) -> dict[str, Any]:
    payload = _identity_fields(spec)
    payload["content_sha256"] = jsonl_content_digest(jsonl_path)
    payload["row_count"] = row_count
    payload["resolved_revision"] = resolved_revision
    payload["source_uri"] = _source_uri(spec, resolved_revision)
    return payload


def _identity_fields(spec: MappedHfSpec) -> dict[str, Any]:
    return {
        "schema": NORMALIZED_CACHE_SCHEMA,
        "name": spec.name,
        "hf_id": spec.hf_id,
        "hf_subset": spec.hf_subset,
        "split": spec.split,
        "revision": spec.revision,
        "mapper_version": spec.mapper_version,
        "digest": cache_key_digest(spec),
    }


def _sidecar_matches(
    sidecar: dict[str, Any], spec: MappedHfSpec, row_count: int
) -> bool:
    expected = _identity_fields(spec)
    expected["row_count"] = row_count
    return all(sidecar.get(field) == value for field, value in expected.items())


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(sidecar, dict):
        return sidecar
    return None


def _load_sidecar(jsonl_path: Path) -> dict[str, Any] | None:
    sidecar_path = cache_sidecar_path(jsonl_path)
    if not jsonl_path.exists() or not sidecar_path.exists():
        return None
    return _read_sidecar(sidecar_path)


def _read_matching_cache(jsonl_path: Path, spec: MappedHfSpec) -> CachedRows | None:
    try:
        return _pair_if_valid(jsonl_path, spec)
    except (OSError, ValueError, TypeError):
        return None


def _pair_if_valid(jsonl_path: Path, spec: MappedHfSpec) -> CachedRows | None:
    sidecar = _load_sidecar(jsonl_path)
    if sidecar is None:
        return None
    records = records_from_jsonl(jsonl_path, require_canonical_record, None)
    if _rows_match(jsonl_path, spec, sidecar, records):
        return records, sidecar
    return None


def _rows_match(
    jsonl_path: Path,
    spec: MappedHfSpec,
    sidecar: dict[str, Any],
    records: list[DatasetRecord],
) -> bool:
    if not _sidecar_matches(sidecar, spec, len(records)):
        return False
    return sidecar.get("content_sha256") == jsonl_content_digest(jsonl_path)


def _require_cache_dir(spec: MappedHfSpec) -> Path:
    if spec.cache_dir is None:
        raise ValueError(f"hf dataset '{spec.name}' is missing cache_dir")
    return spec.cache_dir


def _require_fetch_spec(spec: MappedHfSpec) -> None:
    if not spec.hf_id:
        raise ValueError(f"hf dataset '{spec.name}' is missing hf_id")
    if spec.max_samples is not None and spec.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    _require_cache_dir(spec)


def _source_uri(spec: MappedHfSpec, resolved_revision: str | None) -> str:
    revision = resolved_revision or spec.revision or "main"
    return dataset_source_uri(spec.hf_id, spec.hf_subset, revision)


def _hub_metadata(
    spec: MappedHfSpec, source: str, resolved_revision: str | None
) -> dict[str, Any]:
    return {
        "source": source,
        "hf_id": spec.hf_id,
        "hf_subset": spec.hf_subset,
        "split": spec.split,
        "revision": spec.revision,
        "resolved_revision": resolved_revision,
        "source_uri": _source_uri(spec, resolved_revision),
        "cache_key_digest": cache_key_digest(spec),
    }


def _hit_metadata(
    path: Path, spec: MappedHfSpec, sidecar: dict[str, Any]
) -> dict[str, Any]:
    resolved = sidecar.get("resolved_revision")
    if not isinstance(resolved, str):
        resolved = None
    metadata = _hub_metadata(spec, "hf_cache", resolved)
    metadata["path"] = str(path)
    stored_uri = sidecar.get("source_uri")
    if isinstance(stored_uri, str) and stored_uri:
        metadata["source_uri"] = stored_uri
    return metadata


def _remote_commit(spec: MappedHfSpec) -> str | None:
    """Current Hub commit for a movable revision.

    ``None`` means the Hub client is not installed, so the caller keeps a
    validated cache instead of failing closed in offline-style environments.
    """
    probe = DatasetSpec(
        name=spec.name,
        hf_id=spec.hf_id,
        hf_subset=spec.hf_subset,
        split=spec.split,
        revision=spec.revision,
    )
    try:
        return resolve_dataset_revision(probe)
    except ImportError:
        return None


def _online_commit(spec: MappedHfSpec, allow_fetch: bool) -> str | None:
    if not allow_fetch or is_commit_sha(spec.revision):
        return None
    return _remote_commit(spec)


def _cache_is_fresh(metadata: dict[str, Any], remote: str | None) -> bool:
    if remote is None:
        return True
    return metadata.get("resolved_revision") == remote


def _fresh_hit(
    path: Path, spec: MappedHfSpec, remote: str | None
) -> CacheHit | None:
    loaded = _read_matching_cache(path, spec)
    if loaded is None:
        return None
    records, sidecar = loaded
    metadata = _hit_metadata(path, spec, sidecar)
    if not _cache_is_fresh(metadata, remote):
        return None
    return apply_max_samples(records, spec.max_samples), metadata


def _require_hf_loader() -> Callable[..., Any]:
    if hf_load_dataset is None:
        raise ImportError(
            "datasets is required for Hugging Face sources; "
            "install with `pip install datasets`"
        )
    return hf_load_dataset


def _load_hf_rows(spec: MappedHfSpec, revision: str | None) -> Iterable[Any]:
    loader = _require_hf_loader()
    hf_cache = _require_cache_dir(spec) / "hf"
    hf_cache.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"split": spec.split, "cache_dir": str(hf_cache)}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return _call_hf_loader(loader, spec, kwargs)
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Hugging Face dataset {spec.hf_id}: {exc}"
        ) from exc


def _call_hf_loader(
    loader: Callable[..., Any], spec: MappedHfSpec, kwargs: dict[str, Any]
) -> Iterable[Any]:
    if spec.hf_subset:
        return loader(spec.hf_id, spec.hf_subset, **kwargs)
    return loader(spec.hf_id, **kwargs)


def _map_hf_split(
    spec: MappedHfSpec, map_row: RowMapper, revision: str | None
) -> tuple[list[DatasetRecord], int]:
    mapped: list[DatasetRecord] = []
    skipped = 0
    for row in _load_hf_rows(spec, revision):
        record = map_row(row_as_dict(row))
        if record is None:
            skipped += 1
            continue
        mapped.append(record)
    return mapped, skipped


def _temporary_jsonl(directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=directory, suffix=".jsonl", delete=False
    ) as handle:
        return Path(handle.name)


def _persist_mapped_cache(
    path: Path,
    records: list[DatasetRecord],
    spec: MappedHfSpec,
    resolved_revision: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_jsonl(path.parent)
    temporary_sidecar = cache_sidecar_path(temporary_path)
    try:
        write_normalized_jsonl(temporary_path, records)
        write_cache_sidecar(temporary_path, spec, len(records), resolved_revision)
        os.replace(temporary_path, path)
        os.replace(temporary_sidecar, cache_sidecar_path(path))
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)


def _stored_revision(revision: str | None) -> str | None:
    if is_commit_sha(revision):
        return revision
    return None


def _same_commit_winner(
    path: Path, spec: MappedHfSpec, resolved: str | None
) -> CacheHit | None:
    loaded = _winner_rows(path, spec, resolved)
    if loaded is None:
        return None
    records, sidecar = loaded
    metadata = _hit_metadata(path, spec, sidecar)
    return apply_max_samples(records, spec.max_samples), metadata


def _winner_rows(
    path: Path, spec: MappedHfSpec, resolved: str | None
) -> CachedRows | None:
    pinned = _stored_revision(resolved)
    if pinned is None:
        return None
    loaded = _read_matching_cache(path, spec)
    if loaded is None or loaded[1].get("resolved_revision") != pinned:
        return None
    return loaded


def _fetch_metadata(
    path: Path, spec: MappedHfSpec, resolved: str | None, skipped: int
) -> dict[str, Any]:
    metadata = _hub_metadata(spec, "hf", _stored_revision(resolved))
    metadata["skipped"] = skipped
    metadata["cached_path"] = str(path)
    return metadata


def _fetch_mapped_from_hf(
    path: Path,
    spec: MappedHfSpec,
    map_row: RowMapper,
    remote: str | None,
) -> CacheHit:
    revision = remote if remote is not None else spec.revision
    mapped, skipped = _map_hf_split(spec, map_row, revision)
    winner = _same_commit_winner(path, spec, revision)
    if winner is not None:
        return winner
    _persist_mapped_cache(path, mapped, spec, _stored_revision(revision))
    records = apply_max_samples(mapped, spec.max_samples)
    return records, _fetch_metadata(path, spec, revision, skipped)
