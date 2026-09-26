"""Mapped Hugging Face fetch plus a validated sidecar JSONL cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from benchmarks.dataset_jsonl import (
    apply_max_samples,
    canonical_record,
    records_from_jsonl,
    row_as_dict,
)
from benchmarks.dataset_types import DatasetRecord

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

NORMALIZED_CACHE_SCHEMA = "combine.normalized_hf_cache.v1"
RowMapper = Callable[[dict[str, Any]], DatasetRecord | None]


def write_normalized_jsonl(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_payload(), ensure_ascii=False))
            handle.write("\n")


def cache_key_digest(hf_id: str, subset: str | None, split: str) -> str:
    key = f"{hf_id}\0{subset or ''}\0{split}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def normalized_cache_path(
    cache_dir: Path, name: str, hf_id: str, subset: str | None, split: str
) -> Path:
    digest = cache_key_digest(hf_id, subset, split)
    return cache_dir / "normalized" / name / split / f"{digest}.jsonl"


def cache_sidecar_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".meta.json")


def write_cache_sidecar(
    jsonl_path: Path,
    *,
    name: str,
    hf_id: str,
    subset: str | None,
    split: str,
    row_count: int,
) -> None:
    digest = cache_key_digest(hf_id, subset, split)
    payload = {
        "schema": NORMALIZED_CACHE_SCHEMA,
        "name": name,
        "hf_id": hf_id,
        "hf_subset": subset,
        "split": split,
        "digest": digest,
        "row_count": row_count,
    }
    cache_sidecar_path(jsonl_path).write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _sidecar_matches(
    sidecar: dict[str, Any],
    *,
    name: str,
    hf_id: str,
    subset: str | None,
    split: str,
    row_count: int,
) -> bool:
    expected = cache_key_digest(hf_id, subset, split)
    return (
        sidecar.get("schema") == NORMALIZED_CACHE_SCHEMA
        and sidecar.get("name") == name
        and sidecar.get("hf_id") == hf_id
        and sidecar.get("hf_subset") == subset
        and sidecar.get("split") == split
        and sidecar.get("digest") == expected
        and sidecar.get("row_count") == row_count
    )


def _cache_records_valid(records: list[DatasetRecord]) -> bool:
    return all(str(record.prompt).strip() for record in records)


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(sidecar, dict):
        return sidecar
    return None


def read_validated_cache(
    jsonl_path: Path,
    *,
    name: str,
    hf_id: str,
    subset: str | None,
    split: str,
    max_samples: int | None,
) -> list[DatasetRecord] | None:
    sidecar_path = cache_sidecar_path(jsonl_path)
    if not jsonl_path.exists() or not sidecar_path.exists():
        return None
    sidecar = _read_sidecar(sidecar_path)
    if sidecar is None:
        return None
    records = records_from_jsonl(jsonl_path, canonical_record, None)
    if not _sidecar_matches(
        sidecar,
        name=name,
        hf_id=hf_id,
        subset=subset,
        split=split,
        row_count=len(records),
    ):
        return None
    if not _cache_records_valid(records):
        return None
    return apply_max_samples(records, max_samples)


def discard_invalid_cache(jsonl_path: Path) -> None:
    sidecar = cache_sidecar_path(jsonl_path)
    if jsonl_path.exists():
        jsonl_path.unlink()
    if sidecar.exists():
        sidecar.unlink()


def _require_hf_loader() -> Callable[..., Any]:
    if hf_load_dataset is None:
        raise ImportError(
            "datasets is required for Hugging Face sources; "
            "install with `pip install datasets`"
        )
    return hf_load_dataset


def _load_hf_rows(
    hf_id: str,
    hf_subset: str | None,
    split: str,
    hf_cache: Path,
) -> Iterable[Any]:
    loader = _require_hf_loader()
    hf_cache.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"split": split, "cache_dir": str(hf_cache)}
    try:
        if hf_subset:
            return loader(hf_id, hf_subset, **kwargs)
        return loader(hf_id, **kwargs)
    except ImportError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Hugging Face dataset {hf_id}: {exc}"
        ) from exc


def _hf_cache_hit(
    path: Path,
    *,
    name: str,
    hf_id: str,
    hf_subset: str | None,
    split: str,
    max_samples: int | None,
) -> tuple[list[DatasetRecord], dict] | None:
    cached = read_validated_cache(
        path,
        name=name,
        hf_id=hf_id,
        subset=hf_subset,
        split=split,
        max_samples=max_samples,
    )
    if cached is None:
        return None
    return cached, {
        "source": "hf_cache",
        "path": str(path),
        "hf_id": hf_id,
        "hf_subset": hf_subset,
        "split": split,
    }


def _map_hf_split(
    *,
    hf_id: str,
    hf_subset: str | None,
    split: str,
    cache_dir: Path,
    map_row: RowMapper,
) -> tuple[list[DatasetRecord], int]:
    mapped: list[DatasetRecord] = []
    skipped = 0
    for row in _load_hf_rows(hf_id, hf_subset, split, cache_dir / "hf"):
        record = map_row(row_as_dict(row))
        if record is None:
            skipped += 1
            continue
        mapped.append(record)
    return mapped, skipped


def _persist_mapped_cache(
    path: Path,
    mapped: list[DatasetRecord],
    *,
    name: str,
    hf_id: str,
    hf_subset: str | None,
    split: str,
) -> None:
    write_normalized_jsonl(path, mapped)
    write_cache_sidecar(
        path,
        name=name,
        hf_id=hf_id,
        subset=hf_subset,
        split=split,
        row_count=len(mapped),
    )


def records_from_hf(
    *,
    name: str,
    hf_id: str,
    hf_subset: str | None,
    split: str,
    map_row: RowMapper,
    max_samples: int | None,
    cache_dir: Path,
) -> tuple[list[DatasetRecord], dict]:
    if not hf_id:
        raise ValueError(f"hf dataset '{name}' is missing hf_id")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")

    path = normalized_cache_path(cache_dir, name, hf_id, hf_subset, split)
    hit = _hf_cache_hit(
        path,
        name=name,
        hf_id=hf_id,
        hf_subset=hf_subset,
        split=split,
        max_samples=max_samples,
    )
    if hit is not None:
        return hit
    discard_invalid_cache(path)
    mapped, skipped = _map_hf_split(
        hf_id=hf_id,
        hf_subset=hf_subset,
        split=split,
        cache_dir=cache_dir,
        map_row=map_row,
    )
    _persist_mapped_cache(
        path, mapped, name=name, hf_id=hf_id, hf_subset=hf_subset, split=split
    )
    return apply_max_samples(mapped, max_samples), {
        "source": "hf",
        "hf_id": hf_id,
        "hf_subset": hf_subset,
        "split": split,
        "skipped": skipped,
        "cached_path": str(path),
    }
