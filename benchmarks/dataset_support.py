from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from benchmarks.dataset_cache_models import (
    DEFAULT_CACHE_MODE,
    CacheMode,
    jsonl_provenance_metadata,
    parse_cache_mode,
)
from benchmarks.dataset_jsonl import records_from_jsonl, require_canonical_record
from benchmarks.dataset_mapped_cache import MappedHfSpec, records_from_hf
from benchmarks.dataset_types import (
    CatalogEntry,
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
    TaskKind,
)
from benchmarks.dataset_validation import validate_loaded

REPO_ROOT = Path(__file__).resolve().parents[1]
ROW_MAPPERS: dict[str, Callable[[dict[str, Any]], DatasetRecord | None]] = {}

Prefer = Literal["auto", "jsonl", "hf"]

CATALOG: dict[str, CatalogEntry] = {}


def _alias_collision_message(alias: str, owner: CatalogEntry) -> str:
    return (
        f"alias {alias!r} collides with primary catalog name {owner.name!r}"
    )


def _check_alias(alias: str, entry: CatalogEntry) -> None:
    if alias == entry.name:
        return
    existing = CATALOG.get(alias)
    if existing is None:
        return
    if existing.name == alias and existing.name != entry.name:
        raise ValueError(_alias_collision_message(alias, existing))
    if existing.name != entry.name:
        raise ValueError(
            f"alias {alias!r} is already bound to catalog entry {existing.name!r}"
        )


def _register_entry(entry: CatalogEntry) -> CatalogEntry:
    for alias in entry.aliases:
        _check_alias(alias, entry)
    CATALOG[entry.name] = entry
    for alias in entry.aliases:
        CATALOG[alias] = entry
    return entry


_register_entry(
    CatalogEntry(
        name="lambada",
        task=TaskKind.CLOZE,
        hf_id="EleutherAI/lambada_openai",
        sample_relpath="configs/datasets/lambada.sample.jsonl",
        default_split="test",
        # v2 keeps the gold last token verbatim, including punctuation.
        mapper_version="2",
    )
)
_register_entry(
    CatalogEntry(
        name="hellaswag",
        task=TaskKind.MULTIPLE_CHOICE,
        hf_id="Rowan/hellaswag",
        sample_relpath="configs/datasets/hellaswag.sample.jsonl",
        default_split="validation",
    )
)
_register_entry(
    CatalogEntry(
        name="wikitext2",
        task=TaskKind.LANGUAGE_MODELING,
        hf_id="Salesforce/wikitext",
        hf_subset="wikitext-2-raw-v1",
        sample_relpath="configs/datasets/wikitext2.sample.jsonl",
        default_split="validation",
        aliases=("wikitext",),
    )
)
_register_entry(
    CatalogEntry(
        name="gsm8k",
        task=TaskKind.MATH,
        hf_id="openai/gsm8k",
        hf_subset="main",
        sample_relpath="configs/datasets/gsm8k.sample.jsonl",
        default_split="test",
    )
)
_register_entry(
    CatalogEntry(
        name="piqa",
        task=TaskKind.MULTIPLE_CHOICE,
        hf_id="ybisk/piqa",
        sample_relpath="configs/datasets/piqa.sample.jsonl",
        default_split="validation",
    )
)
_register_entry(
    CatalogEntry(
        name="arc_easy",
        task=TaskKind.MULTIPLE_CHOICE,
        hf_id="allenai/ai2_arc",
        hf_subset="ARC-Easy",
        sample_relpath="configs/datasets/arc_easy.sample.jsonl",
        default_split="validation",
        aliases=("arc-easy",),
    )
)


def register_row_mapper(
    name: str, mapper: Callable[[dict[str, Any]], DatasetRecord | None]
) -> None:
    # Store only on the canonical catalog name. Aliases resolve in mapper_for
    # so registration order cannot leave an alias without a mapper.
    ROW_MAPPERS[canonical_catalog_name(name)] = mapper


def catalog_entry(name: str) -> CatalogEntry | None:
    return CATALOG.get(name)


def is_language_modeling(dataset: LoadedDataset) -> bool:
    task = dataset.metadata.get("task")
    if task == TaskKind.LANGUAGE_MODELING.value:
        return True
    entry = catalog_entry(dataset.spec.source) or catalog_entry(dataset.spec.name)
    return entry is not None and entry.task is TaskKind.LANGUAGE_MODELING


def canonical_catalog_name(name: str) -> str:
    entry = CATALOG.get(name)
    if entry is None:
        return name
    return entry.name


def sample_path_for(entry: CatalogEntry) -> Path:
    return REPO_ROOT / entry.sample_relpath


def default_cache_dir() -> Path:
    override = os.environ.get("COMBINE_DATASET_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "combine-for-ai" / "datasets"


def resolve_cache_dir(spec: DatasetSpec) -> Path:
    if spec.cache_dir:
        return Path(spec.cache_dir)
    if spec.cache_root:
        return Path(spec.cache_root)
    return default_cache_dir()


def mapper_for(name: str) -> Callable[[dict[str, Any]], DatasetRecord | None]:
    return ROW_MAPPERS.get(canonical_catalog_name(name), require_canonical_record)


def resolve_hf_split(spec: DatasetSpec, entry: CatalogEntry | None) -> str:
    if spec.split is not None:
        return spec.split
    if entry is not None:
        return entry.default_split
    return "validation"


def apply_resolved_split(
    spec: DatasetSpec, entry: CatalogEntry | None
) -> DatasetSpec:
    split = resolve_hf_split(spec, entry)
    if spec.split == split:
        return spec
    return cast(DatasetSpec, replace(spec, split=split))


def resolve_jsonl_path(
    spec: DatasetSpec, entry: CatalogEntry | None
) -> Path | None:
    explicit = _explicit_jsonl_path(spec)
    if explicit is not None:
        return explicit
    return _sample_jsonl_path(entry)


def _explicit_jsonl_path(spec: DatasetSpec) -> Path | None:
    if not spec.path:
        return None
    path = Path(spec.path)
    if path.exists():
        return path
    repo_path = REPO_ROOT / spec.path
    if repo_path.exists():
        return repo_path
    return path


def _sample_jsonl_path(entry: CatalogEntry | None) -> Path | None:
    if entry is None:
        return None
    sample = sample_path_for(entry)
    if sample.exists():
        return sample
    return None


def _with_catalog_task(metadata: dict, entry: CatalogEntry | None) -> dict:
    if entry is None:
        return metadata
    stamped = dict(metadata)
    stamped["task"] = entry.task.value
    return stamped


def _jsonl_metadata(
    spec: DatasetSpec,
    path: Path,
    row_count: int,
    *,
    fallback: str | None = None,
) -> dict:
    metadata = jsonl_provenance_metadata(spec, path, row_count)
    if fallback:
        metadata["fallback"] = True
        metadata["hf_error"] = fallback
    return metadata


def _load_jsonl_or_raise(
    path: Path | None,
    spec: DatasetSpec,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    *,
    missing_message: str,
) -> tuple[list[DatasetRecord], dict]:
    if path is None:
        raise ValueError(missing_message)
    records = records_from_jsonl(path, map_row, spec.max_samples)
    return records, _jsonl_metadata(spec, path, len(records))


HfLoadError = (
    ImportError | OSError | RuntimeError | ValueError | TypeError
)


def _fallback_jsonl(
    spec: DatasetSpec,
    jsonl_path: Path | None,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    exc: HfLoadError,
) -> tuple[list[DatasetRecord], dict]:
    fallback_path = jsonl_path if jsonl_path is not None and jsonl_path.exists() else None
    if fallback_path is None:
        raise exc
    records = records_from_jsonl(fallback_path, map_row, spec.max_samples)
    return records, _jsonl_metadata(
        spec, fallback_path, len(records), fallback=str(exc)
    )


def _cache_mode(spec: DatasetSpec) -> CacheMode:
    raw = spec.cache_mode or os.environ.get("COMBINE_DATASET_CACHE_MODE")
    return parse_cache_mode(raw or DEFAULT_CACHE_MODE)


def _hf_request(
    spec: DatasetSpec, entry: CatalogEntry | None
) -> tuple[str | None, str | None, str, Path]:
    hf_id = spec.hf_id or (entry.hf_id if entry else None)
    hf_subset = spec.hf_subset
    if hf_subset is None and entry is not None:
        hf_subset = entry.hf_subset
    return hf_id, hf_subset, resolve_hf_split(spec, entry), resolve_cache_dir(spec)


def _mapper_version(entry: CatalogEntry | None) -> str:
    if entry is None:
        return "1"
    return entry.mapper_version


def _selected_jsonl(
    spec: DatasetSpec,
    jsonl_path: Path | None,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    prefer: Prefer,
) -> tuple[list[DatasetRecord], dict] | None:
    if prefer == "jsonl":
        return _load_jsonl_or_raise(
            jsonl_path,
            spec,
            map_row,
            missing_message=f"jsonl dataset '{spec.name}' is missing a path",
        )
    # Automatic selection stops at a resolved local file, including a catalog
    # sample. Named loaders pass prefer="hf" unless an explicit path is set.
    if prefer == "auto" and jsonl_path is not None:
        return _load_jsonl_or_raise(
            jsonl_path,
            spec,
            map_row,
            missing_message=f"dataset file not found: {spec.path}",
        )
    return None


def _load_hf_or_fallback(
    spec: DatasetSpec,
    entry: CatalogEntry | None,
    jsonl_path: Path | None,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
) -> tuple[list[DatasetRecord], dict]:
    hf_id, hf_subset, split, cache_dir = _hf_request(spec, entry)
    if not hf_id:
        return _fallback_jsonl(
            spec,
            jsonl_path,
            map_row,
            ValueError(f"hf dataset '{spec.name}' is missing hf_id"),
        )
    try:
        cache_mode = _cache_mode(spec)
        return records_from_hf(
            MappedHfSpec(
                name=spec.name,
                hf_id=hf_id,
                hf_subset=hf_subset,
                split=split,
                revision=spec.revision,
                max_samples=spec.max_samples,
                cache_dir=cache_dir,
                mapper_version=_mapper_version(entry),
                task=entry.task if entry is not None else TaskKind.GENERIC,
            ),
            map_row,
            allow_fetch=cache_mode is not CacheMode.OFFLINE,
            refresh=cache_mode is CacheMode.ONLINE,
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        return _fallback_jsonl(spec, jsonl_path, map_row, exc)


def load_with_fallback(
    spec: DatasetSpec,
    *,
    entry: CatalogEntry | None,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    prefer: Prefer,
) -> tuple[list[DatasetRecord], dict]:
    jsonl_path = resolve_jsonl_path(spec, entry)
    selected = _selected_jsonl(spec, jsonl_path, map_row, prefer)
    if selected is not None:
        return selected
    return _load_hf_or_fallback(spec, entry, jsonl_path, map_row)


class JsonlDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if not spec.path:
            raise ValueError(f"jsonl dataset '{spec.name}' is missing a path")
        entry = catalog_entry(spec.name)
        spec = apply_resolved_split(spec, entry)
        records, metadata = load_with_fallback(
            spec,
            entry=entry,
            map_row=mapper_for(spec.name),
            prefer="jsonl",
        )
        loaded = LoadedDataset(
            spec=spec,
            records=records,
            metadata=_with_catalog_task(metadata, entry),
        )
        validate_loaded(loaded, entry)
        return loaded


class HuggingFaceDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        entry = catalog_entry(spec.name)
        spec = apply_resolved_split(spec, entry)
        records, metadata = load_with_fallback(
            spec,
            entry=entry,
            map_row=mapper_for(spec.name),
            prefer="hf",
        )
        loaded = LoadedDataset(
            spec=spec,
            records=records,
            metadata=_with_catalog_task(metadata, entry),
        )
        validate_loaded(loaded, entry)
        return loaded


class MappedDatasetLoader:
    catalog_name: str

    def map_row(self, row: dict[str, Any]) -> DatasetRecord | None:
        raise NotImplementedError

    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if spec.max_samples is not None and spec.max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        entry = catalog_entry(self.catalog_name)
        spec = apply_resolved_split(spec, entry)
        records, metadata = load_with_fallback(
            spec,
            entry=entry,
            map_row=self.map_row,
            prefer="jsonl" if spec.path else "hf",
        )
        loaded = LoadedDataset(
            spec=spec,
            records=records,
            metadata=_with_catalog_task(metadata, entry),
        )
        validate_loaded(loaded, entry)
        return loaded
