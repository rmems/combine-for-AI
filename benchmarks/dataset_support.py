from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Never

from benchmarks.dataset_jsonl import records_from_jsonl, require_canonical_record
from benchmarks.dataset_mapped_cache import records_from_hf
from benchmarks.dataset_types import (
    CatalogEntry,
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
    TaskKind,
)

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
        hf_id="wikitext",
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
        hf_id="piqa",
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
    if entry is None:
        return spec.split
    if spec.split == "validation" and entry.default_split != "validation":
        return entry.default_split
    return spec.split


def resolve_jsonl_path(
    spec: DatasetSpec, entry: CatalogEntry | None
) -> Path | None:
    if spec.path:
        path = Path(spec.path)
        if path.exists():
            return path
        repo_path = REPO_ROOT / spec.path
        if repo_path.exists():
            return repo_path
        return path
    if entry is None:
        return None
    sample = sample_path_for(entry)
    if sample.exists():
        return sample
    return None


def _jsonl_metadata(path: Path, *, fallback: str | None = None) -> dict:
    metadata: dict = {"source": "jsonl", "path": str(path)}
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
    return records, _jsonl_metadata(path)


def _fallback_jsonl(
    spec: DatasetSpec,
    jsonl_path: Path | None,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    exc: Exception,
) -> tuple[list[DatasetRecord], dict]:
    fallback_path = jsonl_path if jsonl_path is not None and jsonl_path.exists() else None
    if fallback_path is None:
        raise exc
    records = records_from_jsonl(fallback_path, map_row, spec.max_samples)
    return records, _jsonl_metadata(fallback_path, fallback=str(exc))


def _hf_request(
    spec: DatasetSpec, entry: CatalogEntry | None
) -> tuple[str | None, str | None, str, Path]:
    hf_id = spec.hf_id or (entry.hf_id if entry else None)
    hf_subset = spec.hf_subset
    if hf_subset is None and entry is not None:
        hf_subset = entry.hf_subset
    return hf_id, hf_subset, resolve_hf_split(spec, entry), resolve_cache_dir(spec)


def load_with_fallback(
    spec: DatasetSpec,
    *,
    entry: CatalogEntry | None,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    prefer: Prefer,
) -> tuple[list[DatasetRecord], dict]:
    jsonl_path = resolve_jsonl_path(spec, entry)
    if prefer == "jsonl":
        return _load_jsonl_or_raise(
            jsonl_path,
            spec,
            map_row,
            missing_message=f"jsonl dataset '{spec.name}' is missing a path",
        )
    if prefer == "auto" and spec.path:
        return _load_jsonl_or_raise(
            jsonl_path,
            spec,
            map_row,
            missing_message=f"dataset file not found: {spec.path}",
        )

    hf_id, hf_subset, split, cache_dir = _hf_request(spec, entry)
    if not hf_id:
        return _fallback_jsonl(
            spec,
            jsonl_path,
            map_row,
            ValueError(f"hf dataset '{spec.name}' is missing hf_id"),
        )

    try:
        return records_from_hf(
            name=spec.name,
            hf_id=hf_id,
            hf_subset=hf_subset,
            split=split,
            map_row=map_row,
            max_samples=spec.max_samples,
            cache_dir=cache_dir,
        )
    except (ImportError, OSError, FileNotFoundError, ConnectionError, RuntimeError) as exc:
        return _fallback_jsonl(spec, jsonl_path, map_row, exc)


def _assert_never(value: Never) -> Never:
    raise ValueError(f"unhandled task kind: {value!r}")


def _validate_multiple_choice(record: DatasetRecord, label: str) -> None:
    if not record.choices or len(record.choices) < 2:
        raise ValueError(f"{label} must include at least two choices")
    if any(not str(choice).strip() for choice in record.choices):
        raise ValueError(f"{label} has an empty choice")
    if record.answer_index is None:
        raise ValueError(f"{label} is missing answer_index")
    if not 0 <= record.answer_index < len(record.choices):
        raise ValueError(
            f"{label} answer_index {record.answer_index} is out of range "
            f"for {len(record.choices)} choices"
        )


def _validate_reference(record: DatasetRecord, label: str) -> None:
    if record.reference is None or not str(record.reference).strip():
        raise ValueError(f"{label} is missing a reference")


def _validate_generic(record: DatasetRecord, label: str) -> None:
    if record.choices is not None or record.answer_index is not None:
        _validate_multiple_choice(record, label)


def _validate_record(record: DatasetRecord, task: TaskKind, label: str) -> None:
    if not str(record.prompt).strip():
        raise ValueError(f"{label} has an empty prompt")
    match task:
        case TaskKind.CLOZE | TaskKind.MATH:
            _validate_reference(record, label)
        case TaskKind.MULTIPLE_CHOICE:
            _validate_multiple_choice(record, label)
        case TaskKind.LANGUAGE_MODELING:
            return
        case TaskKind.GENERIC:
            _validate_generic(record, label)
        case _:
            _assert_never(task)


def _required_min_samples(spec: DatasetSpec, entry: CatalogEntry | None) -> int:
    if spec.max_samples == 0:
        return 0
    if spec.min_samples is not None:
        return spec.min_samples
    if entry is None:
        return 0
    return entry.min_samples


def validate_loaded(loaded: LoadedDataset, entry: CatalogEntry | None) -> None:
    spec = loaded.spec
    if spec.max_samples is not None and spec.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if spec.min_samples is not None and spec.min_samples < 0:
        raise ValueError("min_samples must be non-negative")

    task = entry.task if entry is not None else TaskKind.GENERIC
    min_samples = _required_min_samples(spec, entry)
    if len(loaded.records) < min_samples:
        raise ValueError(
            f"{spec.name}: expected at least {min_samples} samples, "
            f"got {len(loaded.records)}"
        )
    for index, record in enumerate(loaded.records):
        _validate_record(record, task, f"{spec.name} record {index}")


class JsonlDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if not spec.path:
            raise ValueError(f"jsonl dataset '{spec.name}' is missing a path")
        entry = catalog_entry(spec.name)
        records, metadata = load_with_fallback(
            spec,
            entry=entry,
            map_row=mapper_for(spec.name),
            prefer="jsonl",
        )
        loaded = LoadedDataset(spec=spec, records=records, metadata=metadata)
        validate_loaded(loaded, entry)
        return loaded


class HuggingFaceDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        entry = catalog_entry(spec.name)
        records, metadata = load_with_fallback(
            spec,
            entry=entry,
            map_row=mapper_for(spec.name),
            prefer="hf",
        )
        loaded = LoadedDataset(spec=spec, records=records, metadata=metadata)
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
        records, metadata = load_with_fallback(
            spec,
            entry=entry,
            map_row=self.map_row,
            prefer="auto",
        )
        loaded = LoadedDataset(spec=spec, records=records, metadata=metadata)
        validate_loaded(loaded, entry)
        return loaded


def multiple_choice_accuracy(
    predictions: list[int], answer_indices: list[int]
) -> float:
    if len(predictions) != len(answer_indices):
        raise ValueError("predictions and answer_indices must have the same length")
    if not predictions:
        return 0.0
    correct = sum(
        int(pred == answer)
        for pred, answer in zip(predictions, answer_indices, strict=True)
    )
    return correct / len(predictions)
