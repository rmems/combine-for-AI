from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, Never

from benchmarks.dataset_types import (
    CatalogEntry,
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
    TaskKind,
)

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

REPO_ROOT = Path(__file__).resolve().parents[1]
ROW_MAPPERS: dict[str, Callable[[dict[str, Any]], DatasetRecord | None]] = {}

Prefer = Literal["auto", "jsonl", "hf"]

CATALOG: dict[str, CatalogEntry] = {}


def _register_entry(entry: CatalogEntry) -> CatalogEntry:
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
    ROW_MAPPERS[name] = mapper
    entry = CATALOG.get(name)
    if entry is None:
        return
    for alias in entry.aliases:
        ROW_MAPPERS[alias] = mapper
    ROW_MAPPERS[entry.name] = mapper


def catalog_entry(name: str) -> CatalogEntry | None:
    return CATALOG.get(name)


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
    return default_cache_dir()


def canonical_record(row: dict[str, Any]) -> DatasetRecord | None:
    if "prompt" not in row or row["prompt"] is None:
        return None
    choices = row.get("choices")
    if choices is not None:
        choices = [str(choice) for choice in choices]
    answer_index = row.get("answer_index")
    if answer_index is not None:
        answer_index = int(answer_index)
    reference = row.get("reference")
    return DatasetRecord(
        prompt=str(row["prompt"]),
        reference=None if reference is None else str(reference),
        choices=choices,
        answer_index=answer_index,
    )


def row_as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    raise TypeError(f"cannot convert dataset row of type {type(row)!r} to dict")


def apply_max_samples(
    records: list[DatasetRecord], max_samples: int | None
) -> list[DatasetRecord]:
    if max_samples is None:
        return records
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    return records[:max_samples]


def iter_jsonl_payloads(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid json on line {line_number} in {path}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"jsonl object on line {line_number} in {path} must be a mapping"
                )
            yield line_number, payload


def require_canonical_record(row: dict[str, Any]) -> DatasetRecord:
    record = canonical_record(row)
    if record is None:
        raise ValueError("missing required field 'prompt'")
    return record


def mapper_for(name: str) -> Callable[[dict[str, Any]], DatasetRecord | None]:
    return ROW_MAPPERS.get(name, require_canonical_record)


def _map_jsonl_row(
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    payload: dict[str, Any],
    *,
    line_number: int,
    path: Path,
) -> DatasetRecord | None:
    try:
        return map_row(payload)
    except ValueError as exc:
        raise ValueError(f"{exc} on line {line_number} in {path}") from exc


def _normalized_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None:
        return None
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    return max_samples


def records_from_jsonl(
    path: Path,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    max_samples: int | None,
) -> list[DatasetRecord]:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    limit = _normalized_max_samples(max_samples)
    if limit == 0:
        return []

    records: list[DatasetRecord] = []
    for line_number, payload in iter_jsonl_payloads(path):
        record = _map_jsonl_row(
            map_row, payload, line_number=line_number, path=path
        )
        if record is None:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def write_normalized_jsonl(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_payload(), ensure_ascii=False))
            handle.write("\n")


def normalized_cache_path(
    cache_dir: Path, name: str, hf_id: str, subset: str | None, split: str
) -> Path:
    key = f"{hf_id}\0{subset or ''}\0{split}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / "normalized" / name / split / f"{digest}.jsonl"


def resolve_hf_split(spec: DatasetSpec, entry: CatalogEntry | None) -> str:
    if entry is None:
        return spec.split
    if spec.split == "validation" and entry.default_split != "validation":
        return entry.default_split
    return spec.split


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


def records_from_hf(
    *,
    name: str,
    hf_id: str,
    hf_subset: str | None,
    split: str,
    map_row: Callable[[dict[str, Any]], DatasetRecord | None],
    max_samples: int | None,
    cache_dir: Path,
) -> tuple[list[DatasetRecord], dict]:
    if not hf_id:
        raise ValueError(f"hf dataset '{name}' is missing hf_id")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")

    normalized_path = normalized_cache_path(cache_dir, name, hf_id, hf_subset, split)
    if normalized_path.exists():
        records = records_from_jsonl(normalized_path, canonical_record, max_samples)
        return records, {
            "source": "hf_cache",
            "path": str(normalized_path),
            "hf_id": hf_id,
            "hf_subset": hf_subset,
            "split": split,
        }

    mapped: list[DatasetRecord] = []
    skipped = 0
    for row in _load_hf_rows(hf_id, hf_subset, split, cache_dir / "hf"):
        record = map_row(row_as_dict(row))
        if record is None:
            skipped += 1
            continue
        mapped.append(record)

    write_normalized_jsonl(normalized_path, mapped)
    records = apply_max_samples(mapped, max_samples)
    return records, {
        "source": "hf",
        "hf_id": hf_id,
        "hf_subset": hf_subset,
        "split": split,
        "skipped": skipped,
        "cached_path": str(normalized_path),
    }


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
