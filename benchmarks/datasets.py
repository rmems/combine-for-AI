from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from benchmarks.dataset_cache import (
    DatasetCache,
    DatasetFetchFn,
    fetch_huggingface_dataset,
    jsonl_provenance_metadata,
    normalize_license,
)
from benchmarks.dataset_types import (
    DatasetRecord,
    DatasetSpec,
    LoadedDataset,
    validate_dataset_record,
)

__all__ = [
    "DatasetSpec",
    "DatasetRecord",
    "LoadedDataset",
    "DatasetLoader",
    "JsonlDatasetLoader",
    "HuggingFaceDatasetLoader",
    "DatasetRegistry",
    "default_dataset_registry",
]


class DatasetLoader(Protocol):
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        ...


class JsonlDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        path = _jsonl_path(spec)
        records = _read_jsonl_records(path, spec.max_samples)
        return LoadedDataset(
            spec=spec,
            records=records,
            metadata=jsonl_provenance_metadata(spec, path, len(records)),
        )


def _jsonl_path(spec: DatasetSpec) -> Path:
    if not spec.path:
        raise ValueError(f"jsonl dataset '{spec.name}' is missing a path")
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    return path


def _read_jsonl_records(path: Path, max_samples: int | None) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _jsonl_record(path, line_number, line, max_samples, len(records))
            if record is None:
                continue
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def _jsonl_record(
    path: Path,
    line_number: int,
    line: str,
    max_samples: int | None,
    loaded: int,
) -> DatasetRecord | None:
    line = line.strip()
    if not line:
        return None
    if max_samples is not None and loaded >= max_samples:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json on line {line_number} in {path}") from exc
    record = DatasetRecord(
        prompt=payload["prompt"],
        reference=payload.get("reference"),
        choices=payload.get("choices"),
        answer_index=payload.get("answer_index"),
    )
    validate_dataset_record(record)
    return record


class HuggingFaceDatasetLoader:
    def __init__(self, fetch: DatasetFetchFn | None = None) -> None:
        self._fetch = fetch or fetch_huggingface_dataset

    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if not spec.hf_id:
            raise ValueError(f"hf dataset '{spec.name}' is missing hf_id")

        cache = DatasetCache.for_spec(spec)
        loaded = cache.load(spec, fetch=self._fetch)
        records = loaded.records
        if spec.max_samples is not None:
            records = records[: spec.max_samples]
        metadata = loaded.metadata(source="hf")
        metadata["hf_id"] = spec.hf_id
        metadata["hf_subset"] = spec.hf_subset
        if spec.upstream_license:
            metadata["dataset_upstream_license"] = normalize_license(spec.upstream_license)
        return LoadedDataset(spec=spec, records=records, metadata=metadata)


class DatasetRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}

    def register(self, source: str, loader: DatasetLoader) -> None:
        self._loaders[source] = loader

    def loader_for(self, source: str) -> DatasetLoader:
        if source not in self._loaders:
            raise ValueError(f"dataset source '{source}' is not registered")
        return self._loaders[source]

    def available_sources(self) -> Iterable[str]:
        return sorted(self._loaders.keys())


def default_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register("jsonl", JsonlDatasetLoader())
    registry.register("hf", HuggingFaceDatasetLoader())

    # Imported here to avoid a cycle: lambada.loader imports DatasetSpec from
    # this module, and this registry is the only caller of LAMBADALoader.
    try:
        from benchmarks.lambada.loader import LAMBADALoader

        registry.register("lambada", LAMBADALoader())
    except ImportError:
        pass

    return registry
