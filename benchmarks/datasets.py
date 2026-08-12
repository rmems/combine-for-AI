from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str = "validation"
    source: str = "jsonl"
    path: str | None = None
    hf_id: str | None = None
    hf_subset: str | None = None
    max_samples: int | None = None

    @staticmethod
    def from_dict(raw: dict) -> "DatasetSpec":
        return DatasetSpec(
            name=raw["name"],
            split=raw.get("split", "validation"),
            source=raw.get("source", "jsonl"),
            path=raw.get("path"),
            hf_id=raw.get("hf_id"),
            hf_subset=raw.get("hf_subset"),
            max_samples=raw.get("max_samples"),
        )


@dataclass(frozen=True)
class DatasetRecord:
    prompt: str
    reference: str | None = None
    choices: list[str] | None = None
    answer_index: int | None = None

    @property
    def is_multiple_choice(self) -> bool:
        return self.choices is not None and self.answer_index is not None


@dataclass(frozen=True)
class LoadedDataset:
    spec: DatasetSpec
    records: list[DatasetRecord]
    metadata: dict


class DatasetLoader(Protocol):
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        ...


class JsonlDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if not spec.path:
            raise ValueError(f"jsonl dataset '{spec.name}' is missing a path")

        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"dataset file not found: {path}")

        records: list[DatasetRecord] = []
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

                record = DatasetRecord(
                    prompt=payload["prompt"],
                    reference=payload.get("reference"),
                    choices=payload.get("choices"),
                    answer_index=payload.get("answer_index"),
                )
                records.append(record)

                if spec.max_samples and len(records) >= spec.max_samples:
                    break

        return LoadedDataset(
            spec=spec,
            records=records,
            metadata={"source": "jsonl", "path": str(path)},
        )


class HuggingFaceDatasetLoader:
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if not spec.hf_id:
            raise ValueError(f"hf dataset '{spec.name}' is missing hf_id")

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for Hugging Face sources; "
                "install with `pip install datasets`"
            ) from exc

        dataset = load_dataset(spec.hf_id, spec.hf_subset, split=spec.split)
        records: list[DatasetRecord] = []
        for row in dataset:
            record = DatasetRecord(
                prompt=row["prompt"],
                reference=row.get("reference"),
                choices=row.get("choices"),
                answer_index=row.get("answer_index"),
            )
            records.append(record)
            if spec.max_samples and len(records) >= spec.max_samples:
                break

        return LoadedDataset(
            spec=spec,
            records=records,
            metadata={"source": "hf", "hf_id": spec.hf_id, "hf_subset": spec.hf_subset},
        )


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
    
    # Try to register LAMBADA loader if available
    try:
        from benchmarks.lambada.loader import LAMBADALoader
        registry.register("lambada", LAMBADALoader())
    except ImportError:
        # LAMBADA loader not available
        pass
    
    return registry
