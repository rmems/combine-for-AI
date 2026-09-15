from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class TaskKind(Enum):
    CLOZE = "cloze"
    MULTIPLE_CHOICE = "multiple_choice"
    LANGUAGE_MODELING = "language_modeling"
    MATH = "math"
    GENERIC = "generic"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str = "validation"
    source: str = "jsonl"
    path: str | None = None
    hf_id: str | None = None
    hf_subset: str | None = None
    max_samples: int | None = None
    min_samples: int | None = None
    cache_dir: str | None = None

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
            min_samples=raw.get("min_samples"),
            cache_dir=raw.get("cache_dir"),
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

    def to_payload(self) -> dict:
        payload: dict = {"prompt": self.prompt}
        if self.reference is not None:
            payload["reference"] = self.reference
        if self.choices is not None:
            payload["choices"] = self.choices
        if self.answer_index is not None:
            payload["answer_index"] = self.answer_index
        return payload


@dataclass(frozen=True)
class LoadedDataset:
    spec: DatasetSpec
    records: list[DatasetRecord]
    metadata: dict


class DatasetLoader(Protocol):
    def load(self, spec: DatasetSpec) -> LoadedDataset:
        ...


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    task: TaskKind
    hf_id: str
    sample_relpath: str
    hf_subset: str | None = None
    default_split: str = "validation"
    min_samples: int = 1
    aliases: tuple[str, ...] = ()
