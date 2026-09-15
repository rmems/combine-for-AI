"""Shared dataset record types used by loaders and the offline cache."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str = "validation"
    source: str = "jsonl"
    path: str | None = None
    hf_id: str | None = None
    hf_subset: str | None = None
    max_samples: int | None = None
    revision: str | None = None
    cache_mode: str | None = None
    cache_root: str | None = None
    upstream_license: str | None = None

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples < 0:
            raise ValueError("max_samples must be non-negative")

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
            revision=raw.get("revision"),
            cache_mode=raw.get("cache_mode"),
            cache_root=raw.get("cache_root"),
            upstream_license=raw.get("upstream_license"),
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


def validate_dataset_record(record: DatasetRecord) -> None:
    _require_text_field(record.prompt, "prompt")
    if record.reference is not None and not isinstance(record.reference, str):
        raise ValueError("dataset record reference must be a string")
    _require_choice_fields(record)


def _require_text_field(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"dataset record {field} must be a non-empty string")


def _require_choice_fields(record: DatasetRecord) -> None:
    if record.choices is None:
        if record.answer_index is not None:
            raise ValueError("answer_index requires choices")
        return
    _require_choice_list(record)


def _require_choice_list(record: DatasetRecord) -> None:
    if not isinstance(record.choices, list) or not all(
        isinstance(choice, str) for choice in record.choices
    ):
        raise ValueError("dataset record choices must be a list of strings")
    _require_answer_index(record)


def _require_answer_index(record: DatasetRecord) -> None:
    if not isinstance(record.answer_index, int):
        raise ValueError("multiple-choice records require an integer answer_index")
    if record.choices is None or record.answer_index < 0 or record.answer_index >= len(
        record.choices
    ):
        raise ValueError("answer_index is out of range")
