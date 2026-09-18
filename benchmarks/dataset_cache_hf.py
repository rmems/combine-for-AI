"""Hugging Face dataset fetch used to populate the offline cache."""

from __future__ import annotations

from typing import Any

from benchmarks.dataset_cache_models import (
    FetchResult,
    normalize_license,
    requested_revision,
)
from benchmarks.dataset_types import DatasetRecord, DatasetSpec, validate_dataset_record

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None


def huggingface_source_uri(spec: DatasetSpec) -> str:
    if not spec.hf_id:
        raise ValueError(f"hf dataset {spec.name!r} is missing hf_id")
    uri = f"hf://datasets/{spec.hf_id}"
    if spec.hf_subset:
        uri += f"/{spec.hf_subset}"
    return f"{uri}@{requested_revision(spec)}"


def load_hf_split(spec: DatasetSpec):
    if hf_load_dataset is None:
        raise ImportError(
            "datasets is required for Hugging Face sources; "
            "install with `pip install datasets`"
        )
    if not spec.hf_id:
        raise ValueError(f"hf dataset '{spec.name}' is missing hf_id")
    return hf_load_dataset(
        spec.hf_id,
        spec.hf_subset,
        split=spec.split,
        revision=requested_revision(spec),
    )


def record_from_row(row: dict[str, Any]) -> DatasetRecord:
    record = DatasetRecord(
        prompt=row["prompt"],
        reference=row.get("reference"),
        choices=row.get("choices"),
        answer_index=row.get("answer_index"),
    )
    validate_dataset_record(record)
    return record


def hf_license(spec: DatasetSpec, dataset: Any) -> str | None:
    if spec.upstream_license:
        return spec.upstream_license
    info = getattr(dataset, "info", None)
    if info is None:
        return None
    license_id = getattr(info, "license", None) or getattr(info, "license_name", None)
    return license_id if license_id else None


def fetch_huggingface_dataset(spec: DatasetSpec) -> FetchResult:
    dataset = load_hf_split(spec)
    records = [record_from_row(row) for row in dataset]
    return FetchResult(
        records=records,
        source_uri=huggingface_source_uri(spec),
        resolved_revision=requested_revision(spec),
        upstream_license=normalize_license(hf_license(spec, dataset)),
    )
