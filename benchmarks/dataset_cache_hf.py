"""Hugging Face dataset fetch used to populate the offline cache."""

from __future__ import annotations

import re
from typing import Any

from benchmarks.dataset_cache_models import (
    FetchResult,
    normalize_license,
    requested_revision,
)
from benchmarks.dataset_jsonl import row_as_dict
from benchmarks.dataset_support import mapper_for
from benchmarks.dataset_types import DatasetRecord, DatasetSpec, validate_dataset_record

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)


def resolve_huggingface_revision(spec: DatasetSpec) -> str:
    revision = requested_revision(spec)
    if _COMMIT_SHA.fullmatch(revision):
        return revision
    if HfApi is None:
        raise ImportError(
            "huggingface_hub is required to resolve Hugging Face revisions; "
            "install with `pip install datasets`"
        )
    if not spec.hf_id:
        raise ValueError(f"hf dataset {spec.name!r} is missing hf_id")
    resolved = HfApi().dataset_info(spec.hf_id, revision=revision).sha
    if not resolved or not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError(
            f"Hugging Face returned an invalid commit SHA for {spec.hf_id!r}: "
            f"{resolved!r}"
        )
    return resolved


def huggingface_source_uri(
    spec: DatasetSpec,
    *,
    resolved_revision: str | None = None,
) -> str:
    if not spec.hf_id:
        raise ValueError(f"hf dataset {spec.name!r} is missing hf_id")
    uri = f"hf://datasets/{spec.hf_id}"
    if spec.hf_subset:
        uri += f"/{spec.hf_subset}"
    revision = resolved_revision or requested_revision(spec)
    return f"{uri}@{revision}"


def load_hf_split(
    hf_id: str,
    subset: str | None,
    split: str,
    revision: str,
) -> Any:
    if hf_load_dataset is None:
        raise ImportError(
            "datasets is required for Hugging Face sources; "
            "install with `pip install datasets`"
        )
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(
            f"Hugging Face loads require a pinned commit SHA revision, got {revision!r}"
        )
    load = hf_load_dataset
    return load(
        path=hf_id,
        name=subset,
        split=split,
        revision=revision,
    )


def record_from_row(
    row: dict[str, Any], spec: DatasetSpec | None = None
) -> DatasetRecord:
    payload = row_as_dict(row)
    if spec is None:
        record = DatasetRecord(
            prompt=payload["prompt"],
            reference=payload.get("reference"),
            choices=payload.get("choices"),
            answer_index=payload.get("answer_index"),
        )
    else:
        mapped = mapper_for(spec.name)(payload)
        if mapped is None:
            raise ValueError(f"unrecognized {spec.name} Hugging Face row")
        record = mapped
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
    resolved_revision = resolve_huggingface_revision(spec)
    if not spec.hf_id:
        raise ValueError(f"hf dataset '{spec.name}' is missing hf_id")
    dataset = load_hf_split(
        spec.hf_id,
        spec.hf_subset,
        spec.split,
        resolved_revision,
    )
    records: list[DatasetRecord] = []
    for row in dataset:
        payload = row_as_dict(row)
        mapped = mapper_for(spec.name)(payload)
        if mapped is None:
            continue
        validate_dataset_record(mapped)
        records.append(mapped)
    return FetchResult(
        records=records,
        source_uri=huggingface_source_uri(spec, resolved_revision=resolved_revision),
        resolved_revision=resolved_revision,
        upstream_license=normalize_license(hf_license(spec, dataset)),
    )
