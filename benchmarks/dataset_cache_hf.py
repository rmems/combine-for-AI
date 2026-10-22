"""Hugging Face dataset fetch used to populate the offline cache."""

from __future__ import annotations

from typing import Any

from benchmarks.dataset_cache_models import (
    FetchResult,
    normalize_license,
    requested_revision,
)
from benchmarks.dataset_hf_pin import (
    dataset_source_uri,
    is_commit_sha,
    resolve_dataset_revision,
)
from benchmarks.dataset_jsonl import row_as_dict
from benchmarks.dataset_support import catalog_entry, mapper_for
from benchmarks.dataset_types import (
    DatasetRecord,
    DatasetSpec,
    TaskKind,
    validate_dataset_record,
)

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None


def resolve_huggingface_revision(spec: DatasetSpec) -> str:
    return resolve_dataset_revision(spec)


def huggingface_source_uri(
    spec: DatasetSpec,
    *,
    resolved_revision: str | None = None,
) -> str:
    if not spec.hf_id:
        raise ValueError(f"hf dataset {spec.name!r} is missing hf_id")
    revision = resolved_revision or requested_revision(spec)
    return dataset_source_uri(spec.hf_id, spec.hf_subset, revision)


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
    if not is_commit_sha(revision):
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


def _empty_prompt_allowed(spec: DatasetSpec) -> bool:
    entry = catalog_entry(spec.name)
    return entry is not None and entry.task is TaskKind.LANGUAGE_MODELING


def _mapped_hf_row(
    row: Any, spec: DatasetSpec, *, allow_empty_prompt: bool
) -> DatasetRecord | None:
    mapped = mapper_for(spec.name)(row_as_dict(row))
    if mapped is None:
        return None
    validate_dataset_record(mapped, allow_empty_prompt=allow_empty_prompt)
    return mapped


def _collect_hf_records(
    dataset: Any, spec: DatasetSpec, allow_empty_prompt: bool
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for row in dataset:
        mapped = _mapped_hf_row(
            row, spec, allow_empty_prompt=allow_empty_prompt
        )
        if mapped is not None:
            records.append(mapped)
    return records


def fetch_huggingface_dataset(spec: DatasetSpec) -> FetchResult:
    resolved_revision = resolve_huggingface_revision(spec)
    if not spec.hf_id:
        raise ValueError(f"hf dataset '{spec.name}' is missing hf_id")
    dataset = load_hf_split(
        spec.hf_id,
        spec.hf_subset,
        spec.generic_split(),
        resolved_revision,
    )
    allow_empty_prompt = _empty_prompt_allowed(spec)
    records = _collect_hf_records(dataset, spec, allow_empty_prompt)
    return FetchResult(
        records=records,
        source_uri=huggingface_source_uri(spec, resolved_revision=resolved_revision),
        resolved_revision=resolved_revision,
        upstream_license=normalize_license(hf_license(spec, dataset)),
        allow_empty_prompt=allow_empty_prompt,
    )
