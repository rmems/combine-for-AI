"""Resolve Hugging Face dataset revisions to immutable commits."""

from __future__ import annotations

import re

from benchmarks.dataset_cache_models import requested_revision
from benchmarks.dataset_types import DatasetSpec

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


_COMMIT_SHA = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)


def is_commit_sha(revision: str | None) -> bool:
    return revision is not None and _COMMIT_SHA.fullmatch(revision) is not None


def dataset_source_uri(hf_id: str, subset: str | None, revision: str) -> str:
    uri = f"hf://datasets/{hf_id}"
    if subset:
        uri += f"/{subset}"
    return f"{uri}@{revision}"


def resolve_dataset_revision(spec: DatasetSpec) -> str:
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
    if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError(
            f"Hugging Face returned an invalid commit SHA for {spec.hf_id!r}: "
            f"{resolved!r}"
        )
    return resolved
