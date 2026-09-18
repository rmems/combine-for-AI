"""Cache contracts: keys, manifests, licenses, and record encoding."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from benchmarks.dataset_types import DatasetRecord, DatasetSpec, validate_dataset_record

LOADER_SCHEMA_VERSION = "1"
UNKNOWN_LICENSE = "unknown"
LICENSE_SCOPE_DATASET_SOURCE = "dataset_source_not_repository"
UNSPECIFIED_REVISION = "unspecified"
DEFAULT_HF_REVISION = "main"
RECORDS_FILENAME = "records.jsonl"
MANIFEST_FILENAME = "manifest.json"
QUARANTINE_DIRNAME = "quarantine"

# Explicit sentinel so a missing upstream license cannot be read as the
# repository license (this project may not even ship a LICENSE file).
DEFAULT_CACHE_MODE = "prefer-cache"


class CacheMode(str, Enum):
    ONLINE = "online"
    PREFER_CACHE = "prefer-cache"
    OFFLINE = "offline"


class DatasetCacheError(Exception):
    """Base error for cache misses and rejected entries."""

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        reason: str | None = None,
        quarantined_to: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.reason = reason
        self.quarantined_to = quarantined_to


class CacheMissError(DatasetCacheError):
    """Offline mode found no usable artifact for this cache key."""


class CacheValidationError(DatasetCacheError):
    """Cached bytes or schema cannot be used; original files are kept."""


@dataclass(frozen=True)
class CacheKey:
    dataset_name: str
    hf_id: str
    configuration: str
    split: str
    revision: str
    schema_version: str

    def digest(self) -> str:
        payload = {
            "configuration": self.configuration,
            "dataset_name": self.dataset_name,
            "hf_id": self.hf_id,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "split": self.split,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset_name": self.dataset_name,
            "hf_id": self.hf_id,
            "configuration": self.configuration,
            "split": self.split,
            "revision": self.revision,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class FetchResult:
    records: list[DatasetRecord]
    source_uri: str
    resolved_revision: str
    upstream_license: str = UNKNOWN_LICENSE


@dataclass(frozen=True)
class CacheManifest:
    cache_key: dict[str, str]
    cache_key_digest: str
    source_uri: str
    resolved_revision: str
    retrieved_at: str
    checksum_sha256: str
    row_count: int
    upstream_license: str
    license_scope: str
    loader_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "CacheManifest":
        return CacheManifest(
            cache_key=dict(raw["cache_key"]),
            cache_key_digest=str(raw["cache_key_digest"]),
            source_uri=str(raw["source_uri"]),
            resolved_revision=str(raw["resolved_revision"]),
            retrieved_at=str(raw["retrieved_at"]),
            checksum_sha256=str(raw["checksum_sha256"]),
            row_count=int(raw["row_count"]),
            upstream_license=str(raw.get("upstream_license") or UNKNOWN_LICENSE),
            license_scope=str(
                raw.get("license_scope") or LICENSE_SCOPE_DATASET_SOURCE
            ),
            loader_schema_version=str(raw["loader_schema_version"]),
        )


@dataclass(frozen=True)
class CacheLoad:
    records: list[DatasetRecord]
    manifest: CacheManifest
    cache_path: Path
    cache_hit: bool

    def metadata(self, *, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "cache_hit": self.cache_hit,
            "cache_path": str(self.cache_path),
            "cache_key_digest": self.manifest.cache_key_digest,
            "source_uri": self.manifest.source_uri,
            "resolved_revision": self.manifest.resolved_revision,
            "retrieved_at": self.manifest.retrieved_at,
            "checksum_sha256": self.manifest.checksum_sha256,
            "row_count": self.manifest.row_count,
            "dataset_upstream_license": self.manifest.upstream_license,
            "dataset_license_scope": self.manifest.license_scope,
            "loader_schema_version": self.manifest.loader_schema_version,
        }


DatasetFetchFn = Callable[[DatasetSpec], FetchResult]


def parse_cache_mode(value: str) -> CacheMode:
    normalized = value.strip().lower().replace("_", "-")
    for mode in CacheMode:
        if mode.value == normalized:
            return mode
    allowed = ", ".join(mode.value for mode in CacheMode)
    raise ValueError(f"unknown cache mode {value!r}; expected one of: {allowed}")


def default_cache_root() -> Path:
    override = os.environ.get("COMBINE_DATASET_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "combine-for-ai" / "datasets"


def requested_revision(spec: DatasetSpec) -> str:
    if spec.revision:
        return spec.revision
    if spec.hf_id:
        return DEFAULT_HF_REVISION
    return UNSPECIFIED_REVISION


def cache_key_for(spec: DatasetSpec, *, schema_version: str = LOADER_SCHEMA_VERSION) -> CacheKey:
    return CacheKey(
        dataset_name=spec.name,
        hf_id=spec.hf_id or "",
        configuration=spec.hf_subset or "",
        split=spec.split,
        revision=requested_revision(spec),
        schema_version=schema_version,
    )


def serialize_record(record: DatasetRecord) -> dict[str, Any]:
    return {
        "answer_index": record.answer_index,
        "choices": record.choices,
        "prompt": record.prompt,
        "reference": record.reference,
    }


def deserialize_record(raw: dict[str, Any]) -> DatasetRecord:
    record = DatasetRecord(
        prompt=raw["prompt"],
        reference=raw.get("reference"),
        choices=raw.get("choices"),
        answer_index=raw.get("answer_index"),
    )
    validate_dataset_record(record)
    return record


def encode_records(records: list[DatasetRecord]) -> bytes:
    for record in records:
        validate_dataset_record(record)
    lines = [
        json.dumps(serialize_record(record), sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    body = "\n".join(lines)
    if body:
        body += "\n"
    return body.encode("utf-8")


def checksum_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_license(value: str | None) -> str:
    if value is None:
        return UNKNOWN_LICENSE
    stripped = value.strip()
    if not stripped:
        return UNKNOWN_LICENSE
    return stripped


def provenance_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Flat run-record fields that cannot be mistaken for a repository license."""
    return {
        "dataset_upstream_license": metadata.get(
            "dataset_upstream_license", UNKNOWN_LICENSE
        ),
        "dataset_license_scope": metadata.get(
            "dataset_license_scope", LICENSE_SCOPE_DATASET_SOURCE
        ),
        "dataset_source_uri": metadata.get("source_uri") or metadata.get("path"),
        "dataset_resolved_revision": metadata.get("resolved_revision"),
        "dataset_cache_key": metadata.get("cache_key_digest"),
    }


def jsonl_provenance_metadata(spec: DatasetSpec, path: Path, row_count: int) -> dict[str, Any]:
    return {
        "source": "jsonl",
        "path": str(path),
        "source_uri": path.resolve().as_uri(),
        "resolved_revision": "local",
        "row_count": row_count,
        "dataset_upstream_license": normalize_license(spec.upstream_license),
        "dataset_license_scope": LICENSE_SCOPE_DATASET_SOURCE,
        "cache_hit": False,
        "cache_key_digest": None,
        "loader_schema_version": LOADER_SCHEMA_VERSION,
    }
