"""Offline-first dataset cache with machine-readable license manifests.

Cache keys include dataset name, configuration, split, revision, and the
loader schema version so a source revision always maps to one immutable
directory. Offline mode never calls a network client.
"""

from __future__ import annotations

from benchmarks.dataset_cache_hf import (
    fetch_huggingface_dataset,
    huggingface_source_uri,
    resolve_huggingface_revision,
)
from benchmarks.dataset_cache_models import (
    DEFAULT_CACHE_MODE,
    DEFAULT_HF_REVISION,
    LICENSE_SCOPE_DATASET_SOURCE,
    LOADER_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    QUARANTINE_DIRNAME,
    RECORDS_FILENAME,
    UNKNOWN_LICENSE,
    UNSPECIFIED_REVISION,
    CacheKey,
    CacheLoad,
    CacheManifest,
    CacheMissError,
    CacheMode,
    CacheValidationError,
    DatasetCacheError,
    DatasetFetchFn,
    FetchResult,
    cache_key_for,
    checksum_bytes,
    default_cache_root,
    deserialize_record,
    encode_records,
    jsonl_provenance_metadata,
    normalize_license,
    parse_cache_mode,
    provenance_from_metadata,
    requested_revision,
    serialize_record,
    utc_timestamp,
)
from benchmarks.dataset_cache_store import DatasetCache

__all__ = [
    "DEFAULT_CACHE_MODE",
    "DEFAULT_HF_REVISION",
    "LICENSE_SCOPE_DATASET_SOURCE",
    "LOADER_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "QUARANTINE_DIRNAME",
    "RECORDS_FILENAME",
    "UNKNOWN_LICENSE",
    "UNSPECIFIED_REVISION",
    "CacheKey",
    "CacheLoad",
    "CacheManifest",
    "CacheMissError",
    "CacheMode",
    "CacheValidationError",
    "DatasetCache",
    "DatasetCacheError",
    "DatasetFetchFn",
    "FetchResult",
    "cache_key_for",
    "checksum_bytes",
    "default_cache_root",
    "deserialize_record",
    "encode_records",
    "fetch_huggingface_dataset",
    "huggingface_source_uri",
    "jsonl_provenance_metadata",
    "normalize_license",
    "parse_cache_mode",
    "provenance_from_metadata",
    "requested_revision",
    "resolve_huggingface_revision",
    "serialize_record",
    "utc_timestamp",
]
