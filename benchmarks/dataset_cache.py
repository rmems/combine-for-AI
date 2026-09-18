"""Offline-first dataset cache with machine-readable license manifests.

Cache keys include dataset name, configuration, split, revision, and the
loader schema version so a source revision always maps to one immutable
directory. Offline mode never calls a network client.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Never

from benchmarks.dataset_cache_hf import fetch_huggingface_dataset, huggingface_source_uri
from benchmarks.dataset_cache_io import (
    destination_occupied_by_live_entry,
    is_directory,
    is_symlink,
    lstat_or_none,
    open_dir_nofollow,
    parse_manifest,
    read_in_dir,
    records_if_valid,
    write_cache_files,
    write_text_nofollow,
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
from benchmarks.dataset_types import DatasetSpec

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
    "serialize_record",
    "utc_timestamp",
]


class DatasetCache:
    def __init__(self, root: Path, mode: CacheMode) -> None:
        self.root = root
        self.mode = mode

    @staticmethod
    def for_spec(spec: DatasetSpec) -> "DatasetCache":
        mode_raw = (
            spec.cache_mode
            or os.environ.get("COMBINE_DATASET_CACHE_MODE")
            or DEFAULT_CACHE_MODE
        )
        root_raw = spec.cache_root
        root = Path(root_raw).expanduser() if root_raw else default_cache_root()
        return DatasetCache(root=root, mode=parse_cache_mode(mode_raw))

    def key_for(self, spec: DatasetSpec) -> CacheKey:
        return cache_key_for(spec)

    def entry_dir(self, key: CacheKey) -> Path:
        return self.root / key.digest()

    def load(self, spec: DatasetSpec, fetch: DatasetFetchFn) -> CacheLoad:
        key = self.key_for(spec)
        entry = self.entry_dir(key)
        existing = self._load_existing(entry, key)
        if existing is not None and self._use_existing_without_fetch():
            return existing
        if not self._may_fetch():
            raise CacheMissError(
                self._miss_message(spec, key, entry),
                path=entry,
                reason="missing-artifact",
            )
        return self._fetch_and_store(spec, key, entry, existing, fetch)

    def _load_existing(self, entry: Path, key: CacheKey) -> CacheLoad | None:
        if lstat_or_none(entry) is None:
            return None
        try:
            return self._read_valid(entry, key)
        except CacheValidationError as exc:
            quarantined = self._quarantine(entry, reason=exc.reason or str(exc))
            if not self._may_fetch():
                raise CacheValidationError(
                    f"Rejected cache entry {entry}: {exc.reason}. "
                    f"Quarantined to {quarantined} (original bytes were not deleted).",
                    path=entry,
                    reason=exc.reason,
                    quarantined_to=quarantined,
                ) from exc
            return None

    def _fetch_and_store(
        self,
        spec: DatasetSpec,
        key: CacheKey,
        entry: Path,
        existing: CacheLoad | None,
        fetch: DatasetFetchFn,
    ) -> CacheLoad:
        fetched = fetch(spec)
        payload = encode_records(fetched.records)
        checksum = checksum_bytes(payload)
        if existing is not None and _fetched_matches_manifest(existing.manifest, fetched, checksum):
            return existing
        if existing is not None:
            self._quarantine(
                entry,
                reason=(
                    "online fetch checksum or provenance differed from the cached "
                    f"entry (cached={existing.manifest.checksum_sha256}, "
                    f"fetched={checksum})"
                ),
            )
        return self._store(key, fetched, payload, checksum)

    def _use_existing_without_fetch(self) -> bool:
        match self.mode:
            case CacheMode.OFFLINE | CacheMode.PREFER_CACHE:
                return True
            case CacheMode.ONLINE:
                return False
            case _:
                unreachable: Never = self.mode
                raise ValueError(f"unhandled cache mode: {unreachable}")

    def _may_fetch(self) -> bool:
        match self.mode:
            case CacheMode.OFFLINE:
                return False
            case CacheMode.PREFER_CACHE:
                return True
            case CacheMode.ONLINE:
                return True
            case _:
                unreachable: Never = self.mode
                raise ValueError(f"unhandled cache mode: {unreachable}")

    def _read_valid(self, entry: Path, key: CacheKey) -> CacheLoad:
        dir_fd = open_dir_nofollow(entry)
        try:
            payload = read_in_dir(dir_fd, RECORDS_FILENAME, entry)
            manifest_bytes = read_in_dir(dir_fd, MANIFEST_FILENAME, entry)
        finally:
            os.close(dir_fd)
        manifest = parse_manifest(manifest_bytes, entry)
        records = records_if_valid(entry, key, manifest, payload)
        return CacheLoad(
            records=records,
            manifest=manifest,
            cache_path=entry,
            cache_hit=True,
        )

    def _store(
        self,
        key: CacheKey,
        fetched: FetchResult,
        payload: bytes,
        checksum: str,
    ) -> CacheLoad:
        self._ensure_writable_root()
        token = f"{os.getpid()}-{secrets.token_hex(8)}"
        staging = self.root / f".{key.digest()}.staging-{token}"
        staging.mkdir(parents=False)
        manifest = _manifest_for_store(key, fetched, checksum)
        live = self.entry_dir(key)
        try:
            write_cache_files(staging, payload, manifest)
            self._publish_staging(staging, live)
        except (OSError, CacheValidationError):
            if lstat_or_none(staging) is not None:
                self._quarantine(staging, reason="incomplete-store")
            raise
        return CacheLoad(
            records=list(fetched.records),
            manifest=manifest,
            cache_path=live,
            cache_hit=False,
        )

    def _publish_staging(self, staging: Path, live: Path) -> None:
        try:
            os.rename(staging, live)
        except OSError as exc:
            if not destination_occupied_by_live_entry(exc, live):
                raise
            if lstat_or_none(staging) is not None:
                self._quarantine(staging, reason="lost-publish-race")

    def _ensure_writable_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            st = self.root.stat()
        except OSError as exc:
            raise CacheValidationError(
                f"cache root is not usable: {self.root}",
                path=self.root,
                reason="cache-root-unusable",
            ) from exc
        if not is_directory(st):
            raise CacheValidationError(
                f"cache root is not a directory: {self.root}",
                path=self.root,
                reason="not-a-directory",
            )

    def _quarantine(self, entry: Path, *, reason: str) -> Path:
        dest_parent = self.root / QUARANTINE_DIRNAME
        self._ensure_real_directory(dest_parent)
        dest = self._unused_quarantine_path(dest_parent, entry.name)
        source = lstat_or_none(entry)
        if source is None:
            dest.mkdir(parents=True, exist_ok=True)
        elif is_symlink(source):
            # Move only the link inode. Never follow it into another tree, and
            # never write quarantine notes through the link target.
            dest.mkdir(parents=True, exist_ok=True)
            write_text_nofollow(dest / "rejected-symlink", os.readlink(entry) + "\n")
            entry.unlink()
        elif is_directory(source):
            os.rename(entry, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            os.rename(entry, dest / entry.name)
        note = {
            "reason": reason,
            "original_path": str(entry),
            "quarantined_at": utc_timestamp(),
        }
        write_text_nofollow(
            dest / "quarantine.json",
            json.dumps(note, indent=2, sort_keys=True) + "\n",
        )
        return dest

    def _unused_quarantine_path(self, dest_parent: Path, entry_name: str) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = dest_parent / f"{entry_name}-{stamp}"
        suffix = 1
        while lstat_or_none(dest) is not None:
            dest = dest_parent / f"{entry_name}-{stamp}-{suffix}"
            suffix += 1
        return dest

    def _ensure_real_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        st = path.lstat()
        if is_symlink(st) or not is_directory(st):
            raise CacheValidationError(
                f"cache path is not a real directory: {path}",
                path=path,
                reason="unsafe-symlink",
            )

    def _miss_message(self, spec: DatasetSpec, key: CacheKey, entry: Path) -> str:
        return (
            f"Offline cache miss for dataset {spec.name!r} "
            f"(configuration={key.configuration!r}, split={key.split!r}, "
            f"revision={key.revision!r}, schema={key.schema_version!r}). "
            f"No artifact at {entry}. Place {RECORDS_FILENAME} and "
            f"{MANIFEST_FILENAME} there, or rerun with cache mode "
            f"{CacheMode.PREFER_CACHE.value!r} or {CacheMode.ONLINE.value!r} "
            "to populate the cache. No network client was invoked."
        )


def _manifest_for_store(key: CacheKey, fetched: FetchResult, checksum: str) -> CacheManifest:
    return CacheManifest(
        cache_key=key.as_dict(),
        cache_key_digest=key.digest(),
        source_uri=fetched.source_uri,
        resolved_revision=fetched.resolved_revision,
        retrieved_at=utc_timestamp(),
        checksum_sha256=checksum,
        row_count=len(fetched.records),
        upstream_license=normalize_license(fetched.upstream_license),
        license_scope=LICENSE_SCOPE_DATASET_SOURCE,
        loader_schema_version=key.schema_version,
    )


def _fetched_matches_manifest(
    manifest: CacheManifest,
    fetched: FetchResult,
    checksum: str,
) -> bool:
    return (
        manifest.checksum_sha256 == checksum
        and manifest.resolved_revision == fetched.resolved_revision
        and manifest.source_uri == fetched.source_uri
        and manifest.upstream_license == normalize_license(fetched.upstream_license)
    )
