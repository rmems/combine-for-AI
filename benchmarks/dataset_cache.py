"""Offline-first dataset cache with machine-readable license manifests.

Cache keys include dataset name, configuration, split, revision, and the
loader schema version so a source revision always maps to one immutable
directory. Offline mode never calls a network client.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Never

from benchmarks.dataset_types import DatasetRecord, DatasetSpec, validate_dataset_record

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

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


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _is_symlink(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode)


def _is_directory(st: os.stat_result) -> bool:
    return stat.S_ISDIR(st.st_mode)


def _is_regular_file(st: os.stat_result) -> bool:
    return stat.S_ISREG(st.st_mode)


def _destination_occupied_by_live_entry(exc: OSError, live: Path) -> bool:
    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR}:
        return False
    st = _lstat_or_none(live)
    return st is not None and _is_directory(st) and not _is_symlink(st)


def _open_nofollow(path: Path, flags: int, mode: int = 0o644) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, mode)


def _write_bytes_nofollow(path: Path, payload: bytes) -> None:
    try:
        fd = _open_nofollow(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    except OSError as exc:
        raise CacheValidationError(
            f"cannot write {path}: {exc}",
            path=path,
            reason="unsafe-symlink",
        ) from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
    finally:
        if fd >= 0:
            os.close(fd)


def _write_text_nofollow(path: Path, text: str) -> None:
    _write_bytes_nofollow(path, text.encode("utf-8"))


def _dir_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_dir_nofollow(path: Path) -> int:
    try:
        return os.open(path, _dir_open_flags())
    except OSError as exc:
        raise CacheValidationError(
            f"cache entry is not a real directory: {path}",
            path=path,
            reason="unsafe-symlink",
        ) from exc


def _read_in_dir(dir_fd: int, name: str, entry: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError as exc:
        raise CacheValidationError(
            f"incomplete cache entry at {entry}",
            path=entry,
            reason="incomplete-entry",
        ) from exc
    except OSError as exc:
        raise CacheValidationError(
            f"cannot read {entry / name}: {exc}",
            path=entry,
            reason="unsafe-symlink",
        ) from exc
    try:
        st = os.fstat(fd)
        if not _is_regular_file(st):
            raise CacheValidationError(
                f"not a regular file: {entry / name}",
                path=entry,
                reason="not-a-regular-file",
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _write_in_dir(dir_fd: int, name: str, payload: bytes, entry: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, 0o644, dir_fd=dir_fd)
    except OSError as exc:
        raise CacheValidationError(
            f"cannot write {entry / name}: {exc}",
            path=entry,
            reason="unsafe-symlink",
        ) from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
    finally:
        if fd >= 0:
            os.close(fd)


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
        if _lstat_or_none(entry) is None:
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
        dir_fd = _open_dir_nofollow(entry)
        try:
            payload = _read_in_dir(dir_fd, RECORDS_FILENAME, entry)
            manifest_bytes = _read_in_dir(dir_fd, MANIFEST_FILENAME, entry)
        finally:
            os.close(dir_fd)
        manifest = _parse_manifest(manifest_bytes, entry)
        records = _records_if_valid(entry, key, manifest, payload)
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
            _write_cache_files(staging, payload, manifest)
            self._publish_staging(staging, live)
        except (OSError, CacheValidationError):
            if _lstat_or_none(staging) is not None:
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
            if not _destination_occupied_by_live_entry(exc, live):
                raise
            if _lstat_or_none(staging) is not None:
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
        if not _is_directory(st):
            raise CacheValidationError(
                f"cache root is not a directory: {self.root}",
                path=self.root,
                reason="not-a-directory",
            )

    def _quarantine(self, entry: Path, *, reason: str) -> Path:
        dest_parent = self.root / QUARANTINE_DIRNAME
        self._ensure_real_directory(dest_parent)
        dest = self._unused_quarantine_path(dest_parent, entry.name)
        source = _lstat_or_none(entry)
        if source is None:
            dest.mkdir(parents=True, exist_ok=True)
        elif _is_symlink(source):
            # Move only the link inode. Never follow it into another tree, and
            # never write quarantine notes through the link target.
            dest.mkdir(parents=True, exist_ok=True)
            _write_text_nofollow(dest / "rejected-symlink", os.readlink(entry) + "\n")
            entry.unlink()
        elif _is_directory(source):
            os.rename(entry, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            os.rename(entry, dest / entry.name)
        note = {
            "reason": reason,
            "original_path": str(entry),
            "quarantined_at": utc_timestamp(),
        }
        _write_text_nofollow(
            dest / "quarantine.json",
            json.dumps(note, indent=2, sort_keys=True) + "\n",
        )
        return dest

    def _unused_quarantine_path(self, dest_parent: Path, entry_name: str) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = dest_parent / f"{entry_name}-{stamp}"
        suffix = 1
        while _lstat_or_none(dest) is not None:
            dest = dest_parent / f"{entry_name}-{stamp}-{suffix}"
            suffix += 1
        return dest

    def _ensure_real_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        st = path.lstat()
        if _is_symlink(st) or not _is_directory(st):
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


def _write_cache_files(staging: Path, payload: bytes, manifest: CacheManifest) -> None:
    dir_fd = _open_dir_nofollow(staging)
    try:
        _write_in_dir(dir_fd, RECORDS_FILENAME, payload, staging)
        encoded = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        _write_in_dir(dir_fd, MANIFEST_FILENAME, encoded.encode("utf-8"), staging)
    finally:
        os.close(dir_fd)


def _parse_manifest(manifest_bytes: bytes, entry: Path) -> CacheManifest:
    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
        return CacheManifest.from_dict(raw_manifest)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise CacheValidationError(
            f"unreadable manifest at {entry / MANIFEST_FILENAME}: {exc}",
            path=entry,
            reason="unreadable-manifest",
        ) from exc


def _records_if_valid(
    entry: Path,
    key: CacheKey,
    manifest: CacheManifest,
    payload: bytes,
) -> list[DatasetRecord]:
    _require_checksum(entry, manifest, payload)
    _require_schema(entry, key, manifest)
    _require_cache_key(entry, key, manifest)
    records = _decode_records(payload, entry)
    if len(records) != manifest.row_count:
        raise CacheValidationError(
            f"row count mismatch at {entry / RECORDS_FILENAME}",
            path=entry,
            reason=(
                f"row-count-mismatch expected={manifest.row_count} actual={len(records)}"
            ),
        )
    return records


def _require_checksum(entry: Path, manifest: CacheManifest, payload: bytes) -> None:
    digest = checksum_bytes(payload)
    if digest != manifest.checksum_sha256:
        raise CacheValidationError(
            f"checksum mismatch at {entry / RECORDS_FILENAME}",
            path=entry,
            reason=(
                f"checksum-mismatch expected={manifest.checksum_sha256} actual={digest}"
            ),
        )


def _require_schema(entry: Path, key: CacheKey, manifest: CacheManifest) -> None:
    if manifest.loader_schema_version != key.schema_version:
        raise CacheValidationError(
            f"schema mismatch at {entry / MANIFEST_FILENAME}",
            path=entry,
            reason=(
                "stale-schema "
                f"entry={manifest.loader_schema_version} expected={key.schema_version}"
            ),
        )


def _require_cache_key(entry: Path, key: CacheKey, manifest: CacheManifest) -> None:
    if manifest.cache_key_digest != key.digest():
        raise CacheValidationError(
            f"cache key mismatch at {entry / MANIFEST_FILENAME}",
            path=entry,
            reason=(
                "cache-key-mismatch "
                f"entry={manifest.cache_key_digest} expected={key.digest()}"
            ),
        )


def _decode_records(payload: bytes, entry: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CacheValidationError(
            f"invalid utf-8 in {entry / RECORDS_FILENAME}",
            path=entry,
            reason="invalid-record encoding=utf-8",
        ) from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            records.append(deserialize_record(raw))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CacheValidationError(
                f"invalid record on line {line_number} in {entry / RECORDS_FILENAME}",
                path=entry,
                reason=f"invalid-record line={line_number}",
            ) from exc
    return records


def huggingface_source_uri(spec: DatasetSpec) -> str:
    if not spec.hf_id:
        raise ValueError(f"hf dataset {spec.name!r} is missing hf_id")
    uri = f"hf://datasets/{spec.hf_id}"
    if spec.hf_subset:
        uri += f"/{spec.hf_subset}"
    return f"{uri}@{requested_revision(spec)}"


def _load_hf_split(spec: DatasetSpec):
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


def _record_from_row(row: dict[str, Any]) -> DatasetRecord:
    record = DatasetRecord(
        prompt=row["prompt"],
        reference=row.get("reference"),
        choices=row.get("choices"),
        answer_index=row.get("answer_index"),
    )
    validate_dataset_record(record)
    return record


def _hf_license(spec: DatasetSpec, dataset: Any) -> str | None:
    if spec.upstream_license:
        return spec.upstream_license
    info = getattr(dataset, "info", None)
    if info is None:
        return None
    license_id = getattr(info, "license", None) or getattr(info, "license_name", None)
    return license_id if license_id else None


def fetch_huggingface_dataset(spec: DatasetSpec) -> FetchResult:
    dataset = _load_hf_split(spec)
    records = [_record_from_row(row) for row in dataset]
    return FetchResult(
        records=records,
        source_uri=huggingface_source_uri(spec),
        resolved_revision=requested_revision(spec),
        upstream_license=normalize_license(_hf_license(spec, dataset)),
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
