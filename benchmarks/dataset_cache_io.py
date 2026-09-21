"""No-follow cache filesystem and on-disk entry validation."""

from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path

from benchmarks.dataset_cache_models import (
    LICENSE_SCOPE_DATASET_SOURCE,
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    CacheKey,
    CacheManifest,
    CacheValidationError,
    DatasetRecord,
    checksum_bytes,
    deserialize_record,
)


def lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def is_symlink(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode)


def is_directory(st: os.stat_result) -> bool:
    return stat.S_ISDIR(st.st_mode)


def is_regular_file(st: os.stat_result) -> bool:
    return stat.S_ISREG(st.st_mode)


def destination_occupied_by_live_entry(exc: OSError, live: Path) -> bool:
    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR}:
        return False
    st = lstat_or_none(live)
    return st is not None and is_directory(st) and not is_symlink(st)


def open_nofollow(path: Path, flags: int, mode: int = 0o644) -> int:
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, mode)


def write_bytes_nofollow(path: Path, payload: bytes) -> None:
    try:
        fd = open_nofollow(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
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


def write_text_nofollow(path: Path, text: str) -> None:
    write_bytes_nofollow(path, text.encode("utf-8"))


def dir_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def open_dir_nofollow(path: Path) -> int:
    try:
        if is_symlink(path.lstat()):
            raise CacheValidationError(
                f"cache entry is an unsafe symlink: {path}",
                path=path,
                reason="unsafe-symlink",
            )
    except CacheValidationError:
        raise
    except OSError as exc:
        raise CacheValidationError(
            f"cannot inspect cache entry directory {path}: {exc}",
            path=path,
            reason=_dir_open_error_reason(exc),
        ) from exc
    try:
        return os.open(path, dir_open_flags())
    except OSError as exc:
        raise CacheValidationError(
            f"cannot open cache entry directory {path}: {exc}",
            path=path,
            reason=_dir_open_error_reason(exc),
        ) from exc


def _dir_open_error_reason(exc: OSError) -> str:
    return {
        errno.ELOOP: "unsafe-symlink",
        errno.ENOTDIR: "not-a-directory",
    }.get(exc.errno, "unreadable-entry")


def read_in_dir(dir_fd: int, name: str, entry: Path) -> bytes:
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
        if not is_regular_file(st):
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


def write_in_dir(dir_fd: int, name: str, payload: bytes, entry: Path) -> None:
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


def write_cache_files(staging: Path, payload: bytes, manifest: CacheManifest) -> None:
    dir_fd = open_dir_nofollow(staging)
    try:
        write_in_dir(dir_fd, RECORDS_FILENAME, payload, staging)
        encoded = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        write_in_dir(dir_fd, MANIFEST_FILENAME, encoded.encode("utf-8"), staging)
    finally:
        os.close(dir_fd)


def parse_manifest(manifest_bytes: bytes, entry: Path) -> CacheManifest:
    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
        return CacheManifest.from_dict(raw_manifest)
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise CacheValidationError(
            f"unreadable manifest at {entry / MANIFEST_FILENAME}: {exc}",
            path=entry,
            reason="unreadable-manifest",
        ) from exc


def records_if_valid(
    entry: Path,
    key: CacheKey,
    manifest: CacheManifest,
    payload: bytes,
) -> list[DatasetRecord]:
    _require_checksum(entry, manifest, payload)
    _require_schema(entry, key, manifest)
    _require_cache_key(entry, key, manifest)
    _require_license_scope(entry, manifest)
    records = decode_records(payload, entry)
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


def _require_license_scope(entry: Path, manifest: CacheManifest) -> None:
    if manifest.license_scope != LICENSE_SCOPE_DATASET_SOURCE:
        raise CacheValidationError(
            f"invalid license scope at {entry / MANIFEST_FILENAME}",
            path=entry,
            reason="invalid-license-scope",
        )


def decode_records(payload: bytes, entry: Path) -> list[DatasetRecord]:
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
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CacheValidationError(
                f"invalid record on line {line_number} in {entry / RECORDS_FILENAME}",
                path=entry,
                reason=f"invalid-record line={line_number}",
            ) from exc
    return records
