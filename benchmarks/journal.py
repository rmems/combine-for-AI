"""Append-safe JSONL journal for matrix cell state transitions.

Each record is one JSON object with a CRC32 over a canonical encoding of its
body. Records are written with ``flush`` + ``fsync`` so a crash can at worst
leave a truncated or corrupt *tail*. Replay keeps every valid earlier record
and truncates that tail instead of silently dropping it.
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from benchmarks.jsonio import ensure_dir
from benchmarks.matrix import MatrixError, canonical_dumps

JOURNAL_SCHEMA = "combine.matrix.journal.v1"
JOURNAL_VERSION = 1


class CellState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class CommitPoint(StrEnum):
    """Documented durability points a crash-injection test can interrupt."""

    JOURNAL_WRITE = "journal_write"
    RESULT_COMMIT = "result_commit"
    AGGREGATE_UPDATE = "aggregate_update"


class JournalError(MatrixError):
    """Unrecoverable journal damage (not a recoverable truncated tail)."""


def _crc32_hex(payload: bytes) -> str:
    return f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"


def encode_record(body: Mapping[str, Any]) -> bytes:
    canonical = canonical_dumps(body).encode("utf-8")
    record = {
        "body": body,
        "crc32": _crc32_hex(canonical),
        "v": JOURNAL_VERSION,
    }
    return (canonical_dumps(record) + "\n").encode("utf-8")


def decode_record_line(line: bytes) -> dict[str, Any]:
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JournalError("journal line is not valid JSON") from exc
    if not isinstance(record, dict):
        raise JournalError("journal record is not an object")
    body = record.get("body")
    crc = record.get("crc32")
    version = record.get("v")
    if version != JOURNAL_VERSION or not isinstance(body, dict) or not isinstance(crc, str):
        raise JournalError("journal record is missing v/crc32/body")
    expected = _crc32_hex(canonical_dumps(body).encode("utf-8"))
    if crc.lower() != expected:
        raise JournalError("journal record CRC32 mismatch")
    return body


@dataclass(frozen=True)
class JournalReplay:
    """Prefix of valid records plus whether a corrupt tail was dropped."""

    records: tuple[dict[str, Any], ...]
    valid_bytes: int
    truncated_tail: bool
    discarded_bytes: int


def replay_journal(path: Path) -> JournalReplay:
    """Read a journal, keeping only the valid prefix.

    A truncated last line, a CRC mismatch, or a malformed JSON object at the
    end is reported as ``truncated_tail`` and is *not* applied. Earlier valid
    records are preserved. Corruption after a valid prefix is treated as the
    start of the tail (append-only files should not have holes in the middle).
    """

    if not path.exists():
        return JournalReplay(records=(), valid_bytes=0, truncated_tail=False, discarded_bytes=0)

    data = path.read_bytes()
    if not data:
        return JournalReplay(records=(), valid_bytes=0, truncated_tail=False, discarded_bytes=0)

    records: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline < 0:
            return JournalReplay(
                records=tuple(records),
                valid_bytes=offset,
                truncated_tail=True,
                discarded_bytes=len(data) - offset,
            )
        line = data[offset:newline]
        # A trailing newline is required for a complete record. Empty lines
        # between records are treated as corruption from that point.
        try:
            if not line:
                raise JournalError("empty journal line")
            records.append(decode_record_line(line))
        except JournalError:
            return JournalReplay(
                records=tuple(records),
                valid_bytes=offset,
                truncated_tail=True,
                discarded_bytes=len(data) - offset,
            )
        offset = newline + 1

    return JournalReplay(
        records=tuple(records),
        valid_bytes=len(data),
        truncated_tail=False,
        discarded_bytes=0,
    )


def recover_journal_file(path: Path) -> JournalReplay:
    """Replay and truncate a corrupt tail in place so the next append is clean."""

    replay = replay_journal(path)
    if replay.truncated_tail:
        ensure_dir(path.parent)
        with path.open("r+b") as handle:
            handle.truncate(replay.valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return replay


class ResumeJournal:
    """Opened journal used for atomic state transitions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def open(self) -> JournalReplay:
        ensure_dir(self.path.parent)
        replay = recover_journal_file(self.path)
        self._handle = self.path.open("ab")
        return replay

    def append(self, body: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise JournalError("journal is not open")
        self._handle.write(encode_record(body))
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "ResumeJournal":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
