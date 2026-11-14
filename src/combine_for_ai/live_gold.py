"""Gold-patch hygiene flags: git blobs, unified diffs, test-path heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from combine_for_ai.git_blob import git_blob_sha1
from combine_for_ai.live_errors import LiveBaselineError

_DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<old>.+?) b/(?P<new>.+?)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__"})
_TEST_FILE_PREFIXES = ("test_",)
_TEST_FILE_SUFFIXES = (
    "_test.py",
    "_test.rs",
    "_test.go",
    "_test.java",
    "_spec.rb",
    ".spec.ts",
    ".spec.tsx",
    ".spec.js",
    ".spec.jsx",
    ".test.ts",
    ".test.tsx",
    ".test.js",
    ".test.jsx",
)


@dataclass(frozen=True, slots=True)
class _HunkLine:
    kind: str  # "context" | "delete" | "add" | "no_newline"
    text: str


@dataclass(frozen=True, slots=True)
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[_HunkLine, ...]


@dataclass(frozen=True, slots=True)
class _FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: tuple[_Hunk, ...]
    deleted: bool = False
    new_file: bool = False
    renamed: bool = False


def _split_patch_lines(patch: str) -> list[str]:
    """Split a patch on ``\\n`` only, preserving ``\\r`` inside content lines."""
    if not patch:
        return []
    parts = patch.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def _split_lines_keepends(text: str) -> list[str]:
    """Split text into lines, keeping each line's terminator (``\\n`` or ``\\r\\n``)."""
    if not text:
        return []
    lines: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            lines.append(text[start : i + 1])
            start = i + 1
    if start < len(text):
        lines.append(text[start:])
    return lines


def _line_content(line: str) -> str:
    """Return the line content without its ``\\n`` terminator, preserving ``\\r``."""
    if line.endswith("\n"):
        return line[:-1]
    return line


def _parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    match = _HUNK_HEADER_RE.match(line)
    if not match:
        return None
    old_start = int(match.group(1))
    old_count = int(match.group(2)) if match.group(2) is not None else 1
    new_start = int(match.group(3))
    new_count = int(match.group(4)) if match.group(4) is not None else 1
    return old_start, old_count, new_start, new_count


def _parse_file_patch(lines: list[str], start: int) -> tuple[_FilePatch, int]:
    """Parse one ``diff --git`` block starting at ``lines[start]``."""
    diff_match = _DIFF_FILE_RE.match(lines[start])
    if not diff_match:
        raise LiveBaselineError(f"malformed diff header: {lines[start]!r}")

    old_path = diff_match.group("old")
    new_path = diff_match.group("new")
    deleted = False
    new_file = False
    renamed = False
    hunks: list[_Hunk] = []
    i = start + 1

    # Pre-hunk metadata.
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@ "):
            break
        if line.startswith("deleted file mode"):
            deleted = True
        elif line.startswith("new file mode"):
            new_file = True
        elif line.startswith(("rename from ", "rename to ")):
            renamed = True
        elif line.startswith("--- "):
            target = line[4:].strip()
            if target == "/dev/null":
                old_path = None
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                new_path = None
                deleted = True
        i += 1

    # Hunks.
    while i < len(lines):
        header = _parse_hunk_header(lines[i])
        if header is None:
            break
        old_start, old_count, new_start, new_count = header
        i += 1
        hunk_lines: list[_HunkLine] = []
        while i < len(lines) and not lines[i].startswith("@@ ") and not lines[i].startswith("diff --git"):
            raw = lines[i]
            if raw.startswith("\\"):
                hunk_lines.append(_HunkLine(kind="no_newline", text=""))
            elif raw.startswith("+"):
                hunk_lines.append(_HunkLine(kind="add", text=raw[1:]))
            elif raw.startswith("-"):
                hunk_lines.append(_HunkLine(kind="delete", text=raw[1:]))
            elif raw.startswith(" "):
                hunk_lines.append(_HunkLine(kind="context", text=raw[1:]))
            elif raw == "":
                # Empty line inside a hunk is treated as a context line with empty content.
                hunk_lines.append(_HunkLine(kind="context", text=""))
            else:
                raise LiveBaselineError(f"unsupported hunk line: {raw!r}")
            i += 1
        hunks.append(
            _Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=tuple(hunk_lines),
            )
        )

    if new_path is None and old_path is None:
        raise LiveBaselineError("file patch has no old or new path")

    return (
        _FilePatch(
            old_path=old_path,
            new_path=new_path,
            hunks=tuple(hunks),
            deleted=deleted,
            new_file=new_file,
            renamed=renamed,
        ),
        i,
    )


def _parse_patch_files(patch: str) -> dict[str, _FilePatch]:
    """Parse a unified diff into file patches keyed by destination path."""
    if not patch.strip():
        return {}
    lines = _split_patch_lines(patch)
    files: dict[str, _FilePatch] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("diff --git"):
            i += 1
            continue
        file_patch, i = _parse_file_patch(lines, i)
        key = file_patch.new_path if file_patch.new_path is not None else file_patch.old_path
        if key is None:
            raise LiveBaselineError("file patch has no usable path")
        if key in files:
            raise LiveBaselineError(f"duplicate file path in patch: {key!r}")
        files[key] = file_patch
    return files


def _format_file_patch(fp: _FilePatch) -> str:
    """Render a parsed file patch back to unified-diff text (for tests/helpers)."""
    lines: list[str] = []
    old = f"a/{fp.old_path}" if fp.old_path else "/dev/null"
    new = f"b/{fp.new_path}" if fp.new_path else "/dev/null"
    lines.append(f"diff --git {old} {new}")
    if fp.new_file:
        lines.append("new file mode 100644")
    if fp.deleted:
        lines.append("deleted file mode 100644")
    if fp.renamed:
        lines.append(f"rename from {fp.old_path}")
        lines.append(f"rename to {fp.new_path}")
    lines.append(f"--- {old}")
    lines.append(f"+++ {new}")
    for hunk in fp.hunks:
        lines.append(
            f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
        )
        for hl in hunk.lines:
            if hl.kind == "context":
                lines.append(f" {hl.text}")
            elif hl.kind == "delete":
                lines.append(f"-{hl.text}")
            elif hl.kind == "add":
                lines.append(f"+{hl.text}")
            elif hl.kind == "no_newline":
                lines.append("\\ No newline at end of file")
    return "\n".join(lines) + "\n"


def _hunk_target_index(old_start: int, old_count: int) -> int:
    if old_start == 0 and old_count == 0:
        return 0
    if old_count == 0:
        return old_start
    return old_start - 1


def _seek_hunk(
    out: list[str],
    base_lines: list[str],
    idx: int,
    hunk: _Hunk,
) -> int:
    target_idx = _hunk_target_index(hunk.old_start, hunk.old_count)
    if target_idx < idx:
        raise LiveBaselineError("patch hunks overlap or are out of order")
    if target_idx > len(base_lines):
        raise LiveBaselineError("patch hunk target exceeds base file length")
    out.extend(base_lines[idx:target_idx])
    return target_idx


def _require_base_line(base_lines: list[str], idx: int, expected: str, kind: str) -> None:
    if idx >= len(base_lines):
        raise LiveBaselineError(f"patch {kind} exceeds base file length")
    if _line_content(base_lines[idx]) != expected:
        raise LiveBaselineError(f"patch {kind} does not match base file")


def _apply_hunk(hunk: _Hunk, base_lines: list[str], out: list[str], idx: int) -> int:
    idx = _seek_hunk(out, base_lines, idx, hunk)
    consumed_old = 0
    emitted_new = 0
    last_add_index: int | None = None

    for hl in hunk.lines:
        if hl.kind == "no_newline":
            if last_add_index is not None and last_add_index < len(out):
                out[last_add_index] = out[last_add_index].removesuffix("\n")
            last_add_index = None
            continue
        last_add_index = None
        if hl.kind == "context":
            _require_base_line(base_lines, idx, hl.text, "context")
            out.append(base_lines[idx])
            idx += 1
            consumed_old += 1
            emitted_new += 1
        elif hl.kind == "delete":
            _require_base_line(base_lines, idx, hl.text, "deletion")
            idx += 1
            consumed_old += 1
        elif hl.kind == "add":
            addition = hl.text
            if not addition.endswith("\n"):
                addition += "\n"
            out.append(addition)
            emitted_new += 1
            last_add_index = len(out) - 1

    if consumed_old != hunk.old_count:
        raise LiveBaselineError(
            f"hunk old-count mismatch: declared {hunk.old_count}, consumed {consumed_old}"
        )
    if emitted_new != hunk.new_count:
        raise LiveBaselineError(
            f"hunk new-count mismatch: declared {hunk.new_count}, emitted {emitted_new}"
        )
    return idx


def _apply_file_patch(base: str, file_patch: _FilePatch) -> str | None:
    """Apply a parsed file patch to ``base``. Returns ``None`` for deletions."""
    if file_patch.deleted and not file_patch.hunks:
        return None
    base_lines = _split_lines_keepends(base)
    out: list[str] = []
    idx = 0
    for hunk in file_patch.hunks:
        idx = _apply_hunk(hunk, base_lines, out, idx)
    out.extend(base_lines[idx:])
    result = "".join(out)
    if file_patch.deleted:
        # A deletion patch with hunks removes all content.
        return None
    return result


def _apply_unified_hunks(base: str, file_patch: str) -> str:
    """Backward-compatible helper: apply raw hunk text to ``base``."""
    if not file_patch.strip():
        return base
    # Wrap the raw hunk text in a minimal diff so the parser accepts it.
    wrapped = f"diff --git a/x b/x\n--- a/x\n+++ b/x\n{file_patch}"
    files = _parse_patch_files(wrapped)
    fp = next(iter(files.values()))
    result = _apply_file_patch(base, fp)
    if result is None:
        return ""
    return result


def _is_rust_test_path(path_lower: str, filename: str) -> bool:
    segments = path_lower.split("/")
    return "tests" in segments[:-1] or filename.endswith("_test.rs")


def _is_test_directory(path_lower: str) -> bool:
    segments = path_lower.split("/")
    return any(segment in _TEST_DIR_NAMES for segment in segments[:-1]) or path_lower.startswith(
        "test/"
    )


def _is_test_filename(filename: str) -> bool:
    return (
        filename.startswith(_TEST_FILE_PREFIXES)
        or filename.endswith(_TEST_FILE_SUFFIXES)
        or ".test." in filename
    )


def is_test_path(path: str, *, test_patch: str | None = None) -> bool:
    if test_patch:
        try:
            if path in _parse_patch_files(test_patch):
                return True
        except LiveBaselineError:
            pass
    lower = path.replace("\\", "/").lower()
    filename = lower.rsplit("/", 1)[-1]
    if filename.endswith(".rs"):
        return _is_rust_test_path(lower, filename)
    return _is_test_directory(lower) or _is_test_filename(filename)


def load_base_files(root: Path, paths: Iterable[str]) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for path in paths:
        candidate = root / path
        if candidate.is_file():
            loaded[path] = candidate.read_text(encoding="utf-8")
    return loaded


def _base_text_for_file(path: str, file_patch: _FilePatch, base_files: Mapping[str, str]) -> str | None:
    if file_patch.new_file:
        return ""
    # For renames, the base content lives at the old path.
    lookup = file_patch.old_path if file_patch.renamed and file_patch.old_path else path
    if lookup in base_files:
        return base_files[lookup]
    if file_patch.deleted:
        # Deletions require the file to exist in base.
        return None
    return None


def _non_test_paths(files: Mapping[str, _FilePatch], test_patch: str | None) -> set[str]:
    return {path for path in files if not is_test_path(path, test_patch=test_patch)}


def _patch_result_for_path(
    path: str,
    file_patch: _FilePatch,
    base_files: Mapping[str, str],
) -> str | None:
    """Apply ``file_patch`` and return the resulting content, or ``None`` for deletions."""
    base = _base_text_for_file(path, file_patch, base_files)
    if base is None:
        raise LiveBaselineError(f"missing base file for {path!r}")
    return _apply_file_patch(base, file_patch)


def byte_identical_gold_src(
    *,
    base_files: Mapping[str, str],
    gold_patch: str,
    model_patch: str,
    test_patch: str | None = None,
) -> bool:
    gold_files = _parse_patch_files(gold_patch)
    model_files = _parse_patch_files(model_patch)
    gold_src = _non_test_paths(gold_files, test_patch)
    if not gold_src or gold_src != _non_test_paths(model_files, test_patch):
        return False
    for path in gold_src:
        gold_fp = gold_files[path]
        model_fp = model_files[path]
        # Renames must have the same destination and source.
        if gold_fp.renamed != model_fp.renamed:
            return False
        if gold_fp.renamed and (gold_fp.old_path != model_fp.old_path or gold_fp.new_path != model_fp.new_path):
            return False
        try:
            gold_result = _patch_result_for_path(path, gold_fp, base_files)
            model_result = _patch_result_for_path(path, model_fp, base_files)
        except LiveBaselineError:
            return False
        if gold_result is None and model_result is None:
            continue
        if gold_result is None or model_result is None:
            return False
        if gold_result != model_result:
            return False
    return True


def _normalize_patch(patch: str) -> str:
    lines = [line.rstrip() for line in patch.splitlines()]
    return "\n".join(line for line in lines if line)


def _patch_contains_lines(container: str, needle: str) -> bool:
    """True when ``needle`` appears as a sequence of complete lines in ``container``."""
    if not needle:
        return False
    container_lines = container.splitlines()
    needle_lines = needle.splitlines()
    if not needle_lines:
        return False
    n = len(needle_lines)
    for i in range(len(container_lines) - n + 1):
        if container_lines[i : i + n] == needle_lines:
            return True
    return False


def contains_gold_test_patch(*, model_patch: str, test_patch: str) -> bool:
    if not test_patch.strip():
        return False
    normalized_model = _normalize_patch(model_patch)
    normalized_test = _normalize_patch(test_patch)
    if _patch_contains_lines(normalized_model, normalized_test):
        return True
    model_files = _parse_patch_files(model_patch)
    for path, chunk in _parse_patch_files(test_patch).items():
        normalized_chunk = _normalize_patch(_format_file_patch(chunk))
        if _patch_contains_lines(normalized_model, normalized_chunk):
            return True
        if path in model_files:
            model_chunk = _normalize_patch(_format_file_patch(model_files[path]))
            if model_chunk == normalized_chunk:
                return True
    return False


__all__ = [
    "byte_identical_gold_src",
    "contains_gold_test_patch",
    "git_blob_sha1",
    "is_test_path",
    "load_base_files",
]
