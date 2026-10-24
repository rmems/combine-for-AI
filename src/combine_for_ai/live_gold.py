"""Gold-patch hygiene flags: git blobs, unified diffs, test-path heuristics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

from combine_for_ai.git_blob import git_blob_sha1
from combine_for_ai.live_errors import LiveBaselineError

_DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<path>.+?) b/(?P<path2>.+?)$")
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
    ".spec.js",
    ".test.ts",
    ".test.js",
)


def _split_patch_files(patch: str) -> dict[str, str]:
    if not patch.strip():
        return {}
    files: dict[str, list[str]] = {}
    current: str | None = None
    for line in patch.splitlines():
        match = _DIFF_FILE_RE.match(line)
        if match:
            current = match.group("path")
            files[current] = [line]
            continue
        if current is not None:
            files[current].append(line)
    return {path: "\n".join(chunk) + "\n" for path, chunk in files.items()}


def _hunk_target_index(old_start: int, old_count: int | None) -> int:
    if old_start == 0:
        return 0
    count = 1 if old_count is None else old_count
    if count == 0:
        return old_start
    return old_start - 1


def _seek_hunk(
    out: list[str],
    base_lines: list[str],
    idx: int,
    old_start: int,
    old_count: int | None,
) -> int:
    target_idx = _hunk_target_index(old_start, old_count)
    if target_idx < idx:
        raise LiveBaselineError("patch hunks overlap or are out of order")
    out.extend(base_lines[idx:target_idx])
    return target_idx


def _require_base_line(base_lines: list[str], idx: int, expected: str, kind: str) -> None:
    if idx >= len(base_lines):
        raise LiveBaselineError(f"patch {kind} exceeds base file length")
    if base_lines[idx].rstrip("\r\n") != expected:
        raise LiveBaselineError(f"patch {kind} does not match base file")


def _apply_hunk_body_line(
    line: str, *, base_lines: list[str], out: list[str], idx: int
) -> int:
    if line.startswith(" "):
        _require_base_line(base_lines, idx, line[1:], "context")
        out.append(base_lines[idx])
        return idx + 1
    if line.startswith("-"):
        _require_base_line(base_lines, idx, line[1:], "deletion")
        return idx + 1
    if line.startswith("+"):
        addition = line[1:]
        if not addition.endswith("\n"):
            addition += "\n"
        out.append(addition)
        return idx
    if line == "\\ No newline at end of file":
        return idx
    raise LiveBaselineError(f"unsupported hunk line: {line!r}")


def _start_hunk(
    line: str, out: list[str], base_lines: list[str], idx: int
) -> tuple[bool, int]:
    hunk_match = _HUNK_HEADER_RE.match(line)
    if not hunk_match:
        return False, idx
    old_count = int(hunk_match.group(2)) if hunk_match.group(2) is not None else None
    return True, _seek_hunk(out, base_lines, idx, int(hunk_match.group(1)), old_count)


def _is_pre_hunk_file_header(line: str, in_hunk: bool) -> bool:
    return not in_hunk and line.startswith(("index ", "---", "+++"))


def _apply_unified_hunks(base: str, file_patch: str) -> str:
    if not file_patch.strip():
        return base
    base_lines = [] if base == "" else base.splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    in_hunk = False
    for line in file_patch.splitlines():
        if line.startswith("diff --git"):
            in_hunk = False
            continue
        if _is_pre_hunk_file_header(line, in_hunk):
            continue
        started, idx = _start_hunk(line, out, base_lines, idx)
        if started:
            in_hunk = True
            continue
        if in_hunk:
            idx = _apply_hunk_body_line(line, base_lines=base_lines, out=out, idx=idx)
    out.extend(base_lines[idx:])
    return "".join(out)


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
    if test_patch and path in _split_patch_files(test_patch):
        return True
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


def _is_new_file_patch(file_patch: str) -> bool:
    if "--- /dev/null" in file_patch:
        return True
    match = _HUNK_HEADER_RE.search(file_patch)
    return bool(match and int(match.group(1)) == 0)


def _base_text_for_gold_path(path: str, file_patch: str, base_files: Mapping[str, str]) -> str | None:
    if path in base_files:
        return base_files[path]
    if _is_new_file_patch(file_patch):
        return ""
    return None


def byte_identical_gold_src(
    *,
    base_files: Mapping[str, str],
    gold_patch: str,
    model_patch: str,
    test_patch: str | None = None,
) -> bool:
    gold_files = _split_patch_files(gold_patch)
    model_files = _split_patch_files(model_patch)
    compared = 0
    for path in gold_files:
        if is_test_path(path, test_patch=test_patch):
            continue
        compared += 1
        base = _base_text_for_gold_path(path, gold_files[path], base_files)
        if base is None:
            return False
        try:
            gold_bytes = _apply_unified_hunks(base, gold_files[path]).encode("utf-8")
            if path not in model_files:
                return False
            model_bytes = _apply_unified_hunks(base, model_files[path]).encode("utf-8")
        except LiveBaselineError:
            return False
        if git_blob_sha1(gold_bytes) != git_blob_sha1(model_bytes):
            return False
    return compared > 0


def _normalize_patch(patch: str) -> str:
    lines = [line.rstrip() for line in patch.splitlines()]
    return "\n".join(line for line in lines if line)


def contains_gold_test_patch(*, model_patch: str, test_patch: str) -> bool:
    if not test_patch.strip():
        return False
    normalized_model = _normalize_patch(model_patch)
    normalized_test = _normalize_patch(test_patch)
    if normalized_test in normalized_model:
        return True
    model_files = _split_patch_files(model_patch)
    for path, chunk in _split_patch_files(test_patch).items():
        if _normalize_patch(chunk) in normalized_model:
            return True
        if path in model_files and _normalize_patch(model_files[path]) == _normalize_patch(chunk):
            return True
    return False


__all__ = [
    "byte_identical_gold_src",
    "contains_gold_test_patch",
    "git_blob_sha1",
    "is_test_path",
    "load_base_files",
]
