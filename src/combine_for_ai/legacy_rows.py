"""Validate normalized GOZ1 result arrays before they become records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from combine_for_ai.legacy_schema import GOZ_METRICS, GOZ_SCHEMAS, SUPPORTED_VERSIONS
from combine_for_ai.legacy_values import is_nonnegative_int, require_finite_metric


def normalized_rows(raw: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    _require_normalized_schema(raw, path)
    rows = raw["results"]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: results must be an array of objects")
    for index, row in enumerate(rows):
        _require_result_row(row, path, index)
    return rows


def require_legacy_metadata(raw: dict[str, Any]) -> None:
    certification = raw.get("certification") or {}
    summary = raw.get("summary") or []
    if not isinstance(certification, dict) or not isinstance(summary, list):
        raise ValueError("legacy certification/summary must be an object/array")


def _require_normalized_schema(raw: dict[str, Any], path: Path) -> None:
    version = raw.get("combine_import_schema", raw.get("schema_version", 1))
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("unsupported legacy schema_version")
    if raw["run"].get("schema") not in GOZ_SCHEMAS:
        raise ValueError(
            f"{path}: legacy report lacks a supported GOZ1 schema; "
            "prepare combine.plot-evidence.v1 summaries (do not promote mock presets)"
        )


def _require_result_row(row: Any, path: Path, index: int) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{path}: results must be an array of objects")
    _require_arm(row, path, index)
    _require_block_index(row, path, index)
    _require_present_metrics(row, f"results[{index}]")


def _require_arm(row: dict[str, Any], path: Path, index: int) -> None:
    arm = row.get("arm")
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError(f"{path}: results[{index}] requires a non-empty arm")


def _require_block_index(row: dict[str, Any], path: Path, index: int) -> None:
    if "block_index" not in row or row["block_index"] is None:
        return
    if not is_nonnegative_int(row["block_index"]):
        raise ValueError(
            f"{path}: results[{index}] block_index must be a nonnegative integer"
        )


def _require_present_metrics(row: dict[str, Any], label: str) -> None:
    for name in GOZ_METRICS:
        if name in row and row[name] is not None:
            require_finite_metric(row[name], f"{label}.{name}")
