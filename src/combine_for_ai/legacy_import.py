"""Import raw GOZ1 archives without treating malformed numbers as missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from combine_for_ai.goz_import import (
    GozExperimentKind,
    detect_experiment_kind,
    import_multiblock_metrics,
    import_route_preservation,
)
from combine_for_ai.legacy_schema import ARMS, RAW_METRIC_KEYS
from combine_for_ai.legacy_values import is_nonnegative_int, require_finite_metric


def imported_rows(raw: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if detect_experiment_kind(raw) is GozExperimentKind.MULTIBLOCK:
        return _multiblock_rows(raw, path)
    return _route_rows(raw, path)


def _route_rows(raw: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    _validate_route_observations(raw.get("summary"))
    # `path` is the archive the caller already opened, stored as provenance.
    imported = import_route_preservation(raw, source_path=str(path))
    return [row.to_report_row() for row in imported.rows]


def _validate_route_observations(summary: Any) -> None:
    if not isinstance(summary, list):
        return
    for item in summary:
        _validate_observed(item)


def _validate_observed(item: Any) -> None:
    if not isinstance(item, dict) or "observed" not in item or item["observed"] is None:
        return
    require_finite_metric(item["observed"], str(item.get("name") or "summary"))


def _multiblock_rows(raw: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    chain = raw["chain"]
    entries = _block_entries(chain)
    by_block = _entries_by_block(entries)
    padded = _padded_blocks(_declared_blocks(chain, entries), by_block)
    if not padded:
        return []
    imported = import_multiblock_metrics(
        {**raw, "chain": {**chain, "per_block": padded}},
        source_path=str(path),
    )
    rows = [row.to_report_row() for row in imported.rows]
    _apply_declared_statuses(rows, by_block)
    return rows


def _block_entries(chain: dict[str, Any]) -> list[dict[str, Any]]:
    entries = chain["per_block"]
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("chain.per_block must contain only objects")
    return entries


def _declared_blocks(chain: dict[str, Any], entries: list[dict[str, Any]]) -> list[Any]:
    blocks = chain.get("blocks", [entry.get("block") for entry in entries])
    if not isinstance(blocks, list) or any(not is_nonnegative_int(b) for b in blocks):
        raise ValueError("chain.blocks must be nonnegative integers")
    if len(blocks) != len(set(blocks)):
        raise ValueError("duplicate chain.blocks")
    return blocks


def _entries_by_block(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_block: dict[int, dict[str, Any]] = {}
    for entry in entries:
        block = entry.get("block")
        if not is_nonnegative_int(block) or block in by_block:
            raise ValueError("chain.per_block requires unique nonnegative block indices")
        by_block[block] = entry
    return by_block


def _padded_blocks(
    blocks: list[Any], by_block: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    order = dict.fromkeys([*blocks, *by_block])
    return [_padded_entry(block, by_block.get(block)) for block in order]


def _padded_entry(block: int, entry: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(entry or {"block": block})
    for arm in ARMS:
        _ensure_arm_object(merged, arm)
    return merged


def _ensure_arm_object(entry: dict[str, Any], arm: str) -> None:
    metrics = entry.get(arm)
    if metrics is None:
        entry[arm] = {}
        return
    if not isinstance(metrics, dict):
        raise ValueError(f"chain.per_block {arm} must be an object or null")
    _reject_bad_arm_metrics(metrics, arm)


def _reject_bad_arm_metrics(metrics: dict[str, Any], arm: str) -> None:
    for key in RAW_METRIC_KEYS:
        if key in metrics and metrics[key] is not None:
            require_finite_metric(metrics[key], f"{arm}.{key}")


def _apply_declared_statuses(
    rows: list[dict[str, Any]], by_block: dict[int, dict[str, Any]]
) -> None:
    for row in rows:
        declared = by_block.get(row["block_index"], {}).get(row["arm"], {})
        if isinstance(declared, dict) and "status" in declared:
            row["status"] = declared["status"]
