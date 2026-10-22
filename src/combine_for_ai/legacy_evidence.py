"""Turn legacy GOZ1 archives into unqualified rendering records.

The existing importer still owns row normalization. Present non-numeric
measurements are rejected here so they cannot be reported as missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from combine_for_ai.legacy_import import imported_rows
from combine_for_ai.legacy_metrics import legacy_record
from combine_for_ai.legacy_rows import normalized_rows, require_legacy_metadata
from combine_for_ai.report_models import EvidenceRecord


def legacy_records(raw: dict[str, Any], path: Path) -> list[EvidenceRecord]:
    rows = _legacy_rows(raw, path)
    # An empty raw multiblock chain returns before metadata checks. Normalized
    # result arrays still validate certification/summary, including when empty.
    if rows or "results" in raw:
        require_legacy_metadata(raw)
    return [legacy_record(row, raw, path) for row in rows]


def _legacy_rows(raw: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    run = raw.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("legacy run must be an object")
    if "results" in raw:
        return normalized_rows(raw, path)
    return imported_rows(raw, path)
