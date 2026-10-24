"""CSV report writing for packaged combine_for_ai report writers.

``benchmarks.reporting`` re-exports :func:`write_csv`, so dev checkouts keep a
single implementation (same pattern as ``jsonio``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from combine_for_ai.jsonio import ensure_dir

#: Cells starting with these are spreadsheet formulas; the writer prefixes ``'``.
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError("no rows to write to csv report")

    fieldnames = list(rows[0].keys())

    def csv_safe(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
            return f"'{value}"
        return value

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: csv_safe(value) for key, value in row.items()} for row in rows
        )


__all__ = ["CSV_FORMULA_PREFIXES", "write_csv"]
