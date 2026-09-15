"""Matrix cell serialization and fingerprint compatibility warnings.

This is the environment-fingerprint slice of the cross-model matrix runner
(RM-105 / RM-1318). It does not orchestrate execution; it persists a versioned
fingerprint on every cell and refuses to silently pool incompatible runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.jsonio import write_json

from combine_for_ai.environment import (
    EnvironmentFingerprint,
    differing_material_fields,
)


MATRIX_CELL_SCHEMA = "combine_for_ai.matrix_cell.v1"
MATRIX_AGGREGATE_SCHEMA = "combine_for_ai.matrix_aggregate.v1"


@dataclass(frozen=True)
class MatrixCellResult:
    cell_id: str
    model: str
    quantization: str
    dataset: str
    fingerprint: EnvironmentFingerprint
    seed: int | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cell_id": self.cell_id,
            "dataset": self.dataset,
            "fingerprint": self.fingerprint.to_canonical_dict(),
            "fingerprint_digest": self.fingerprint.digest(),
            "metrics": dict(self.metrics),
            "model": self.model,
            "quantization": self.quantization,
            "schema": MATRIX_CELL_SCHEMA,
            "seed": self.seed,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def compatibility_warnings(cells: Sequence[MatrixCellResult]) -> list[str]:
    """Concise warnings when supposedly comparable cells differ materially."""
    if len(cells) < 2:
        return []
    groups: dict[str, list[MatrixCellResult]] = {}
    for cell in cells:
        groups.setdefault(cell.fingerprint.digest(), []).append(cell)
    if len(groups) == 1:
        return []

    ordered_digests = sorted(groups)
    reference = groups[ordered_digests[0]][0]
    warnings = [
        (
            "WARNING: environment fingerprints differ; "
            f"{len(groups)} distinct environments across {len(cells)} cells. "
            "Runs will not be pooled as comparable."
        )
    ]
    for digest in ordered_digests:
        group = groups[digest]
        ids = ", ".join(f"`{cell.cell_id}`" for cell in group)
        sample = group[0]
        if digest == reference.fingerprint.digest():
            warnings.append(f"reference {ids} digest `{_short_digest(digest)}`")
            continue
        fields = differing_material_fields(reference.fingerprint, sample.fingerprint)
        field_list = ", ".join(fields) if fields else "digest"
        warnings.append(
            f"{ids} digest `{_short_digest(digest)}` differs in {field_list}"
        )
    return warnings


def aggregate_matrix_report(
    cells: Sequence[MatrixCellResult],
    *,
    matrix_id: str | None = None,
) -> dict[str, Any]:
    """Build an aggregate payload that references each cell's fingerprint digest."""
    warnings = compatibility_warnings(cells)
    digests = sorted({cell.fingerprint.digest() for cell in cells})
    return {
        "cell_count": len(cells),
        "cells": [cell.to_payload() for cell in cells],
        "compatibility_warnings": warnings,
        "compatible": len(digests) <= 1,
        "fingerprint_digests": digests,
        "matrix_id": matrix_id,
        "schema": MATRIX_AGGREGATE_SCHEMA,
    }


def render_matrix_markdown(payload: Mapping[str, Any]) -> str:
    matrix_id = payload.get("matrix_id") or "unnamed"
    warnings = list(payload.get("compatibility_warnings") or [])
    digests = list(payload.get("fingerprint_digests") or [])
    lines = [
        f"# Matrix aggregate `{matrix_id}`",
        "",
        f"Cells: {payload.get('cell_count', 0)}",
        f"Fingerprint digests: {len(digests)}",
        "",
        "## Compatibility",
        "",
    ]
    if not warnings:
        if digests:
            lines.append(
                f"All cells share fingerprint digest `{digests[0]}`."
            )
        else:
            lines.append("No cells to compare.")
    else:
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    for cell in payload.get("cells") or []:
        cell_id = cell.get("cell_id", "")
        digest = cell.get("fingerprint_digest", "")
        lines.append(f"- `{cell_id}` fingerprint `{digest}`")
    lines.append("")
    return "\n".join(lines)


def write_matrix_reports(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    run_id: str,
    formats: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown aggregate reports. JSON is the machine record."""
    selected = list(formats) if formats is not None else ["json", "markdown"]
    output_dir = Path(output_dir)
    written: dict[str, Path] = {}
    if "json" in selected:
        path = output_dir / "json" / f"{run_id}.matrix.json"
        write_json(path, dict(payload))
        written["json"] = path
    if "markdown" in selected:
        path = output_dir / "markdown" / f"{run_id}.matrix.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_matrix_markdown(payload), encoding="utf-8")
        written["markdown"] = path
    return written


def _short_digest(digest: str) -> str:
    return digest[:12] if len(digest) > 12 else digest


__all__ = [
    "MATRIX_AGGREGATE_SCHEMA",
    "MATRIX_CELL_SCHEMA",
    "MatrixCellResult",
    "aggregate_matrix_report",
    "compatibility_warnings",
    "render_matrix_markdown",
    "write_matrix_reports",
]
