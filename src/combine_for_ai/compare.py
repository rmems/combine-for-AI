"""Compare hybrid experiment rows (issue #15).

Builds comparison matrices from goz-import report JSON or raw grok-ozempic
experiment files (via import). Primary matrix: FP16 control vs expert-only
across blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from combine_for_ai.goz_import import GozImportError, import_goz_experiment


COMPARE_METRIC_KEYS = (
    "route_top1_agreement",
    "route_top2_agreement",
    "block_output_cosine",
    "resid_in_drift",
    "expert_load_js",
    "sparsity",
    "seconds",
)


class CompareError(ValueError):
    """Raised when comparison inputs are invalid."""


@dataclass(frozen=True)
class CompareResult:
    rows: list[dict[str, Any]]
    by_block: list[dict[str, Any]]
    sources: list[str]
    arms: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "comparison": {
                "sources": self.sources,
                "arms": self.arms,
                "metric_keys": list(COMPARE_METRIC_KEYS),
            },
            "rows": self.rows,
            "by_block": self.by_block,
        }


def _load_rows_from_goz_import_report(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"cannot read goz-import report {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CompareError(f"goz-import report root must be an object: {path}")
    results = raw.get("results")
    if not isinstance(results, list) or not results:
        raise CompareError(f"goz-import report missing results[]: {path}")
    rows: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict):
            row = dict(item)
            row.setdefault("source_path", str(path))
            rows.append(row)
    if not rows:
        raise CompareError(f"no dict rows in results[]: {path}")
    return rows


def load_experiment_rows(
    paths: list[Path],
    *,
    arms: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Load rows from goz-import JSON reports and/or raw experiment JSON."""
    if not paths:
        raise CompareError("at least one input path is required")
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompareError(f"cannot read {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CompareError(f"input root must be a JSON object: {path}")

        if isinstance(raw.get("results"), list):
            all_rows.extend(_load_rows_from_goz_import_report(path))
            continue

        # Treat as upstream grok-ozempic experiment JSON.
        try:
            imported = import_goz_experiment(
                path,
                arms=arms or ("expert_only", "fp16_control"),
            )
        except GozImportError as exc:
            raise CompareError(f"import failed for {path}: {exc}") from exc
        for row in imported.rows:
            all_rows.append(row.to_report_row())

    if arms:
        allowed = set(arms)
        all_rows = [r for r in all_rows if r.get("arm") in allowed]
    if not all_rows:
        raise CompareError("no rows available after filtering")
    return all_rows


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def build_comparison(
    rows: list[dict[str, Any]],
    *,
    baseline_arm: str = "fp16_control",
    treatment_arm: str = "expert_only",
) -> CompareResult:
    """Pair baseline vs treatment by block_index; emit flat rows + by_block deltas."""
    arms = sorted({str(r.get("arm")) for r in rows if r.get("arm") is not None})
    sources = sorted({str(r.get("source_path")) for r in rows if r.get("source_path")})

    # Index treatment/baseline by block.
    by_key: dict[tuple[Any, str], dict[str, Any]] = {}
    for row in rows:
        arm = row.get("arm")
        block = row.get("block_index")
        if arm is None:
            continue
        by_key[(block, str(arm))] = row

    blocks = sorted(
        {block for (block, _) in by_key.keys() if block is not None},
        key=lambda b: (isinstance(b, str), b),
    )
    # Also include None block as a single-row series when present.
    if any(block is None for (block, _) in by_key):
        blocks = [None, *blocks]

    by_block: list[dict[str, Any]] = []
    for block in blocks:
        base = by_key.get((block, baseline_arm))
        treat = by_key.get((block, treatment_arm))
        if base is None and treat is None:
            continue
        entry: dict[str, Any] = {
            "block_index": block,
            "baseline_arm": baseline_arm,
            "treatment_arm": treatment_arm,
        }
        for key in COMPARE_METRIC_KEYS:
            bval = _numeric(base.get(key)) if base else None
            tval = _numeric(treat.get(key)) if treat else None
            entry[f"baseline_{key}"] = bval
            entry[f"treatment_{key}"] = tval
            entry[f"delta_{key}"] = _delta(tval, bval)
        if base:
            entry["baseline_label"] = base.get("label")
            entry["baseline_scale_source"] = base.get("scale_source")
            entry["baseline_goz1_version"] = base.get("goz1_version")
        if treat:
            entry["treatment_label"] = treat.get("label")
            entry["treatment_scale_source"] = treat.get("scale_source")
            entry["treatment_goz1_version"] = treat.get("goz1_version")
            entry["treatment_pack_basename"] = treat.get("pack_basename")
        by_block.append(entry)

    # Flat rows sorted for CSV readability.
    flat = sorted(
        rows,
        key=lambda r: (
            r.get("block_index") is None,
            r.get("block_index") if r.get("block_index") is not None else -1,
            str(r.get("arm") or ""),
        ),
    )
    return CompareResult(rows=flat, by_block=by_block, sources=sources, arms=arms)


def write_comparison_reports(
    result: CompareResult,
    output_dir: Path,
    *,
    run_id: str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    from benchmarks.reporting import write_csv, write_json

    formats = formats or ["json", "csv", "markdown"]
    output_dir = Path(output_dir)
    written: dict[str, Path] = {}
    payload = result.to_payload()
    payload["run_id"] = run_id

    if "json" in formats:
        jpath = output_dir / "json" / f"{run_id}.compare.json"
        write_json(jpath, payload)
        written["json"] = jpath

    if "csv" in formats:
        if result.by_block:
            cpath = output_dir / "csv" / f"{run_id}.compare-by-block.csv"
            write_csv(cpath, result.by_block)
            written["csv"] = cpath
        elif result.rows:
            cpath = output_dir / "csv" / f"{run_id}.compare-rows.csv"
            write_csv(cpath, result.rows)
            written["csv"] = cpath

    if "markdown" in formats:
        mpath = output_dir / "markdown" / f"{run_id}.compare.md"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(_markdown_table(result), encoding="utf-8")
        written["markdown"] = mpath

    if not written:
        raise CompareError("no report formats selected")
    return written


def _markdown_table(result: CompareResult) -> str:
    lines = [
        "# Hybrid quant comparison",
        "",
        f"Arms: {', '.join(result.arms) or '(none)'}",
        f"Sources: {len(result.sources)}",
        "",
    ]
    if not result.by_block:
        lines.append("_No baseline/treatment block pairs found._")
        lines.append("")
        return "\n".join(lines)

    headers = [
        "block",
        "top1_base",
        "top1_treat",
        "Δtop1",
        "cos_base",
        "cos_treat",
        "Δcos",
        "resid_base",
        "resid_treat",
        "Δresid",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for row in result.by_block:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("block_index")),
                    fmt(row.get("baseline_route_top1_agreement")),
                    fmt(row.get("treatment_route_top1_agreement")),
                    fmt(row.get("delta_route_top1_agreement")),
                    fmt(row.get("baseline_block_output_cosine")),
                    fmt(row.get("treatment_block_output_cosine")),
                    fmt(row.get("delta_block_output_cosine")),
                    fmt(row.get("baseline_resid_in_drift")),
                    fmt(row.get("treatment_resid_in_drift")),
                    fmt(row.get("delta_resid_in_drift")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "COMPARE_METRIC_KEYS",
    "CompareError",
    "CompareResult",
    "load_experiment_rows",
    "build_comparison",
    "write_comparison_reports",
]
