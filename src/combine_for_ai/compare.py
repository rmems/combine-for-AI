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
    "seconds",
)

_META_KEYS = (
    "baseline_label",
    "baseline_scale_source",
    "baseline_goz1_version",
    "treatment_label",
    "treatment_scale_source",
    "treatment_goz1_version",
    "treatment_pack_basename",
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


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompareError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CompareError(f"input root must be a JSON object: {path}")
    return raw


def _rows_from_goz_import_report(path: Path, raw: dict[str, Any]) -> list[dict[str, Any]]:
    results = raw.get("results")
    if not isinstance(results, list) or not results:
        raise CompareError(f"goz-import report missing results[]: {path}")
    rows = []
    for item in results:
        if isinstance(item, dict):
            row = dict(item)
            row.setdefault("source_path", str(path))
            rows.append(row)
    if not rows:
        raise CompareError(f"no dict rows in results[]: {path}")
    return rows


def _rows_from_experiment(
    path: Path, arms: tuple[str, ...] | None
) -> list[dict[str, Any]]:
    try:
        imported = import_goz_experiment(
            path, arms=arms or ("expert_only", "fp16_control")
        )
    except GozImportError as exc:
        raise CompareError(f"import failed for {path}: {exc}") from exc
    return [row.to_report_row() for row in imported.rows]


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
        raw = _load_json_object(path)
        if isinstance(raw.get("results"), list):
            all_rows.extend(_rows_from_goz_import_report(path, raw))
        else:
            all_rows.extend(_rows_from_experiment(path, arms))
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


def _index_rows_by_block_arm(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, str], dict[str, Any]]:
    by_key: dict[tuple[Any, str], dict[str, Any]] = {}
    for row in rows:
        arm = row.get("arm")
        if arm is None:
            continue
        key = (row.get("block_index"), str(arm))
        if key in by_key:
            raise CompareError(
                f"duplicate row for block={key[0]}, arm={key[1]}: "
                "cannot determine which to use for comparison"
            )
        by_key[key] = row
    return by_key


def _block_order(by_key: dict[tuple[Any, str], dict[str, Any]]) -> list[Any]:
    blocks = sorted(
        {block for (block, _) in by_key if block is not None},
        key=lambda b: (isinstance(b, str), b),
    )
    if any(block is None for (block, _) in by_key):
        return [None, *blocks]
    return blocks


def _fill_metric_deltas(
    entry: dict[str, Any],
    base: dict[str, Any] | None,
    treat: dict[str, Any] | None,
) -> None:
    for key in COMPARE_METRIC_KEYS:
        bval = _numeric(base.get(key)) if base else None
        tval = _numeric(treat.get(key)) if treat else None
        entry[f"baseline_{key}"] = bval
        entry[f"treatment_{key}"] = tval
        entry[f"delta_{key}"] = _delta(tval, bval)


def _fill_arm_meta(
    entry: dict[str, Any],
    base: dict[str, Any] | None,
    treat: dict[str, Any] | None,
) -> None:
    for mk in _META_KEYS:
        entry[mk] = None
    if base:
        entry["baseline_label"] = base.get("label")
        entry["baseline_scale_source"] = base.get("scale_source")
        entry["baseline_goz1_version"] = base.get("goz1_version")
    if treat:
        entry["treatment_label"] = treat.get("label")
        entry["treatment_scale_source"] = treat.get("scale_source")
        entry["treatment_goz1_version"] = treat.get("goz1_version")
        entry["treatment_pack_basename"] = treat.get("pack_basename")


def _pair_block_entry(
    block: Any,
    by_key: dict[tuple[Any, str], dict[str, Any]],
    baseline_arm: str,
    treatment_arm: str,
) -> dict[str, Any] | None:
    base = by_key.get((block, baseline_arm))
    treat = by_key.get((block, treatment_arm))
    if base is None or treat is None:
        return None
    _require_matching_context(block, base, treat)
    entry: dict[str, Any] = {
        "block_index": block,
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
    }
    _fill_metric_deltas(entry, base, treat)
    _fill_arm_meta(entry, base, treat)
    return entry


def _require_matching_context(
    block: Any, base: dict[str, Any], treat: dict[str, Any]
) -> None:
    """Reject a same-block arm pair when its experiment context differs."""
    context_keys = ("model_family", "tokens", "seed")
    mismatches = [
        key for key in context_keys if base.get(key) != treat.get(key)
    ]
    if mismatches:
        raise CompareError(
            f"incompatible comparison context for block={block}: "
            + ", ".join(mismatches)
        )


def _sort_key_block_arm(row: dict[str, Any]) -> tuple[Any, ...]:
    block = row.get("block_index")
    return (
        block is not None,
        block if block is not None else -1,
        str(row.get("arm") or ""),
    )


def _require_both_arms(
    by_key: dict[tuple[Any, str], dict[str, Any]],
    baseline_arm: str,
    treatment_arm: str,
) -> None:
    has_baseline = any(arm == baseline_arm for (_, arm) in by_key)
    has_treatment = any(arm == treatment_arm for (_, arm) in by_key)
    if not has_baseline or not has_treatment:
        raise CompareError(
            f"comparison requires both arms present: "
            f"baseline={baseline_arm!r} found={has_baseline}, "
            f"treatment={treatment_arm!r} found={has_treatment}"
        )


def _build_by_block_rows(
    by_key: dict[tuple[Any, str], dict[str, Any]],
    baseline_arm: str,
    treatment_arm: str,
) -> list[dict[str, Any]]:
    by_block: list[dict[str, Any]] = []
    for block in _block_order(by_key):
        entry = _pair_block_entry(block, by_key, baseline_arm, treatment_arm)
        if entry is not None:
            by_block.append(entry)
    if not by_block:
        raise CompareError("no baseline/treatment block pairs to compare")
    return by_block


def build_comparison(
    rows: list[dict[str, Any]],
    *,
    baseline_arm: str = "fp16_control",
    treatment_arm: str = "expert_only",
) -> CompareResult:
    """Pair baseline vs treatment by block_index; emit flat rows + by_block deltas."""
    if baseline_arm == treatment_arm:
        raise CompareError("baseline and treatment arms must be different")
    arms = sorted({baseline_arm, treatment_arm})
    sources = sorted({str(r.get("source_path")) for r in rows if r.get("source_path")})
    by_key = _index_rows_by_block_arm(rows)
    _require_both_arms(by_key, baseline_arm, treatment_arm)
    by_block = _build_by_block_rows(by_key, baseline_arm, treatment_arm)
    flat = sorted(rows, key=_sort_key_block_arm)
    return CompareResult(rows=flat, by_block=by_block, sources=sources, arms=arms)


def _write_json(result: CompareResult, output_dir: Path, run_id: str) -> Path:
    from benchmarks.reporting import write_json

    payload = result.to_payload()
    payload["run_id"] = run_id
    jpath = output_dir / "json" / f"{run_id}.compare.json"
    write_json(jpath, payload)
    return jpath


def _write_csv(result: CompareResult, output_dir: Path, run_id: str) -> Path:
    from benchmarks.reporting import write_csv

    rows = result.by_block or result.rows
    if not rows:
        raise CompareError("no rows to write for csv")
    suffix = "by-block" if result.by_block else "rows"
    cpath = output_dir / "csv" / f"{run_id}.compare-{suffix}.csv"
    write_csv(cpath, rows)
    return cpath


def _write_markdown(result: CompareResult, output_dir: Path, run_id: str) -> Path:
    mpath = output_dir / "markdown" / f"{run_id}.compare.md"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(_markdown_table(result), encoding="utf-8")
    return mpath


def write_comparison_reports(
    result: CompareResult,
    output_dir: Path,
    *,
    run_id: str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    if formats is None:
        formats = ["json", "csv", "markdown"]
    if not formats:
        raise CompareError("no report formats selected")
    output_dir = Path(output_dir)
    writers = {
        "json": _write_json,
        "csv": _write_csv,
        "markdown": _write_markdown,
    }
    written: dict[str, Path] = {}
    try:
        for fmt in formats:
            writer = writers.get(fmt)
            if writer is None:
                raise CompareError(f"unsupported report format: {fmt}")
            written[fmt] = writer(result, output_dir, run_id)
    except OSError as exc:
        raise CompareError(f"failed to write reports: {exc}") from exc
    except ValueError as exc:
        raise CompareError(f"failed to serialize reports: {exc}") from exc
    return written


def _markdown_table(result: CompareResult) -> str:
    lines = ["# Hybrid quant comparison", "", f"Sources: {len(result.sources)}", ""]
    if not result.by_block:
        lines.append("_No baseline/treatment block pairs found._")
        lines.append("")
        return "\n".join(lines)

    lines[2:2] = [
        f"Baseline arm: {result.by_block[0]['baseline_arm']}; "
        f"Treatment arm: {result.by_block[0]['treatment_arm']}",
    ]

    headers = [
        "block",
        "top1_base",
        "top1_treat",
        "d_top1",
        "cos_base",
        "cos_treat",
        "d_cos",
        "resid_base",
        "resid_treat",
        "d_resid",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def fmt(v: Any) -> str:
        if v is None:
            return "-"
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
