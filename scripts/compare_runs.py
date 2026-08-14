#!/usr/bin/env python3
"""Compare hybrid quant experiment rows (issue #15)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from combine_for_ai.compare import (  # noqa: E402
    CompareError,
    build_comparison,
    load_experiment_rows,
    write_comparison_reports,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compare hybrid MoE/SNN experiment rows (e.g. fp16_control vs "
            "expert_only across blocks). Accepts goz-import reports or raw "
            "grok-ozempic experiment JSON."
        )
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        action="append",
        required=True,
        help="Input JSON path (repeatable): goz-import report or experiment file",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports"),
        help="Report root (json/csv/markdown subdirs)",
    )
    p.add_argument(
        "--baseline-arm",
        default="fp16_control",
        help="Baseline arm name (default: fp16_control)",
    )
    p.add_argument(
        "--treatment-arm",
        default="expert_only",
        help="Treatment arm name (default: expert_only)",
    )
    p.add_argument(
        "--arms",
        default=None,
        help="Optional comma filter of arms to load (default: all in inputs)",
    )
    p.add_argument(
        "--formats",
        default="json,csv,markdown",
        help="Comma-separated outputs: json,csv,markdown",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Optional run id (default: timestamped)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = None
    if args.arms:
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    run_id = args.run_id or (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}."
        f"{int(time.time() % 1 * 1000):03d}Z-compare"
    )
    try:
        rows = load_experiment_rows(args.input, arms=arms)
        result = build_comparison(
            rows,
            baseline_arm=args.baseline_arm,
            treatment_arm=args.treatment_arm,
        )
        written = write_comparison_reports(
            result,
            args.output_dir,
            run_id=run_id,
            formats=formats,
        )
    except CompareError as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"compare arms={result.arms} blocks={len(result.by_block)} "
        f"rows={len(result.rows)}"
    )
    for fmt, path in written.items():
        print(f"  {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
