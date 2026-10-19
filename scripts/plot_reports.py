#!/usr/bin/env python3
"""Regenerate optional plots and summaries without running benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from combine_for_ai.plot_reports import generate_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        action="append",
        required=True,
        help="Archived JSON (repeatable); prepared summaries or legacy GOZ1",
    )
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("reports"))
    parser.add_argument(
        "--config", type=Path, default=_ROOT / "configs" / "plots.sample.json"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Generate summaries without Matplotlib"
    )
    args = parser.parse_args(argv)
    try:
        written = generate_reports(
            args.input, args.output_dir, args.config, plots=not args.no_plots
        )
    except (ValueError, OSError, ValidationError) as exc:
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 1
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
