#!/usr/bin/env python3
"""Import a grok-ozempic experiment JSON into combine report rows (issue #22)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nfl_combine_for_ai.goz_import import (  # noqa: E402
    GozImportError,
    import_goz_experiment,
    write_import_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import grok-ozempic multiblock metrics.json or route-preservation "
            "JSON into combine CSV/JSON reports."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to grok-ozempic experiment JSON",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports"),
        help="Report root (writes json/ and csv/ subdirs)",
    )
    parser.add_argument(
        "--arms",
        default="expert_only,fp16_control",
        help="Comma-separated multiblock arms to import (default: expert_only,fp16_control)",
    )
    parser.add_argument(
        "--formats",
        default="json,csv",
        help="Comma-separated output formats: json,csv",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id (default: timestamped goz-import id)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    try:
        result = import_goz_experiment(args.input, arms=arms)
        written = write_import_reports(
            result,
            args.output_dir,
            formats=formats,
            run_id=args.run_id,
        )
    except GozImportError as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"imported kind={result.kind.value} schema={result.schema} "
        f"rows={len(result.rows)}"
    )
    for fmt, path in written.items():
        print(f"  {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
