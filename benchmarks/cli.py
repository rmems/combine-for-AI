from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.runner import run_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the combine-for-AI quantization benchmark harness."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the benchmark config JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write report outputs (default: reports).",
    )
    parser.add_argument(
        "--formats",
        default="json,csv",
        help="Comma-separated list of output formats (json,csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed for the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    run_benchmarks(args.config, args.output_dir, formats, args.seed)


if __name__ == "__main__":
    main()
