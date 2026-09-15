from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.runner import run_benchmarks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--corinth-canal-dir",
        type=Path,
        default=None,
        help=(
            "Directory of corinth-canal SAAQ run artifacts "
            "(latent_telemetry.csv, summary.json, run_manifest.json). "
            "Overrides telemetry.corinth_canal_dir / telemetry.corinth_canal_path in the config. "
            "Missing files are skipped."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    run_benchmarks(
        args.config,
        args.output_dir,
        formats,
        args.seed,
        corinth_canal_dir=args.corinth_canal_dir,
    )


if __name__ == "__main__":
    main()
