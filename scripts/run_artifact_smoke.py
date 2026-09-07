from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from benchmarks.artifact_smoke import run_artifact_smoke  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-command artifact smoke benchmark from a magere-brug manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to a magere-brug JSON manifest.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory for JSON/CSV outputs.")
    parser.add_argument("--formats", default="json,csv", help="Comma-separated output formats (json,csv).")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional dataset JSONL override.")
    parser.add_argument("--max-samples", type=int, default=2, help="Maximum dataset samples for the smoke run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic mock evaluation.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    caller_cwd = Path.cwd()
    dataset_path = args.dataset
    if dataset_path is not None and not dataset_path.is_absolute():
        dataset_path = (caller_cwd / dataset_path).resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (caller_cwd / output_dir).resolve()
    import os
    os.chdir(REPO_ROOT)
    payload = run_artifact_smoke(
        manifest_path=args.manifest,
        output_dir=output_dir,
        formats=formats,
        dataset_path=dataset_path,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(payload["result"]["status"])
    if payload["result"]["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
