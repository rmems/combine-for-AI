#!/usr/bin/env python3
"""Run a cross-model experiment matrix (RM-105) or a resume journal (RM-1319)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from benchmarks.journal import replay_journal  # noqa: E402
from benchmarks.matrix import RetryPolicy  # noqa: E402
from benchmarks.matrix_runner import load_matrix_config, run_matrix  # noqa: E402
from combine_for_ai.matrix import (  # noqa: E402
    MatrixError as ExperimentMatrixError,
    MatrixSelection,
    load_experiment_matrix,
)
from combine_for_ai.matrix_runner import MatrixRunner, MatrixRunOptions  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a models × quantization × datasets experiment matrix and write "
            "individual cell reports plus a unified comparison report. "
            "Journal-style configs use the append-only resume journal."
        )
    )
    _add_io_args(parser)
    _add_filter_args(parser)
    _add_run_args(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Matrix config path (.json or .toml).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports"),
        help="Report root (json/csv/markdown plus cells/ and progress or journal).",
    )
    parser.add_argument(
        "--formats",
        default="json,csv,markdown",
        help="Comma-separated outputs: json,csv,markdown",
    )


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default=None, help="Comma-separated model names to run.")
    parser.add_argument(
        "--families", default=None, help="Comma-separated architecture families to run."
    )
    parser.add_argument(
        "--quant-methods",
        default=None,
        help="Comma-separated quantization methods to run.",
    )
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated dataset names to run."
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="Override matrix seed.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing progress and rerun every selected cell.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed cell.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional matrix run id (default: timestamped).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Requeue failed journal cells that have remaining attempts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum attempts per journal cell when retrying failed cells.",
    )


def _csv_set(raw: str | None) -> frozenset[str] | None:
    if not raw:
        return None
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _parse_formats(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _is_journal_config(path: Path) -> bool:
    if path.suffix.lower() == ".toml":
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return False
    if raw.get("matrix_version") or raw.get("baseline_quantization"):
        return False
    return True


def _journal_retry(args: argparse.Namespace) -> RetryPolicy | None:
    if not args.retry_failed and args.max_attempts is None:
        return None
    base = load_matrix_config(args.config).retry
    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(base.max_attempts, 2)
    return RetryPolicy(
        retry_failed=True if args.retry_failed else base.retry_failed,
        max_attempts=max_attempts,
    )


def _run_journal_cli(args: argparse.Namespace) -> int:
    result = run_matrix(args.config, args.output_dir, retry=_journal_retry(args))
    replay = replay_journal(result.journal_path)
    print(f"matrix={result.definition.name}")
    print(f"journal={result.journal_path}")
    print(f"aggregate={result.aggregate_path}")
    print(f"truncated_tail_recovered={result.truncated_tail}")
    print(f"records={len(replay.records)}")
    print(f"counts={result.aggregate['counts']}")
    for cell in result.aggregate["cells"]:
        print(
            f"  {cell['cell_id'][:12]} {cell['quantization']}/{cell['dataset']} "
            f"state={cell['state']} retries={cell['retry_count']} "
            f"disposition={cell['disposition']}"
        )
    if int(result.aggregate["counts"].get("failed", 0)) > 0:
        return 1
    return 0


def _run_experiment_cli(args: argparse.Namespace) -> int:
    extra = MatrixSelection(
        models=_csv_set(args.models),
        families=_csv_set(args.families),
        quantization=_csv_set(args.quant_methods),
        datasets=_csv_set(args.datasets),
    )
    try:
        matrix = load_experiment_matrix(args.config, extra_select=extra)
        runner = MatrixRunner(
            matrix,
            args.output_dir,
            options=MatrixRunOptions(
                formats=_parse_formats(args.formats),
                resume=not args.fresh,
                fail_fast=args.fail_fast,
                run_id=args.run_id,
                seed=args.seed,
            ),
        )
        report = runner.run()
    except ExperimentMatrixError as exc:
        print(f"matrix failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"matrix failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"matrix name={report.matrix_name} cells={len(report.cells)} "
        f"completed={report.completed} failed={report.failed} "
        f"families={len(report.by_family)}"
    )
    print(f"  progress: {runner.progress_path}")
    if report.failed > 0:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _is_journal_config(args.config):
        return _run_journal_cli(args)
    return _run_experiment_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
