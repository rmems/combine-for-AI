from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from benchmarks.journal import replay_journal  # noqa: E402
from benchmarks.matrix import RetryPolicy  # noqa: E402
from benchmarks.matrix_runner import load_matrix_config, run_matrix  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resume-safe experiment matrix with an append-only journal."
    )
    parser.add_argument("--config", type=Path, required=True, help="Matrix config JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory that stores the journal, cell artifacts, and aggregate.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Requeue failed cells that have remaining attempts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum attempts per cell when retrying failed cells.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    retry = None
    if args.retry_failed or args.max_attempts is not None:
        base = load_matrix_config(args.config).retry
        max_attempts = args.max_attempts
        if max_attempts is None:
            # A failed cell already has attempt >= 1, so a budget of 1 never retries.
            max_attempts = max(base.max_attempts, 2)
        retry = RetryPolicy(
            retry_failed=True if args.retry_failed else base.retry_failed,
            max_attempts=max_attempts,
        )
    result = run_matrix(args.config, args.output_dir, retry=retry)
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


if __name__ == "__main__":
    main()
