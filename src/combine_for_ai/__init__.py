"""combine-for-ai package entrypoints."""

from __future__ import annotations


def main() -> None:
    """Console script entry: delegate to the real benchmark CLI."""
    from benchmarks.cli import main as cli_main

    cli_main()
