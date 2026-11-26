"""Errors for SWE-bench-Live ingest and ranking."""


class LiveBaselineError(ValueError):
    """Invalid ingest input or baseline registry."""


__all__ = ["LiveBaselineError"]
