"""Standard-JSON writing, shared by every report writer.

Re-exports the packaged implementation so dev checkouts keep a single source
of truth under ``combine_for_ai``.
"""

from __future__ import annotations

from combine_for_ai.jsonio import ensure_dir, json_safe, write_json

__all__ = ["ensure_dir", "json_safe", "write_json"]
