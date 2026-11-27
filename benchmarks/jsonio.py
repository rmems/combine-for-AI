"""Standard-JSON writing, shared by every report writer.

Re-exports the packaged implementation so dev checkouts keep a single source
of truth under ``peira``.
"""

from __future__ import annotations

from peira.jsonio import ensure_dir, json_safe, write_json

__all__ = ["ensure_dir", "json_safe", "write_json"]
