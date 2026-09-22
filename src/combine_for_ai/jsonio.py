"""Standard-JSON writing for packaged combine_for_ai report writers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a report as standard JSON that any conforming parser can read."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False
        )


__all__ = ["ensure_dir", "json_safe", "write_json"]
