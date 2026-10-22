"""Numeric guards for legacy measurements. Absent values are not decided here."""

from __future__ import annotations

import math
from typing import Any


def is_nonnegative_int(value: object) -> bool:
    # bool is an int subclass; JSON true must not become block index 1.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def require_finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"malformed metric {label}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"malformed metric {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric {label}")
    return number
