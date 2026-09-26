"""Scalar parsers for corinth-canal telemetry artifacts."""

from __future__ import annotations

import math
from typing import Any


def _cell(row: dict[str, str | None], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _finite_or_none(number: float) -> float | None:
    if not math.isfinite(number):
        return None
    return number


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return _finite_or_none(float(raw))
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return _finite_or_none(float(value))
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_str(*values: Any) -> str | None:
    for value in values:
        parsed = _as_str(value)
        if parsed is not None:
            return parsed
    return None


def _float_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[float] = []
    for item in value:
        parsed = _as_float(item)
        if parsed is not None:
            out.append(parsed)
    return tuple(out)
