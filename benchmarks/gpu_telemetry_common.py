"""GPU telemetry collectors for NVIDIA, AMD ROCm, and Apple Metal.

Hardware backends are optional. Missing libraries or CLIs produce a warning
and omit GPU fields rather than failing the benchmark run.

Supported devices and system packages
--------------------------------------
NVIDIA (any NVML device, e.g. RTX 4090, H100, B200)
    Python package: ``nvidia-ml-py`` (imports as ``pynvml``)

AMD ROCm (MI300X, MI250X, RX 7900 XTX, and other ROCm devices)
    Preferred: ``amdsmi`` from the ROCm AMD SMI Python bindings
    Fallback: ``rocm-smi`` CLI (``rocm-smi-lib`` / ROCm system package)

Apple Silicon (M3 Max, M4 Ultra, and other unified-memory SoCs)
    ``ioreg`` (always present on macOS) for utilization and memory
    ``powermetrics`` (optional; often requires privileges) for power,
    GPU core clock, and memory bandwidth
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class GPUPlatform(str, Enum):
    """Detected GPU telemetry backend."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    NONE = "none"


class GPUTelemetryUnavailableWarning(UserWarning):
    """No GPU telemetry backend could be initialized."""


@dataclass(frozen=True)
class GPUMetrics:
    """Per-GPU telemetry in a shared schema across NVIDIA, AMD, and Apple.

    ``clock_sm_mhz`` is the SM clock on NVIDIA, the GFX/sclk clock on AMD, and
    the GPU core clock on Apple Silicon.

    ``memory_used_mb`` / ``memory_total_mb`` are VRAM on NVIDIA and AMD, and
    unified memory on Apple Silicon.

    ``memory_bandwidth_gbps`` is populated on Apple Metal when powermetrics
    reports it; NVIDIA and AMD leave it unset.
    """

    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: int | None
    memory_total_mb: int | None
    temperature_c: int | None
    power_draw_w: float | None
    clock_sm_mhz: int | None
    clock_memory_mhz: int | None
    vendor: str | None = None
    memory_bandwidth_gbps: float | None = None


CommandRunner = Callable[..., str | None]


_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")


def ignore_probe_error(exc: BaseException) -> None:
    _ = (type(exc).__name__, str(exc))


_ALLOWED_PROBES = frozenset(
    {"ioreg", "powermetrics", "rocm-smi", "sysctl", "system_profiler"}
)


def _resolved_probe(name: str) -> str | None:
    if name not in _ALLOWED_PROBES:
        return None
    found = shutil.which(name)
    if found is None:
        return None
    path = Path(found)
    if not path.is_absolute():
        return None
    return str(path)


def _run_command(command: list[str], *, timeout: float = 5.0) -> str | None:
    if not command:
        return None
    resolved = _resolved_probe(command[0])
    if resolved is None:
        return None
    argv = [resolved, *command[1:]]
    try:
        # Absolute argv[0], shell=False, allowlisted probe names only.
        return subprocess.check_output(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            argv,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _finite_or_none(number: float) -> float | None:
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _coerce_mapped_float(value: Mapping[str, Any]) -> float | None:
    for key in ("value", "current", "avg", "average", "val"):
        if key in value:
            return _coerce_float(value[key])
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _finite_or_none(float(value))
    if isinstance(value, Mapping):
        return _coerce_mapped_float(value)
    match = _NUMBER.search(str(value).replace(",", ""))
    if match is None:
        return None
    try:
        return _finite_or_none(float(match.group(0)))
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(round(number))


def _lookup(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _bytes_or_mb_to_mb(value: float | None, *, treat_as_bytes: bool) -> int | None:
    if value is None:
        return None
    # 8 TiB in MiB is 8,388,608; any larger reading must be bytes, not megabytes.
    if treat_as_bytes or value >= 8 * 1024 * 1024:
        return int(value // (1024 * 1024))
    return int(round(value))


def _loads_json(text: str) -> Any | None:
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [index for index in (start_obj, start_arr) if index >= 0]
    if not starts:
        return None
    try:
        return json.loads(text[min(starts) :])
    except json.JSONDecodeError:
        return None


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            next_key = f"{prefix} {key}".strip() if prefix else str(key)
            if isinstance(value, Mapping):
                out.update(_flatten(value, next_key))
            else:
                out[next_key] = value
        return out
    if prefix:
        out[prefix] = obj
    return out


def _first_matching_with_key(
    flat: Mapping[str, Any],
    *required_substrings: str,
    exclude: tuple[str, ...] = (),
) -> tuple[str | None, Any]:
    required = tuple(item.lower() for item in required_substrings)
    skipped = tuple(item.lower() for item in exclude)
    for key, value in flat.items():
        haystack = key.lower()
        if all(part in haystack for part in required) and not any(part in haystack for part in skipped):
            return key, value
    return None, None


def _first_matching(
    flat: Mapping[str, Any],
    *required_substrings: str,
    exclude: tuple[str, ...] = (),
) -> Any:
    _key, value = _first_matching_with_key(flat, *required_substrings, exclude=exclude)
    return value


def _key_looks_like_bytes(key: str | None) -> bool:
    if key is None:
        return False
    haystack = key.lower()
    return "(b)" in haystack or "bytes" in haystack

