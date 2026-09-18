"""Apple Metal GPU telemetry collector."""

from __future__ import annotations

import platform
import plistlib
import re
from collections.abc import Mapping
from typing import Any, cast

from benchmarks.gpu_telemetry_common import (
    CommandRunner,
    GPUMetrics,
    GPUPlatform,
    _bytes_or_mb_to_mb,
    _coerce_float,
    _coerce_int,
    _loads_json,
    _run_command,
    ignore_probe_error,
)

POWERMETRICS_COMMAND = [
    "powermetrics",
    "-n",
    "1",
    "-i",
    "1000",
    "--samplers",
    "gpu_power",
]

IOREG_COMMAND = ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-a"]
SYSTEM_PROFILER_COMMAND = ["system_profiler", "SPDisplaysDataType", "-json"]


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _coalesce[T](primary: T | None, fallback: T | None) -> T | None:
    return fallback if primary is None else primary


def parse_powermetrics_text(text: str) -> dict[str, float | None]:
    """Extract GPU utilization, power, clock, and bandwidth from powermetrics."""

    power_mw = _match_number(text, r"GPU Power:\s*([\d.]+)\s*mW")
    if power_mw is not None:
        power_w = power_mw / 1000.0
    else:
        power_w = _match_number(text, r"GPU Power:\s*([\d.]+)\s*W")
    return {
        "utilization_percent": _match_number(text, r"GPU HW active residency:\s*([\d.]+)\s*%"),
        "clock_sm_mhz": _match_number(text, r"GPU HW active frequency:\s*([\d.]+)\s*MHz")
        or _match_number(text, r"GPU HW requested frequency:\s*([\d.]+)\s*MHz"),
        "power_draw_w": power_w,
        "memory_bandwidth_gbps": _match_number(text, r"GPU Memory Bandwidth:\s*([\d.]+)\s*GB/s"),
    }


def _match_number(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return _coerce_float(match.group(1))


def parse_ioreg_accelerator(raw: str | bytes) -> list[dict[str, Any]]:
    """Parse ``ioreg -c IOAccelerator -a`` plist output."""

    blob = raw.encode("utf-8") if isinstance(raw, str) else raw
    try:
        parsed = plistlib.loads(blob)
    except Exception as exc:
        ignore_probe_error(exc)
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _ioreg_stats(entry: Mapping[str, Any]) -> dict[str, Any]:
    stats = entry.get("PerformanceStatistics")
    if not isinstance(stats, Mapping):
        stats = {}
    used_bytes = _coerce_float(
        _first_present(
            stats,
            "In use system memory",
            "In use Tiled memory",
            "vramUsedBytes",
            "Alloc device memory",
        )
    )
    # Allocated IOAccelerator bytes are not installed unified-memory capacity.
    total_bytes = _coerce_float(_first_present(stats, "vramTotalBytes"))
    return {
        "name": str(
            _first_present(entry, "model", "IOAccelClass", "CFBundleIdentifier") or "Apple GPU"
        ),
        "utilization_percent": _coerce_float(
            _first_present(
                stats,
                "Device Utilization %",
                "Renderer Utilization %",
                "gpu-core-utilization",
            )
        ),
        "memory_used_mb": (
            _bytes_or_mb_to_mb(used_bytes, treat_as_bytes=True) if used_bytes is not None else None
        ),
        "memory_total_mb": (
            _bytes_or_mb_to_mb(total_bytes, treat_as_bytes=True) if total_bytes is not None else None
        ),
    }


def _apple_name_from_profiler(text: str) -> str | None:
    payload = _loads_json(text)
    if not isinstance(payload, dict):
        return None
    displays = payload.get("SPDisplaysDataType")
    if not isinstance(displays, list):
        return None
    for item in displays:
        if not isinstance(item, dict):
            continue
        model = item.get("sppci_model") or item.get("_name")
        if model:
            return str(model)
    return None


def _empty_power_stats() -> dict[str, float | None]:
    return {
        "utilization_percent": None,
        "clock_sm_mhz": None,
        "power_draw_w": None,
        "memory_bandwidth_gbps": None,
    }


def _apple_metric(
    index: int,
    name: str,
    *,
    utilization_percent: float | None,
    memory_used_mb: int | None,
    memory_total_mb: int | None,
    power_stats: Mapping[str, float | None],
) -> GPUMetrics:
    return GPUMetrics(
        index=index,
        name=name,
        utilization_percent=utilization_percent,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        temperature_c=None,
        power_draw_w=power_stats.get("power_draw_w"),
        clock_sm_mhz=_coerce_int(power_stats.get("clock_sm_mhz")),
        clock_memory_mhz=None,
        vendor=GPUPlatform.APPLE.value,
        memory_bandwidth_gbps=power_stats.get("memory_bandwidth_gbps"),
    )


class AppleMetalTelemetryCollector:
    """Apple Silicon collector via ``ioreg`` and optional ``powermetrics``."""

    vendor = GPUPlatform.APPLE

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        system: str | None = None,
        machine: str | None = None,
    ) -> None:
        self._run = runner if runner is not None else _run_command
        self._system = system
        self._machine = machine

    def available(self) -> bool:
        system = self._system if self._system is not None else platform.system()
        machine = self._machine if self._machine is not None else platform.machine()
        if system != "Darwin":
            return False
        if machine.lower() in {"arm64", "arm64e"}:
            return True
        return self._translated_rosetta()

    def _translated_rosetta(self) -> bool:
        output = self._run(["sysctl", "-n", "sysctl.proc_translated"], timeout=2.0)
        return output is not None and output.strip() == "1"

    def collect_info(self) -> tuple[int | None, list[str] | None, str | None]:
        metrics = self.collect_metrics()
        if not metrics:
            return None, None, None
        return len(metrics), [item.name for item in metrics], None

    def collect_metrics(self) -> list[GPUMetrics] | None:
        name = self._gpu_name()
        ioreg_rows = self._from_ioreg()
        power_stats = self._from_powermetrics()
        host_total = self._unified_memory_total() if self.available() else None
        if ioreg_rows:
            return self._metrics_from_ioreg(name, ioreg_rows, power_stats, host_total)
        if name is None and not any(power_stats.values()):
            return None
        return [
            _apple_metric(
                0,
                name or "Apple GPU",
                utilization_percent=power_stats.get("utilization_percent"),
                memory_used_mb=None,
                memory_total_mb=host_total,
                power_stats=power_stats,
            )
        ]

    def _metrics_from_ioreg(
        self,
        name: str | None,
        ioreg_rows: list[dict[str, Any]],
        power_stats: Mapping[str, float | None],
        host_total: int | None,
    ) -> list[GPUMetrics]:
        return [
            _apple_metric(
                index,
                name or str(row["name"]),
                utilization_percent=cast(float | None, row["utilization_percent"]),
                memory_used_mb=cast(int | None, row["memory_used_mb"]),
                memory_total_mb=_coalesce(cast(int | None, row["memory_total_mb"]), host_total),
                power_stats=power_stats,
            )
            for index, row in enumerate(ioreg_rows)
        ]

    def _gpu_name(self) -> str | None:
        output = self._run(SYSTEM_PROFILER_COMMAND, timeout=12.0)
        if output is None:
            return None
        return _apple_name_from_profiler(output)

    def _from_ioreg(self) -> list[dict[str, Any]]:
        output = self._run(IOREG_COMMAND, timeout=8.0)
        if output is None:
            return []
        return [_ioreg_stats(entry) for entry in parse_ioreg_accelerator(output)]

    def _from_powermetrics(self) -> dict[str, float | None]:
        output = self._run(POWERMETRICS_COMMAND, timeout=8.0)
        if output is None:
            return _empty_power_stats()
        return parse_powermetrics_text(output)

    def _unified_memory_total(self) -> int | None:
        try:
            import psutil

            return int(psutil.virtual_memory().total // (1024 * 1024))
        except Exception as exc:
            ignore_probe_error(exc)
            return None
