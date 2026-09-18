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

import warnings
from collections.abc import Sequence
from typing import Never, Protocol

from benchmarks.gpu_telemetry_amd import AMDGPUTelemetryCollector, parse_rocm_smi_json
from benchmarks.gpu_telemetry_apple import (
    AppleMetalTelemetryCollector,
    parse_ioreg_accelerator,
    parse_powermetrics_text,
)
from benchmarks.gpu_telemetry_common import (
    GPUMetrics,
    GPUPlatform,
    GPUTelemetryUnavailableWarning,
)
from benchmarks.gpu_telemetry_nvidia import NVIDIAGPUTelemetryCollector

_MISSING_BACKEND_MESSAGE = (
    "No GPU telemetry backend is available. Install nvidia-ml-py (NVIDIA), "
    "amdsmi or the rocm-smi CLI (AMD ROCm), or run on Apple Silicon with ioreg "
    "and optionally powermetrics. GPU fields will be omitted."
)


class _GPUBackend(Protocol):
    vendor: GPUPlatform

    def available(self) -> bool: ...


def _assert_never(value: Never) -> None:
    raise RuntimeError(f"unhandled GPU platform: {value!r}")


def detect_gpu_platform(
    collectors: Sequence[_GPUBackend] | None = None,
) -> GPUPlatform:
    """Return the first available GPU backend: Apple, NVIDIA, AMD, or none."""

    sequence: Sequence[_GPUBackend]
    if collectors is None:
        sequence = (
            AppleMetalTelemetryCollector(),
            NVIDIAGPUTelemetryCollector(),
            AMDGPUTelemetryCollector(),
        )
    else:
        sequence = collectors
    for collector in sequence:
        if collector.available():
            return collector.vendor
    return GPUPlatform.NONE


def _collector_for(
    platform: GPUPlatform,
) -> NVIDIAGPUTelemetryCollector | AMDGPUTelemetryCollector | AppleMetalTelemetryCollector | None:
    match platform:
        case GPUPlatform.NONE:
            return None
        case GPUPlatform.NVIDIA:
            return NVIDIAGPUTelemetryCollector()
        case GPUPlatform.AMD:
            return AMDGPUTelemetryCollector()
        case GPUPlatform.APPLE:
            return AppleMetalTelemetryCollector()
        case _ as unreachable:
            _assert_never(unreachable)


def collect_gpu_info() -> tuple[int | None, list[str] | None, str | None]:
    """GPU count, names, and driver for the detected platform."""

    collector = _collector_for(detect_gpu_platform())
    if collector is None:
        return None, None, None
    try:
        return collector.collect_info()
    except Exception as exc:
        warnings.warn(
            f"{collector.vendor.value} GPU info collection failed: {exc}",
            GPUTelemetryUnavailableWarning,
            stacklevel=2,
        )
        return None, None, None


def collect_gpu_metrics() -> list[GPUMetrics] | None:
    """Per-GPU telemetry for the detected platform, or None if unavailable."""

    platform = detect_gpu_platform()
    collector = _collector_for(platform)
    if collector is None:
        warnings.warn(_MISSING_BACKEND_MESSAGE, GPUTelemetryUnavailableWarning, stacklevel=2)
        return None
    try:
        metrics = collector.collect_metrics()
    except Exception as exc:
        warnings.warn(
            f"{platform.value} GPU telemetry collection failed: {exc}",
            GPUTelemetryUnavailableWarning,
            stacklevel=2,
        )
        return None
    if not metrics:
        warnings.warn(
            f"{platform.value} GPU telemetry collector returned no metrics",
            GPUTelemetryUnavailableWarning,
            stacklevel=2,
        )
        return None
    return metrics


def apple_snapshot_notes(gpu_metrics: list[GPUMetrics] | None) -> str | None:
    """Document Apple-specific field meanings when Metal metrics are present."""

    if not gpu_metrics:
        return None
    if any(item.vendor == GPUPlatform.APPLE.value for item in gpu_metrics):
        return (
            "Apple Silicon: memory_used_mb/memory_total_mb are unified memory; "
            "clock_sm_mhz is GPU core clock; memory_bandwidth_gbps is a "
            "platform-specific field from powermetrics when available."
        )
    return None


def bandwidth_from_metrics(gpu_metrics: list[GPUMetrics] | None) -> float | None:
    if not gpu_metrics:
        return None
    values = [item.memory_bandwidth_gbps for item in gpu_metrics if item.memory_bandwidth_gbps is not None]
    if not values:
        return None
    return max(values)


__all__ = [
    "AMDGPUTelemetryCollector",
    "AppleMetalTelemetryCollector",
    "GPUMetrics",
    "GPUPlatform",
    "GPUTelemetryUnavailableWarning",
    "NVIDIAGPUTelemetryCollector",
    "apple_snapshot_notes",
    "bandwidth_from_metrics",
    "collect_gpu_info",
    "collect_gpu_metrics",
    "detect_gpu_platform",
    "parse_ioreg_accelerator",
    "parse_powermetrics_text",
    "parse_rocm_smi_json",
]
