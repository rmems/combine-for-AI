"""NVIDIA NVML GPU telemetry collector."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchmarks.gpu_telemetry_common import (
    GPUMetrics,
    GPUPlatform,
    ignore_probe_error,
)


def _nvml_name(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _nvml_optional(probe: Callable[[], Any]) -> Any:
    try:
        return probe()
    except Exception as exc:
        ignore_probe_error(exc)
        return None


def _nvml_memory_mb(nvml: Any, handle: Any) -> tuple[int | None, int | None]:
    mem = _nvml_optional(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
    if mem is None:
        return None, None
    return mem.used // (1024 * 1024), mem.total // (1024 * 1024)


def _nvml_device_metrics(
    nvml: Any,
    index: int,
    handle: Any,
    temp_sensor: Any,
    clock_sm: Any,
    clock_mem: Any,
) -> GPUMetrics:
    util_rates = _nvml_optional(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
    power_mw = _nvml_optional(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
    mem_used, mem_total = _nvml_memory_mb(nvml, handle)
    return GPUMetrics(
        index=index,
        name=_nvml_name(nvml.nvmlDeviceGetName(handle)),
        utilization_percent=getattr(util_rates, "gpu", None),
        memory_used_mb=mem_used,
        memory_total_mb=mem_total,
        temperature_c=_nvml_optional(lambda: nvml.nvmlDeviceGetTemperature(handle, temp_sensor)),
        power_draw_w=None if power_mw is None else power_mw / 1000.0,
        clock_sm_mhz=_nvml_optional(lambda: nvml.nvmlDeviceGetClockInfo(handle, clock_sm)),
        clock_memory_mhz=_nvml_optional(lambda: nvml.nvmlDeviceGetClockInfo(handle, clock_mem)),
        vendor=GPUPlatform.NVIDIA.value,
    )


class NVIDIAGPUTelemetryCollector:
    """NVML collector via ``pynvml`` (package ``nvidia-ml-py``)."""

    vendor = GPUPlatform.NVIDIA

    def __init__(self, nvml: Any | None = None) -> None:
        self._nvml = nvml

    def _load_pynvml(self) -> Any:
        if self._nvml is not None:
            return self._nvml
        import pynvml

        return pynvml

    def available(self) -> bool:
        try:
            nvml = self._load_pynvml()
            nvml.nvmlInit()
            try:
                return nvml.nvmlDeviceGetCount() > 0
            finally:
                nvml.nvmlShutdown()
        except Exception as exc:
            ignore_probe_error(exc)
            return False

    def collect_info(self) -> tuple[int | None, list[str] | None, str | None]:
        try:
            nvml = self._load_pynvml()
            nvml.nvmlInit()
            try:
                count = nvml.nvmlDeviceGetCount()
                names = []
                for index in range(count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(index)
                    names.append(_nvml_name(nvml.nvmlDeviceGetName(handle)))
                driver = _nvml_name(nvml.nvmlSystemGetDriverVersion())
                return count, names, driver
            finally:
                nvml.nvmlShutdown()
        except Exception as exc:
            ignore_probe_error(exc)
            return None, None, None

    def collect_metrics(self) -> list[GPUMetrics] | None:
        try:
            nvml = self._load_pynvml()
            nvml.nvmlInit()
            try:
                count = nvml.nvmlDeviceGetCount()
                temp_sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
                clock_sm = getattr(nvml, "NVML_CLOCK_SM", 0)
                clock_mem = getattr(nvml, "NVML_CLOCK_MEM", 1)
                return [
                    _nvml_device_metrics(
                        nvml,
                        index,
                        nvml.nvmlDeviceGetHandleByIndex(index),
                        temp_sensor,
                        clock_sm,
                        clock_mem,
                    )
                    for index in range(count)
                ]
            finally:
                nvml.nvmlShutdown()
        except Exception as exc:
            ignore_probe_error(exc)
            return None
