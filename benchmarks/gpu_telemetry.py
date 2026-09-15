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
import platform
import plistlib
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, Protocol, cast


class GPUPlatform(str, Enum):
    """Detected GPU telemetry backend."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    NONE = "none"


class GPUTelemetryUnavailableWarning(UserWarning):
    """No GPU telemetry backend could be initialized."""


class _GPUBackend(Protocol):
    vendor: GPUPlatform

    def available(self) -> bool: ...


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

_MISSING_BACKEND_MESSAGE = (
    "No GPU telemetry backend is available. Install nvidia-ml-py (NVIDIA), "
    "amdsmi or the rocm-smi CLI (AMD ROCm), or run on Apple Silicon with ioreg "
    "and optionally powermetrics. GPU fields will be omitted."
)

ROCM_SMI_FULL_COMMAND = [
    "rocm-smi",
    "--json",
    "--showtemp",
    "--showpower",
    "--showuse",
    "--showmeminfo",
    "vram",
    "--showclocks",
    "--showproductname",
    "--showdriverversion",
]

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


def _assert_never(value: Never) -> None:
    raise RuntimeError(f"unhandled GPU platform: {value!r}")


def _run_command(command: list[str], *, timeout: float = 5.0) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            return None
        return number
    if isinstance(value, Mapping):
        for key in ("value", "current", "avg", "average", "val"):
            if key in value:
                return _coerce_float(value[key])
        return None
    match = _NUMBER.search(str(value).replace(",", ""))
    if match is None:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


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


def _nvml_name(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------


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
                return nvml.nvmlDeviceGetCount() >= 0
            finally:
                nvml.nvmlShutdown()
        except Exception:
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
        except Exception:
            return None, None, None

    def collect_metrics(self) -> list[GPUMetrics] | None:
        try:
            nvml = self._load_pynvml()
            nvml.nvmlInit()
            try:
                metrics: list[GPUMetrics] = []
                count = nvml.nvmlDeviceGetCount()
                temp_sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
                clock_sm = getattr(nvml, "NVML_CLOCK_SM", 0)
                clock_mem = getattr(nvml, "NVML_CLOCK_MEM", 1)
                for index in range(count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(index)
                    name = _nvml_name(nvml.nvmlDeviceGetName(handle))

                    util = None
                    try:
                        util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    except Exception:
                        pass

                    mem_used = None
                    mem_total = None
                    try:
                        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used = mem.used // (1024 * 1024)
                        mem_total = mem.total // (1024 * 1024)
                    except Exception:
                        pass

                    temp = None
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(handle, temp_sensor)
                    except Exception:
                        pass

                    power = None
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        pass

                    sm_clock = None
                    try:
                        sm_clock = nvml.nvmlDeviceGetClockInfo(handle, clock_sm)
                    except Exception:
                        pass

                    mem_clock = None
                    try:
                        mem_clock = nvml.nvmlDeviceGetClockInfo(handle, clock_mem)
                    except Exception:
                        pass

                    metrics.append(
                        GPUMetrics(
                            index=index,
                            name=name,
                            utilization_percent=util,
                            memory_used_mb=mem_used,
                            memory_total_mb=mem_total,
                            temperature_c=temp,
                            power_draw_w=power,
                            clock_sm_mhz=sm_clock,
                            clock_memory_mhz=mem_clock,
                            vendor=GPUPlatform.NVIDIA.value,
                        )
                    )
                return metrics
            finally:
                nvml.nvmlShutdown()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# AMD ROCm
# ---------------------------------------------------------------------------


def _iter_rocm_cards(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cards: list[tuple[str, dict[str, Any]]] = []
    skip = {"system", "timestamp", "error", "success"}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        lowered = str(key).lower()
        if lowered in skip:
            continue
        if lowered.startswith(("card", "gpu")) or key.isdigit():
            cards.append((str(key), value))
    if cards:
        return cards
    for key in ("gpus", "devices", "gpu_list"):
        items = payload.get(key)
        if isinstance(items, list):
            for index, item in enumerate(items):
                if isinstance(item, dict):
                    cards.append((str(index), item))
    return cards


def _extract_rocm_driver(payload: Mapping[str, Any], cards: Sequence[tuple[str, dict[str, Any]]]) -> str | None:
    system = payload.get("system")
    if isinstance(system, Mapping):
        for key, value in system.items():
            if "driver" in str(key).lower() and value:
                return str(value)
    for _key, card in cards:
        flat = _flatten(card)
        raw = _first_matching(flat, "driver")
        if raw:
            return str(raw)
    return None


def parse_rocm_smi_json(text: str) -> tuple[list[GPUMetrics], str | None]:
    """Parse ``rocm-smi --json`` into the shared GPUMetrics schema."""

    payload = _loads_json(text)
    if not isinstance(payload, dict):
        return [], None
    cards = _iter_rocm_cards(payload)
    driver = _extract_rocm_driver(payload, cards)
    metrics: list[GPUMetrics] = []
    for index, (card_key, card) in enumerate(cards):
        flat = _flatten(card)
        name = (
            _first_matching(flat, "card series")
            or _first_matching(flat, "device name")
            or _first_matching(flat, "card sku")
            or _first_matching(flat, "market name")
            or _first_matching(flat, "gpu name")
            or card_key
        )
        temp = _coerce_int(
            _first_matching(flat, "junction")
            or _first_matching(flat, "hotspot")
            or _first_matching(flat, "sensor edge")
            or _first_matching(flat, "temperature")
        )
        power = _coerce_float(
            _first_matching(flat, "average graphics package power")
            or _first_matching(flat, "socket graphics package power")
            or _first_matching(flat, "package power")
            or _first_matching(flat, "power", "w")
        )
        util = _coerce_float(
            _first_matching(flat, "gpu use")
            or _first_matching(flat, "gpu busy")
            or _first_matching(flat, "gfx activity")
            or _first_matching(flat, "gpu", "%")
        )
        vram_total_key, vram_total_raw = _first_matching_with_key(
            flat, "vram", "total", exclude=("used",)
        )
        used_key, vram_used_raw = _first_matching_with_key(flat, "vram", "used")
        mem_total = _bytes_or_mb_to_mb(
            _coerce_float(vram_total_raw),
            treat_as_bytes=_key_looks_like_bytes(vram_total_key),
        )
        mem_used = _bytes_or_mb_to_mb(
            _coerce_float(vram_used_raw),
            treat_as_bytes=_key_looks_like_bytes(used_key),
        )
        gfx_clock = _coerce_int(
            _first_matching(flat, "sclk clock")
            or _first_matching(flat, "gfx", "clock")
            or _first_matching(flat, "sclk")
        )
        mem_clock = _coerce_int(
            _first_matching(flat, "mclk clock")
            or _first_matching(flat, "memory clock")
            or _first_matching(flat, "mclk")
        )
        metrics.append(
            GPUMetrics(
                index=index,
                name=str(name),
                utilization_percent=util,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                temperature_c=temp,
                power_draw_w=power,
                clock_sm_mhz=gfx_clock,
                clock_memory_mhz=mem_clock,
                vendor=GPUPlatform.AMD.value,
            )
        )
    return metrics, driver


class AMDGPUTelemetryCollector:
    """AMD GPU collector via ``amdsmi`` or the ``rocm-smi`` CLI."""

    vendor = GPUPlatform.AMD

    def __init__(
        self,
        runner: CommandRunner | None = None,
        amdsmi: Any | None = None,
    ) -> None:
        self._run = runner if runner is not None else _run_command
        self._amdsmi = amdsmi
        self._driver_version: str | None = None

    def _load_amdsmi(self) -> Any | None:
        if self._amdsmi is not None:
            return self._amdsmi
        try:
            import amdsmi
        except Exception:
            return None
        return amdsmi

    def available(self) -> bool:
        amdsmi = self._load_amdsmi()
        if amdsmi is not None:
            try:
                amdsmi.amdsmi_init()
                try:
                    handles = amdsmi.amdsmi_get_processor_handles()
                    if handles:
                        return True
                finally:
                    amdsmi.amdsmi_shut_down()
            except Exception:
                pass
        return shutil.which("rocm-smi") is not None

    def collect_info(self) -> tuple[int | None, list[str] | None, str | None]:
        metrics = self.collect_metrics()
        if not metrics:
            return None, None, None
        return len(metrics), [item.name for item in metrics], self._driver_version

    def collect_metrics(self) -> list[GPUMetrics] | None:
        metrics = self._collect_amdsmi()
        if metrics:
            return metrics
        return self._collect_rocm_smi()

    def _collect_rocm_smi(self) -> list[GPUMetrics] | None:
        output = self._run(ROCM_SMI_FULL_COMMAND, timeout=8.0)
        if output is None:
            output = self._run(["rocm-smi", "--json"], timeout=8.0)
        if output is None:
            return None
        metrics, driver = parse_rocm_smi_json(output)
        self._driver_version = driver
        return metrics or None

    def _collect_amdsmi(self) -> list[GPUMetrics] | None:
        amdsmi = self._load_amdsmi()
        if amdsmi is None:
            return None
        try:
            amdsmi.amdsmi_init()
        except Exception:
            return None
        try:
            handles = amdsmi.amdsmi_get_processor_handles()
            if not handles:
                return None
            metrics: list[GPUMetrics] = []
            for index, handle in enumerate(handles):
                metrics.append(self._metrics_from_amdsmi(amdsmi, index, handle))
            driver = _lookup(
                _safe_call(lambda: amdsmi.amdsmi_get_gpu_driver_info(handles[0])),
                "driver_version",
                "driver",
                "version",
            )
            if driver:
                self._driver_version = str(driver)
            return metrics
        except Exception:
            return None
        finally:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass

    def _metrics_from_amdsmi(self, amdsmi: Any, index: int, handle: Any) -> GPUMetrics:
        name = _amdsmi_name(amdsmi, handle)
        util = _coerce_float(
            _lookup(
                _safe_call(lambda: amdsmi.amdsmi_get_gpu_activity(handle)),
                "gfx_activity",
                "gfx",
                "gpu_activity",
                "average_gfx_activity",
            )
        )
        vram = _safe_call(lambda: amdsmi.amdsmi_get_gpu_vram_usage(handle))
        mem_used = _bytes_or_mb_to_mb(
            _coerce_float(_lookup(vram, "vram_used", "used", "vram_used_mb")),
            treat_as_bytes=False,
        )
        mem_total = _bytes_or_mb_to_mb(
            _coerce_float(_lookup(vram, "vram_total", "total", "vram_total_mb")),
            treat_as_bytes=False,
        )
        temp = _amdsmi_temperature(amdsmi, handle)
        power_info = _safe_call(lambda: amdsmi.amdsmi_get_power_info(handle))
        power = _coerce_float(
            _lookup(
                power_info,
                "current_socket_power",
                "average_socket_power",
                "socket_power",
                "power",
                "current_power",
            )
        )
        gfx_clock = _amdsmi_clock(amdsmi, handle, ("GFX", "SYS", "CLK"))
        mem_clock = _amdsmi_clock(amdsmi, handle, ("MEM", "DF"))
        return GPUMetrics(
            index=index,
            name=name,
            utilization_percent=util,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
            temperature_c=temp,
            power_draw_w=power,
            clock_sm_mhz=gfx_clock,
            clock_memory_mhz=mem_clock,
            vendor=GPUPlatform.AMD.value,
        )


def _safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _amdsmi_name(amdsmi: Any, handle: Any) -> str:
    getters = (
        "amdsmi_get_gpu_asic_info",
        "amdsmi_get_gpu_board_info",
        "amdsmi_get_gpu_device_info",
    )
    for getter_name in getters:
        getter = getattr(amdsmi, getter_name, None)
        if getter is None:
            continue
        info = _safe_call(lambda g=getter: g(handle))
        name = _lookup(info, "market_name", "device_name", "product_name", "name", "vendor_id")
        if name:
            return str(name)
    return "AMD GPU"


def _amdsmi_temperature(amdsmi: Any, handle: Any) -> int | None:
    type_enum = getattr(amdsmi, "AmdSmiTemperatureType", None)
    metric_enum = getattr(amdsmi, "AmdSmiTemperatureMetric", None)
    kinds: list[Any] = []
    if type_enum is not None:
        for attr in ("JUNCTION", "HOTSPOT", "EDGE", "GPU"):
            value = getattr(type_enum, attr, None)
            if value is not None:
                kinds.append(value)
    kinds.extend((1, 0))
    metric = getattr(metric_enum, "CURRENT", 0) if metric_enum is not None else 0
    getter = getattr(amdsmi, "amdsmi_get_temp_metric", None)
    if getter is None:
        return None
    for kind in kinds:
        number = _coerce_int(_safe_call(lambda k=kind: getter(handle, k, metric)))
        if number is not None:
            return number
    return None


def _amdsmi_clock(amdsmi: Any, handle: Any, attrs: tuple[str, ...]) -> int | None:
    clk_type = getattr(amdsmi, "AmdSmiClkType", None)
    getter = getattr(amdsmi, "amdsmi_get_clock_info", None)
    if getter is None:
        return None
    kinds: list[Any] = []
    if clk_type is not None:
        for attr in attrs:
            value = getattr(clk_type, attr, None)
            if value is not None:
                kinds.append(value)
    for kind in kinds:
        info = _safe_call(lambda k=kind: getter(handle, k))
        number = _coerce_int(_lookup(info, "clk", "cur_clk", "current_clk", "clock", "freq"))
        if number is not None:
            return number
    return None


# ---------------------------------------------------------------------------
# Apple Metal
# ---------------------------------------------------------------------------


def parse_powermetrics_text(text: str) -> dict[str, float | None]:
    """Extract GPU utilization, power, clock, and bandwidth from powermetrics."""

    util = _match_number(text, r"GPU HW active residency:\s*([\d.]+)\s*%")
    clock = _match_number(
        text,
        r"GPU HW active frequency:\s*([\d.]+)\s*MHz",
    ) or _match_number(text, r"GPU HW requested frequency:\s*([\d.]+)\s*MHz")
    power_mw = _match_number(text, r"GPU Power:\s*([\d.]+)\s*mW")
    if power_mw is not None:
        power_w = power_mw / 1000.0
    else:
        power_w = _match_number(text, r"GPU Power:\s*([\d.]+)\s*W")
    bandwidth = _match_number(text, r"GPU Memory Bandwidth:\s*([\d.]+)\s*GB/s")
    return {
        "utilization_percent": util,
        "clock_sm_mhz": clock,
        "power_draw_w": power_w,
        "memory_bandwidth_gbps": bandwidth,
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
    except Exception:
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
    util = _coerce_float(
        stats.get("Device Utilization %")
        or stats.get("Renderer Utilization %")
        or stats.get("gpu-core-utilization")
    )
    used_bytes = _coerce_float(
        stats.get("In use system memory")
        or stats.get("In use Tiled memory")
        or stats.get("vramUsedBytes")
        or stats.get("Alloc device memory")
    )
    total_bytes = _coerce_float(
        stats.get("Alloc system memory")
        or stats.get("Alloc Tiled memory")
        or stats.get("vramTotalBytes")
    )
    name = (
        entry.get("model")
        or entry.get("IOAccelClass")
        or entry.get("CFBundleIdentifier")
        or "Apple GPU"
    )
    return {
        "name": str(name),
        "utilization_percent": util,
        "memory_used_mb": _bytes_or_mb_to_mb(used_bytes, treat_as_bytes=True) if used_bytes is not None else None,
        "memory_total_mb": _bytes_or_mb_to_mb(total_bytes, treat_as_bytes=True) if total_bytes is not None else None,
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
        return system == "Darwin" and machine.lower() in {"arm64", "arm64e"}

    def collect_info(self) -> tuple[int | None, list[str] | None, str | None]:
        metrics = self.collect_metrics()
        if not metrics:
            return None, None, None
        return len(metrics), [item.name for item in metrics], None

    def collect_metrics(self) -> list[GPUMetrics] | None:
        name = self._gpu_name()
        ioreg_rows = self._from_ioreg()
        power_stats = self._from_powermetrics()
        mem_used, mem_total = (None, None)
        if self.available():
            mem_used, mem_total = self._unified_memory_fallback()

        if ioreg_rows:
            metrics: list[GPUMetrics] = []
            for index, row in enumerate(ioreg_rows):
                metrics.append(
                    GPUMetrics(
                        index=index,
                        name=name or str(row["name"]),
                        utilization_percent=cast(float | None, row["utilization_percent"]),
                        memory_used_mb=cast(int | None, row["memory_used_mb"]) or mem_used,
                        memory_total_mb=cast(int | None, row["memory_total_mb"]) or mem_total,
                        temperature_c=None,
                        power_draw_w=power_stats.get("power_draw_w"),
                        clock_sm_mhz=_coerce_int(power_stats.get("clock_sm_mhz")),
                        clock_memory_mhz=None,
                        vendor=GPUPlatform.APPLE.value,
                        memory_bandwidth_gbps=power_stats.get("memory_bandwidth_gbps"),
                    )
                )
            return metrics

        if name is None and not any(power_stats.values()) and mem_used is None:
            return None

        util = power_stats.get("utilization_percent")
        return [
            GPUMetrics(
                index=0,
                name=name or "Apple GPU",
                utilization_percent=util,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                temperature_c=None,
                power_draw_w=power_stats.get("power_draw_w"),
                clock_sm_mhz=_coerce_int(power_stats.get("clock_sm_mhz")),
                clock_memory_mhz=None,
                vendor=GPUPlatform.APPLE.value,
                memory_bandwidth_gbps=power_stats.get("memory_bandwidth_gbps"),
            )
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
            return {
                "utilization_percent": None,
                "clock_sm_mhz": None,
                "power_draw_w": None,
                "memory_bandwidth_gbps": None,
            }
        return parse_powermetrics_text(output)

    def _unified_memory_fallback(self) -> tuple[int | None, int | None]:
        try:
            import psutil

            mem = psutil.virtual_memory()
            used = int((mem.total - mem.available) // (1024 * 1024))
            total = int(mem.total // (1024 * 1024))
            return used, total
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


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
        return collector.collect_metrics()
    except Exception as exc:
        warnings.warn(
            f"{platform.value} GPU telemetry collection failed: {exc}",
            GPUTelemetryUnavailableWarning,
            stacklevel=2,
        )
        return None


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
