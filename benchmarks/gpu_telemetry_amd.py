"""AMD ROCm GPU telemetry collector."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from benchmarks.gpu_telemetry_common import (
    CommandRunner,
    GPUMetrics,
    GPUPlatform,
    _bytes_or_mb_to_mb,
    _coerce_float,
    _coerce_int,
    _first_matching,
    _first_matching_with_key,
    _flatten,
    _key_looks_like_bytes,
    _loads_json,
    _lookup,
    _run_command,
    ignore_probe_error,
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


def _safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        ignore_probe_error(exc)
        return None


def _is_rocm_card_entry(key: Any, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    lowered = str(key).lower()
    if lowered in {"system", "timestamp", "error", "success"}:
        return False
    return lowered.startswith(("card", "gpu")) or str(key).isdigit()


def _cards_from_named_lists(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cards: list[tuple[str, dict[str, Any]]] = []
    for key in ("gpus", "devices", "gpu_list"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if isinstance(item, dict):
                cards.append((str(index), item))
    return cards


def _iter_rocm_cards(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cards = [
        (str(key), value) for key, value in payload.items() if _is_rocm_card_entry(key, value)
    ]
    return cards or _cards_from_named_lists(payload)


def _extract_rocm_driver(
    payload: Mapping[str, Any], cards: Sequence[tuple[str, dict[str, Any]]]
) -> str | None:
    system = payload.get("system")
    if isinstance(system, Mapping):
        for key, value in system.items():
            if "driver" in str(key).lower() and value:
                return str(value)
    for _key, card in cards:
        raw = _first_matching(_flatten(card), "driver")
        if raw:
            return str(raw)
    return None


def _first_rocm(flat: Mapping[str, Any], *candidates: tuple[str, ...]) -> Any:
    for parts in candidates:
        value = _first_matching(flat, *parts)
        if value is not None:
            return value
    return None


def _rocm_card_name(flat: Mapping[str, Any], card_key: str) -> str:
    name = _first_rocm(
        flat,
        ("card series",),
        ("device name",),
        ("card sku",),
        ("market name",),
        ("gpu name",),
    )
    return str(name or card_key)


def _rocm_card_memory(flat: Mapping[str, Any]) -> tuple[int | None, int | None]:
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
    return mem_used, mem_total


def _metrics_from_rocm_card(index: int, card_key: str, card: Mapping[str, Any]) -> GPUMetrics:
    flat = _flatten(card)
    mem_used, mem_total = _rocm_card_memory(flat)
    return GPUMetrics(
        index=index,
        name=_rocm_card_name(flat, card_key),
        utilization_percent=_coerce_float(
            _first_rocm(flat, ("gpu use",), ("gpu busy",), ("gfx activity",), ("gpu", "%"))
        ),
        memory_used_mb=mem_used,
        memory_total_mb=mem_total,
        temperature_c=_coerce_int(
            _first_rocm(flat, ("junction",), ("hotspot",), ("sensor edge",), ("temperature",))
        ),
        power_draw_w=_coerce_float(
            _first_rocm(
                flat,
                ("average graphics package power",),
                ("socket graphics package power",),
                ("package power",),
                ("power", "w"),
            )
        ),
        clock_sm_mhz=_coerce_int(_first_rocm(flat, ("sclk clock",), ("gfx", "clock"), ("sclk",))),
        clock_memory_mhz=_coerce_int(
            _first_rocm(flat, ("mclk clock",), ("memory clock",), ("mclk",))
        ),
        vendor=GPUPlatform.AMD.value,
    )


def parse_rocm_smi_json(text: str) -> tuple[list[GPUMetrics], str | None]:
    """Parse ``rocm-smi --json`` into the shared GPUMetrics schema."""

    payload = _loads_json(text)
    if not isinstance(payload, dict):
        return [], None
    cards = _iter_rocm_cards(payload)
    metrics = [
        _metrics_from_rocm_card(index, card_key, card)
        for index, (card_key, card) in enumerate(cards)
    ]
    return metrics, _extract_rocm_driver(payload, cards)


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


def _amdsmi_driver_version(amdsmi: Any, handles: Sequence[Any]) -> str | None:
    driver = _lookup(
        _safe_call(lambda: amdsmi.amdsmi_get_gpu_driver_info(handles[0])),
        "driver_version",
        "driver",
        "version",
    )
    return str(driver) if driver else None


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
        except Exception as exc:
            ignore_probe_error(exc)
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
            except Exception as exc:
                ignore_probe_error(exc)
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
        except Exception as exc:
            ignore_probe_error(exc)
            return None
        try:
            return self._amdsmi_metrics(amdsmi)
        except Exception as exc:
            ignore_probe_error(exc)
            return None
        finally:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception as exc:
                ignore_probe_error(exc)

    def _amdsmi_metrics(self, amdsmi: Any) -> list[GPUMetrics] | None:
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return None
        metrics = [
            self._metrics_from_amdsmi(amdsmi, index, handle)
            for index, handle in enumerate(handles)
        ]
        driver = _amdsmi_driver_version(amdsmi, handles)
        if driver:
            self._driver_version = driver
        return metrics

    def _metrics_from_amdsmi(self, amdsmi: Any, index: int, handle: Any) -> GPUMetrics:
        vram = _safe_call(lambda: amdsmi.amdsmi_get_gpu_vram_usage(handle))
        power_info = _safe_call(lambda: amdsmi.amdsmi_get_power_info(handle))
        return GPUMetrics(
            index=index,
            name=_amdsmi_name(amdsmi, handle),
            utilization_percent=_coerce_float(
                _lookup(
                    _safe_call(lambda: amdsmi.amdsmi_get_gpu_activity(handle)),
                    "gfx_activity",
                    "gfx",
                    "gpu_activity",
                    "average_gfx_activity",
                )
            ),
            memory_used_mb=_bytes_or_mb_to_mb(
                _coerce_float(_lookup(vram, "vram_used", "used", "vram_used_mb")),
                treat_as_bytes=False,
            ),
            memory_total_mb=_bytes_or_mb_to_mb(
                _coerce_float(_lookup(vram, "vram_total", "total", "vram_total_mb")),
                treat_as_bytes=False,
            ),
            temperature_c=_amdsmi_temperature(amdsmi, handle),
            power_draw_w=_coerce_float(
                _lookup(
                    power_info,
                    "current_socket_power",
                    "average_socket_power",
                    "socket_power",
                    "power",
                    "current_power",
                )
            ),
            clock_sm_mhz=_amdsmi_clock(amdsmi, handle, ("GFX", "SYS", "CLK")),
            clock_memory_mhz=_amdsmi_clock(amdsmi, handle, ("MEM", "DF")),
            vendor=GPUPlatform.AMD.value,
        )
