from __future__ import annotations

import json
import plistlib
from types import SimpleNamespace

import pytest

from benchmarks.gpu_telemetry import (
    AMDGPUTelemetryCollector,
    AppleMetalTelemetryCollector,
    GPUMetrics,
    GPUPlatform,
    GPUTelemetryUnavailableWarning,
    NVIDIAGPUTelemetryCollector,
    apple_snapshot_notes,
    collect_gpu_metrics,
    detect_gpu_platform,
    parse_powermetrics_text,
    parse_rocm_smi_json,
)


SAMPLE_ROCM_JSON = json.dumps(
    {
        "system": {"Driver version": "6.10.5"},
        "card0": {
            "Card series": "Instinct MI300X",
            "Card model": "0x74a1",
            "Card SKU": "MI300X",
            "Temperature (Sensor edge) (C)": "41.0",
            "Temperature (Sensor junction) (C)": "47.0",
            "Average Graphics Package Power (W)": "350.0",
            "GPU use (%)": "87",
            "VRAM Total Memory (B)": str(192 * 1024 * 1024 * 1024),
            "VRAM Total Used Memory (B)": str(48 * 1024 * 1024 * 1024),
            "sclk clock speed:": "1700Mhz",
            "mclk clock speed:": "900Mhz",
        },
    }
)

SAMPLE_ROCM_7900 = json.dumps(
    {
        "card0": {
            "Device Name": "AMD Radeon RX 7900 XTX",
            "Temperature (Sensor edge) (C)": "62",
            "Current Socket Graphics Package Power (W)": "285.5",
            "GPU use (%)": "94",
            "VRAM Total Memory (B)": str(24 * 1024 * 1024 * 1024),
            "VRAM Total Used Memory (B)": str(10 * 1024 * 1024 * 1024),
            "sclk clock speed:": "2500Mhz",
            "mclk clock speed:": "1250Mhz",
        }
    }
)

SAMPLE_POWERMETRICS = """
**** GPU usage ****
GPU HW active residency:  37.50% (1398 MHz: 37% )
GPU HW active frequency: 1398 MHz
GPU HW requested frequency: 1398 MHz
GPU Power: 12340 mW
GPU Memory Bandwidth: 245.0 GB/s
"""


def _ioreg_plist() -> str:
    payload = [
        {
            "model": "Apple M3 Max",
            "PerformanceStatistics": {
                "Device Utilization %": 42,
                "In use system memory": 12 * 1024 * 1024 * 1024,
                "Alloc system memory": 128 * 1024 * 1024 * 1024,
                "vramTotalBytes": 128 * 1024 * 1024 * 1024,
            },
        }
    ]
    return plistlib.dumps(payload).decode("utf-8")


def _profiler_json(model: str = "Apple M4 Ultra") -> str:
    return json.dumps({"SPDisplaysDataType": [{"sppci_model": model}]})


class _FakeNvml:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 0
    NVML_CLOCK_MEM = 1

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, handle: int) -> str:
        return "NVIDIA GeForce RTX 4090"

    def nvmlSystemGetDriverVersion(self) -> str:
        return "545.23"

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(gpu=85.0)

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(used=8 * 1024 * 1024 * 1024, total=24 * 1024 * 1024 * 1024)

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        return 65

    def nvmlDeviceGetPowerUsage(self, handle: int) -> int:
        return 250_000

    def nvmlDeviceGetClockInfo(self, handle: int, kind: int) -> int:
        return 2000 if kind == 0 else 10500


class _FakeAmdsmi:
    class AmdSmiTemperatureType:
        JUNCTION = 1
        EDGE = 0

    class AmdSmiTemperatureMetric:
        CURRENT = 0

    class AmdSmiClkType:
        GFX = 0
        MEM = 1

    def amdsmi_init(self) -> None:
        return None

    def amdsmi_shut_down(self) -> None:
        return None

    def amdsmi_get_processor_handles(self) -> list[int]:
        return [0]

    def amdsmi_get_gpu_asic_info(self, handle: int) -> dict[str, str]:
        return {"market_name": "AMD Instinct MI250X"}

    def amdsmi_get_gpu_activity(self, handle: int) -> dict[str, int]:
        return {"gfx_activity": 91}

    def amdsmi_get_gpu_vram_usage(self, handle: int) -> dict[str, int]:
        return {"vram_used": 32768, "vram_total": 131072}

    def amdsmi_get_temp_metric(self, handle: int, kind: int, metric: int) -> int:
        return 52

    def amdsmi_get_power_info(self, handle: int) -> dict[str, float]:
        return {"current_socket_power": 420.0}

    def amdsmi_get_clock_info(self, handle: int, kind: int) -> dict[str, int]:
        return {"clk": 1700 if kind == 0 else 1200}

    def amdsmi_get_gpu_driver_info(self, handle: int) -> dict[str, str]:
        return {"driver_version": "6.10.5"}


class _Probe:
    def __init__(self, vendor: GPUPlatform, present: bool) -> None:
        self.vendor = vendor
        self._present = present

    def available(self) -> bool:
        return self._present


def test_nvidia_collector_uses_pynvml_schema() -> None:
    metrics = NVIDIAGPUTelemetryCollector(nvml=_FakeNvml()).collect_metrics()
    assert metrics is not None
    assert len(metrics) == 1
    gpu = metrics[0]
    assert gpu.vendor == "nvidia"
    assert gpu.name == "NVIDIA GeForce RTX 4090"
    assert gpu.utilization_percent == 85.0
    assert gpu.memory_used_mb == 8192
    assert gpu.memory_total_mb == 24576
    assert gpu.temperature_c == 65
    assert gpu.power_draw_w == 250.0
    assert gpu.clock_sm_mhz == 2000
    assert gpu.clock_memory_mhz == 10500
    assert gpu.memory_bandwidth_gbps is None


def test_nvidia_collector_info() -> None:
    count, names, driver = NVIDIAGPUTelemetryCollector(nvml=_FakeNvml()).collect_info()
    assert count == 1
    assert names == ["NVIDIA GeForce RTX 4090"]
    assert driver == "545.23"


def test_parse_rocm_smi_mi300x() -> None:
    metrics, driver = parse_rocm_smi_json(SAMPLE_ROCM_JSON)
    assert driver == "6.10.5"
    assert len(metrics) == 1
    gpu = metrics[0]
    assert gpu.vendor == "amd"
    assert gpu.name == "Instinct MI300X"
    assert gpu.utilization_percent == 87.0
    assert gpu.temperature_c == 47
    assert gpu.power_draw_w == 350.0
    assert gpu.memory_total_mb == 192 * 1024
    assert gpu.memory_used_mb == 48 * 1024
    assert gpu.clock_sm_mhz == 1700
    assert gpu.clock_memory_mhz == 900


def test_parse_rocm_smi_rx_7900_xtx() -> None:
    metrics, driver = parse_rocm_smi_json(SAMPLE_ROCM_7900)
    assert driver is None
    gpu = metrics[0]
    assert gpu.name == "AMD Radeon RX 7900 XTX"
    assert gpu.power_draw_w == 285.5
    assert gpu.memory_total_mb == 24 * 1024
    assert gpu.memory_used_mb == 10 * 1024


def test_amd_collector_mocked_rocm_smi() -> None:
    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        if command and command[0] == "rocm-smi":
            return SAMPLE_ROCM_JSON
        return None

    collector = AMDGPUTelemetryCollector(runner=runner)
    metrics = collector.collect_metrics()
    assert metrics is not None
    assert metrics[0].name == "Instinct MI300X"
    count, names, driver = collector.collect_info()
    assert count == 1
    assert names == ["Instinct MI300X"]
    assert driver == "6.10.5"


def test_amd_collector_prefers_amdsmi() -> None:
    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        raise AssertionError("rocm-smi should not run when amdsmi succeeds")

    collector = AMDGPUTelemetryCollector(runner=runner, amdsmi=_FakeAmdsmi())
    metrics = collector.collect_metrics()
    assert metrics is not None
    gpu = metrics[0]
    assert gpu.name == "AMD Instinct MI250X"
    assert gpu.utilization_percent == 91.0
    assert gpu.memory_used_mb == 32768
    assert gpu.memory_total_mb == 131072
    assert gpu.temperature_c == 52
    assert gpu.power_draw_w == 420.0
    assert gpu.clock_sm_mhz == 1700
    assert gpu.clock_memory_mhz == 1200


def test_parse_powermetrics_text() -> None:
    parsed = parse_powermetrics_text(SAMPLE_POWERMETRICS)
    assert parsed["utilization_percent"] == 37.5
    assert parsed["clock_sm_mhz"] == 1398.0
    assert parsed["power_draw_w"] == pytest.approx(12.34)
    assert parsed["memory_bandwidth_gbps"] == 245.0


def test_apple_collector_mocked_ioreg_and_powermetrics() -> None:
    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        if command and command[0] == "ioreg":
            return _ioreg_plist()
        if command and command[0] == "powermetrics":
            return SAMPLE_POWERMETRICS
        if command and command[0] == "system_profiler":
            return _profiler_json("Apple M3 Max")
        return None

    collector = AppleMetalTelemetryCollector(runner=runner, system="Darwin", machine="arm64")
    metrics = collector.collect_metrics()
    assert metrics is not None
    gpu = metrics[0]
    assert gpu.vendor == "apple"
    assert gpu.name == "Apple M3 Max"
    assert gpu.utilization_percent == 42.0
    assert gpu.memory_used_mb == 12 * 1024
    assert gpu.memory_total_mb == 128 * 1024
    assert gpu.temperature_c is None
    assert gpu.power_draw_w == pytest.approx(12.34)
    assert gpu.clock_sm_mhz == 1398
    assert gpu.clock_memory_mhz is None
    assert gpu.memory_bandwidth_gbps == 245.0
    notes = apple_snapshot_notes(metrics)
    assert notes is not None
    assert "unified memory" in notes


def test_apple_collector_m4_ultra_name_from_profiler() -> None:
    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        if command and command[0] == "powermetrics":
            return SAMPLE_POWERMETRICS
        if command and command[0] == "system_profiler":
            return _profiler_json("Apple M4 Ultra")
        return None

    collector = AppleMetalTelemetryCollector(runner=runner, system="Darwin", machine="arm64")
    metrics = collector.collect_metrics()
    assert metrics is not None
    assert metrics[0].name == "Apple M4 Ultra"
    assert metrics[0].utilization_percent == 37.5


def test_apple_unavailable_on_linux() -> None:
    collector = AppleMetalTelemetryCollector(system="Linux", machine="x86_64")
    assert collector.available() is False


def test_nvidia_zero_devices_is_unavailable() -> None:
    class _EmptyNvml(_FakeNvml):
        def nvmlDeviceGetCount(self) -> int:
            return 0

    assert NVIDIAGPUTelemetryCollector(nvml=_EmptyNvml()).available() is False


def test_apple_preserves_idle_zero_utilization() -> None:
    payload = [
        {
            "model": "Apple M3 Max",
            "PerformanceStatistics": {
                "Device Utilization %": 0,
                "In use system memory": 0,
                "vramTotalBytes": 64 * 1024 * 1024 * 1024,
            },
        }
    ]

    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        if command and command[0] == "ioreg":
            return plistlib.dumps(payload).decode("utf-8")
        return None

    collector = AppleMetalTelemetryCollector(runner=runner, system="Darwin", machine="arm64")
    metrics = collector.collect_metrics()
    assert metrics is not None
    assert metrics[0].utilization_percent == 0.0
    assert metrics[0].memory_used_mb == 0
    assert metrics[0].memory_total_mb == 64 * 1024


def test_apple_does_not_treat_alloc_system_memory_as_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {
            "model": "Apple M3 Max",
            "PerformanceStatistics": {
                "Device Utilization %": 10,
                "In use system memory": 2 * 1024 * 1024 * 1024,
                "Alloc system memory": 8 * 1024 * 1024 * 1024,
            },
        }
    ]

    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        if command and command[0] == "ioreg":
            return plistlib.dumps(payload).decode("utf-8")
        return None

    class _Mem:
        total = 96 * 1024 * 1024 * 1024

    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "psutil", types.SimpleNamespace(virtual_memory=lambda: _Mem())
    )
    collector = AppleMetalTelemetryCollector(runner=runner, system="Darwin", machine="arm64")
    metrics = collector.collect_metrics()
    assert metrics is not None
    assert metrics[0].memory_used_mb == 2 * 1024
    assert metrics[0].memory_total_mb == 96 * 1024


def test_apple_available_under_rosetta() -> None:
    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        if command[:3] == ["sysctl", "-n", "sysctl.proc_translated"]:
            return "1\n"
        return None

    collector = AppleMetalTelemetryCollector(
        runner=runner, system="Darwin", machine="x86_64"
    )
    assert collector.available() is True


def test_collect_gpu_metrics_warns_when_collector_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.gpu_telemetry.detect_gpu_platform", lambda: GPUPlatform.AMD
    )
    monkeypatch.setattr(AMDGPUTelemetryCollector, "collect_metrics", lambda self: None)
    with pytest.warns(GPUTelemetryUnavailableWarning, match="returned no metrics"):
        assert collect_gpu_metrics() is None


def test_detect_prefers_apple_then_nvidia_then_amd() -> None:
    apple = _Probe(GPUPlatform.APPLE, True)
    nvidia = _Probe(GPUPlatform.NVIDIA, True)
    amd = _Probe(GPUPlatform.AMD, True)
    assert detect_gpu_platform((apple, nvidia, amd)) is GPUPlatform.APPLE
    assert detect_gpu_platform((nvidia, amd)) is GPUPlatform.NVIDIA
    assert detect_gpu_platform((amd,)) is GPUPlatform.AMD
    assert detect_gpu_platform(()) is GPUPlatform.NONE


def test_collect_gpu_metrics_dispatches_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [
        GPUMetrics(
            index=0,
            name="RTX 4090",
            utilization_percent=10.0,
            memory_used_mb=1,
            memory_total_mb=2,
            temperature_c=40,
            power_draw_w=100.0,
            clock_sm_mhz=1000,
            clock_memory_mhz=2000,
            vendor="nvidia",
        )
    ]
    monkeypatch.setattr(
        "benchmarks.gpu_telemetry.detect_gpu_platform", lambda: GPUPlatform.NVIDIA
    )
    monkeypatch.setattr(
        NVIDIAGPUTelemetryCollector,
        "collect_metrics",
        lambda self: expected,
    )
    assert collect_gpu_metrics() == expected


def test_collect_gpu_metrics_dispatches_amd(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [
        GPUMetrics(
            index=0,
            name="Instinct MI300X",
            utilization_percent=50.0,
            memory_used_mb=1,
            memory_total_mb=2,
            temperature_c=50,
            power_draw_w=200.0,
            clock_sm_mhz=1600,
            clock_memory_mhz=800,
            vendor="amd",
        )
    ]
    monkeypatch.setattr(
        "benchmarks.gpu_telemetry.detect_gpu_platform", lambda: GPUPlatform.AMD
    )
    monkeypatch.setattr(AMDGPUTelemetryCollector, "collect_metrics", lambda self: expected)
    assert collect_gpu_metrics() == expected


def test_collect_gpu_metrics_dispatches_apple(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [
        GPUMetrics(
            index=0,
            name="Apple M3 Max",
            utilization_percent=20.0,
            memory_used_mb=1,
            memory_total_mb=2,
            temperature_c=None,
            power_draw_w=8.0,
            clock_sm_mhz=1398,
            clock_memory_mhz=None,
            vendor="apple",
            memory_bandwidth_gbps=200.0,
        )
    ]
    monkeypatch.setattr(
        "benchmarks.gpu_telemetry.detect_gpu_platform", lambda: GPUPlatform.APPLE
    )
    monkeypatch.setattr(
        AppleMetalTelemetryCollector,
        "collect_metrics",
        lambda self: expected,
    )
    assert collect_gpu_metrics() == expected


def test_collect_gpu_metrics_warns_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmarks.gpu_telemetry.detect_gpu_platform", lambda: GPUPlatform.NONE
    )
    with pytest.warns(GPUTelemetryUnavailableWarning, match="No GPU telemetry backend"):
        assert collect_gpu_metrics() is None


def test_collectors_do_not_crash_when_tools_missing() -> None:
    nvidia = NVIDIAGPUTelemetryCollector().collect_metrics()
    amd = AMDGPUTelemetryCollector().collect_metrics()
    linux_apple = AppleMetalTelemetryCollector(system="Linux", machine="x86_64")
    assert nvidia is None or isinstance(nvidia, list)
    assert amd is None or isinstance(amd, list)
    assert linux_apple.collect_metrics() is None


def test_amd_rocm_smi_garbage_returns_none() -> None:
    def runner(command: list[str], *, timeout: float = 5.0) -> str | None:
        return "not json at all"

    assert AMDGPUTelemetryCollector(runner=runner).collect_metrics() is None


def test_run_command_resolves_executable_absolutely(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from benchmarks.gpu_telemetry_common import _run_command

    recorded: list[list[str]] = []
    probe = tmp_path / "rocm-smi"
    probe.write_text("#!/bin/sh\n")

    def fake_which(name: str) -> str | None:
        return str(probe) if name == "rocm-smi" else None

    def fake_check_output(argv: list[str], **kwargs: object) -> str:
        recorded.append(list(argv))
        return "{}"

    monkeypatch.setattr("benchmarks.gpu_telemetry_common.shutil.which", fake_which)
    monkeypatch.setattr("benchmarks.gpu_telemetry_common.subprocess.check_output", fake_check_output)
    assert _run_command(["rocm-smi", "--json"]) == "{}"
    assert recorded == [[str(probe), "--json"]]
    assert _run_command(["missing-tool"]) is None
