from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.corinth_canal import CorinthCanalRun, load_corinth_canal, try_load_corinth_canal
from benchmarks.jsonio import write_json
from benchmarks.metrics import SaaqMetricOverlay


@dataclass(frozen=True)
class SystemSnapshot:
    """Cross-platform hardware snapshot collected at benchmark start."""

    cpu_count_logical: int | None
    cpu_count_physical: int | None
    memory_total_gb: float | None
    memory_available_gb: float | None
    gpu_count: int | None
    gpu_names: list[str] | None
    gpu_driver_version: str | None
    cuda_version: str | None
    platform: str
    python_version: str


@dataclass(frozen=True)
class GPUMetrics:
    """GPU telemetry when pynvml / nvidia-ml-py is available."""

    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: int | None
    memory_total_mb: int | None
    temperature_c: int | None
    power_draw_w: float | None
    clock_sm_mhz: int | None
    clock_memory_mhz: int | None


@dataclass(frozen=True)
class RoutingMetrics:
    """Neuromorphic / SAAQ routing metrics."""

    routing_entropy: float | None = None
    spike_density: float | None = None
    latent_stability: float | None = None
    dv_dt_reductions: float | None = None
    event_rate: float | None = None
    firing_rate: float | None = None
    membrane_pressure: float | None = None
    saaq_delta_q: float | None = None
    saaq_delta_q_last: float | None = None
    saaq_delta_q_legacy: float | None = None
    saaq_delta_q_v15: float | None = None


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Complete telemetry collected for a benchmark run or dataset."""

    system: SystemSnapshot
    gpu_metrics: list[GPUMetrics] | None = None
    routing: RoutingMetrics | None = None
    kernel_occupancy: float | None = None
    vram_bandwidth_gbps: float | None = None
    notes: str | None = None
    saaq_rule: str | None = None
    saaq_model_family: str | None = None
    saaq_delta_q_trajectory: tuple[float, ...] | None = None
    saaq_delta_q_legacy_trajectory: tuple[float, ...] | None = None
    saaq_delta_q_v15_trajectory: tuple[float, ...] | None = None


# ---------------------------------------------------------------------------
# System collectors (no CUDA kernels)
# ---------------------------------------------------------------------------

def collect_system_snapshot() -> SystemSnapshot:
    cpu_count_logical = os.cpu_count()
    cpu_count_physical = None
    memory_total_gb = None
    memory_available_gb = None

    try:
        import psutil

        cpu_count_physical = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        memory_total_gb = mem.total / (1024**3)
        memory_available_gb = mem.available / (1024**3)
    except ImportError:
        pass

    gpu_count, gpu_names, gpu_driver = _collect_gpu_info_nvidia()
    cuda_version = _collect_cuda_version()

    return SystemSnapshot(
        cpu_count_logical=cpu_count_logical,
        cpu_count_physical=cpu_count_physical,
        memory_total_gb=memory_total_gb,
        memory_available_gb=memory_available_gb,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        gpu_driver_version=gpu_driver,
        cuda_version=cuda_version,
        platform=platform.platform(),
        python_version=platform.python_version(),
    )


def _collect_cuda_version() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        marker = "CUDA Version:"
        if marker in output:
            tail = output.split(marker, maxsplit=1)[1].strip()
            parts = tail.split()
            if parts:
                return parts[0]
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            ["nvcc", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        marker = "release "
        if marker in output:
            tail = output.split(marker, maxsplit=1)[1]
            parts = tail.split(",", maxsplit=1)
            if parts:
                return parts[0].strip()
    except Exception:
        pass

    return None


def _collect_gpu_info_nvidia() -> tuple[int | None, list[str] | None, str | None]:
    """Use pynvml if available; otherwise return None. No CUDA kernels."""
    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
            nvmlSystemGetDriverVersion,
        )

        nvmlInit()
        try:
            count = nvmlDeviceGetCount()
            names = []
            for i in range(count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name_bytes = nvmlDeviceGetName(handle)
                names.append(name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes))
            driver = nvmlSystemGetDriverVersion()
            driver_str = driver.decode("utf-8") if isinstance(driver, bytes) else str(driver)
            return count, names, driver_str
        finally:
            nvmlShutdown()
    except Exception:
        return None, None, None


def collect_gpu_metrics() -> list[GPUMetrics] | None:
    """Per-GPU telemetry when pynvml is available."""
    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetClockInfo,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlDeviceGetPowerUsage,
            nvmlDeviceGetTemperature,
            nvmlDeviceGetUtilizationRates,
            nvmlInit,
            nvmlShutdown,
            NVML_CLOCK_SM,
            NVML_CLOCK_MEM,
            NVML_TEMPERATURE_GPU,
        )

        nvmlInit()
        try:
            metrics: list[GPUMetrics] = []
            count = nvmlDeviceGetCount()
            for i in range(count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name_bytes = nvmlDeviceGetName(handle)
                name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)

                util = None
                try:
                    util = nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    pass

                mem_used = None
                mem_total = None
                try:
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem.used // (1024 * 1024)
                    mem_total = mem.total // (1024 * 1024)
                except Exception:
                    pass

                temp = None
                try:
                    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                except Exception:
                    pass

                power = None
                try:
                    power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass

                clock_sm = None
                try:
                    clock_sm = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                except Exception:
                    pass

                clock_mem = None
                try:
                    clock_mem = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
                except Exception:
                    pass

                metrics.append(
                    GPUMetrics(
                        index=i,
                        name=name,
                        utilization_percent=util,
                        memory_used_mb=mem_used,
                        memory_total_mb=mem_total,
                        temperature_c=temp,
                        power_draw_w=power,
                        clock_sm_mhz=clock_sm,
                        clock_memory_mhz=clock_mem,
                    )
                )
            return metrics
        finally:
            nvmlShutdown()
    except Exception:
        return None


def collect_telemetry_snapshot() -> TelemetrySnapshot:
    system = collect_system_snapshot()
    gpu_metrics = collect_gpu_metrics()
    return TelemetrySnapshot(system=system, gpu_metrics=gpu_metrics)


# ---------------------------------------------------------------------------
# Compatibility hooks for upstream telemetry artifacts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorinthCanalArtifact:
    """Compatibility hook for corinth-canal SAAQ / telemetry output files."""

    artifact_version: str
    experiment_id: str
    routing_entropy: float | None = None
    spike_density: float | None = None
    event_rate: float | None = None
    latent_stability: float | None = None
    dv_dt_reductions: float | None = None
    raw_path: str | None = None
    firing_rate: float | None = None
    membrane_pressure: float | None = None
    membrane_dv_dt: float | None = None
    saaq_delta_q: float | None = None
    saaq_delta_q_last: float | None = None
    saaq_delta_q_legacy: float | None = None
    saaq_delta_q_v15: float | None = None
    saaq_delta_q_trajectory: tuple[float, ...] = ()
    saaq_delta_q_legacy_trajectory: tuple[float, ...] = ()
    saaq_delta_q_v15_trajectory: tuple[float, ...] = ()
    saaq_rule: str | None = None
    saaq_primary_rule: str | None = None
    model_family: str | None = None
    model_slug: str | None = None
    ticks_completed: int | None = None
    latent_rows: int | None = None
    skipped: tuple[str, ...] = ()

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "CorinthCanalArtifact":
        return CorinthCanalArtifact(
            artifact_version=raw.get("artifact_version", "unknown"),
            experiment_id=raw.get("experiment_id", ""),
            routing_entropy=raw.get("routing_entropy"),
            spike_density=raw.get("spike_density"),
            event_rate=raw.get("event_rate"),
            latent_stability=raw.get("latent_stability"),
            dv_dt_reductions=raw.get("dv_dt_reductions"),
            raw_path=raw.get("raw_path"),
            firing_rate=raw.get("firing_rate"),
            membrane_pressure=raw.get("membrane_pressure"),
            membrane_dv_dt=raw.get("membrane_dv_dt"),
            saaq_delta_q=raw.get("saaq_delta_q", raw.get("saaq_delta_q_mean")),
            saaq_delta_q_last=raw.get("saaq_delta_q_last"),
            saaq_delta_q_legacy=raw.get("saaq_delta_q_legacy"),
            saaq_delta_q_v15=raw.get("saaq_delta_q_v15"),
            saaq_delta_q_trajectory=_tuple_floats(
                raw.get("delta_q_trajectory") or raw.get("saaq_delta_q_trajectory")
            ),
            saaq_delta_q_legacy_trajectory=_tuple_floats(raw.get("delta_q_legacy_trajectory")),
            saaq_delta_q_v15_trajectory=_tuple_floats(raw.get("delta_q_v15_trajectory")),
            saaq_rule=raw.get("saaq_rule"),
            saaq_primary_rule=raw.get("saaq_primary_rule"),
            model_family=raw.get("model_family"),
            model_slug=raw.get("model_slug"),
            ticks_completed=raw.get("ticks_completed"),
            latent_rows=raw.get("latent_rows"),
        )

    @staticmethod
    def from_run(run: CorinthCanalRun) -> "CorinthCanalArtifact":
        return CorinthCanalArtifact(
            artifact_version=run.artifact_version,
            experiment_id=run.experiment_id,
            routing_entropy=run.routing_entropy,
            spike_density=run.spike_density,
            event_rate=run.event_rate,
            latent_stability=run.latent_stability,
            dv_dt_reductions=run.dv_dt_reductions,
            raw_path=run.raw_path,
            firing_rate=run.firing_rate,
            membrane_pressure=run.membrane_pressure,
            membrane_dv_dt=run.membrane_dv_dt,
            saaq_delta_q=run.saaq_delta_q,
            saaq_delta_q_last=run.saaq_delta_q_last,
            saaq_delta_q_legacy=run.saaq_delta_q_legacy,
            saaq_delta_q_v15=run.saaq_delta_q_v15,
            saaq_delta_q_trajectory=run.saaq_delta_q_trajectory,
            saaq_delta_q_legacy_trajectory=run.saaq_delta_q_legacy_trajectory,
            saaq_delta_q_v15_trajectory=run.saaq_delta_q_v15_trajectory,
            saaq_rule=run.saaq_rule,
            saaq_primary_rule=run.saaq_primary_rule,
            model_family=run.model_family,
            model_slug=run.model_slug,
            ticks_completed=run.ticks_completed,
            latent_rows=run.latent_rows,
            skipped=run.skipped,
        )

    @staticmethod
    def from_file(path: Path) -> "CorinthCanalArtifact":
        return CorinthCanalArtifact.from_path(path)

    @staticmethod
    def from_path(path: Path) -> "CorinthCanalArtifact":
        return CorinthCanalArtifact.from_run(load_corinth_canal(path))

    @staticmethod
    def try_from_path(path: Path) -> "CorinthCanalArtifact | None":
        run = try_load_corinth_canal(path)
        if run is None:
            return None
        return CorinthCanalArtifact.from_run(run)

    def to_saaq_overlay(self) -> SaaqMetricOverlay:
        return SaaqMetricOverlay(
            firing_rate=self.firing_rate,
            membrane_pressure=self.membrane_pressure,
            saaq_delta_q=self.saaq_delta_q,
            saaq_delta_q_last=self.saaq_delta_q_last,
            saaq_delta_q_legacy=self.saaq_delta_q_legacy,
            saaq_delta_q_v15=self.saaq_delta_q_v15,
            saaq_rule=self.saaq_rule,
            spike_density=self.spike_density,
        )


@dataclass(frozen=True)
class MyelinAcceleratorArtifact:
    """Compatibility hook for myelin-accelerator benchmark artifacts."""

    artifact_version: str
    benchmark_id: str
    kernel_occupancy: float | None = None
    vram_bandwidth_gbps: float | None = None
    gpu_utilization_percent: float | None = None
    latency_ms: float | None = None
    throughput_tops: float | None = None
    raw_path: str | None = None

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "MyelinAcceleratorArtifact":
        return MyelinAcceleratorArtifact(
            artifact_version=raw.get("artifact_version", "unknown"),
            benchmark_id=raw.get("benchmark_id", ""),
            kernel_occupancy=raw.get("kernel_occupancy"),
            vram_bandwidth_gbps=raw.get("vram_bandwidth_gbps"),
            gpu_utilization_percent=raw.get("gpu_utilization_percent"),
            latency_ms=raw.get("latency_ms"),
            throughput_tops=raw.get("throughput_tops"),
            raw_path=raw.get("raw_path"),
        )

    @staticmethod
    def from_file(path: Path) -> "MyelinAcceleratorArtifact":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return MyelinAcceleratorArtifact.from_dict(raw)


# ---------------------------------------------------------------------------
# Merge upstream artifacts into a TelemetrySnapshot
# ---------------------------------------------------------------------------

def merge_upstream_artifacts(
    telemetry: TelemetrySnapshot,
    corinth: CorinthCanalArtifact | None = None,
    myelin: MyelinAcceleratorArtifact | None = None,
) -> TelemetrySnapshot:
    routing = telemetry.routing
    saaq_rule = telemetry.saaq_rule
    saaq_model_family = telemetry.saaq_model_family
    saaq_traj = telemetry.saaq_delta_q_trajectory
    saaq_legacy_traj = telemetry.saaq_delta_q_legacy_trajectory
    saaq_v15_traj = telemetry.saaq_delta_q_v15_trajectory
    notes = telemetry.notes
    if corinth:
        base = routing or RoutingMetrics()
        routing = RoutingMetrics(
            routing_entropy=_coalesce(corinth.routing_entropy, base.routing_entropy),
            spike_density=_coalesce(corinth.spike_density, base.spike_density),
            latent_stability=_coalesce(corinth.latent_stability, base.latent_stability),
            dv_dt_reductions=_coalesce(corinth.dv_dt_reductions, base.dv_dt_reductions),
            event_rate=_coalesce(corinth.event_rate, base.event_rate),
            firing_rate=_coalesce(corinth.firing_rate, base.firing_rate),
            membrane_pressure=_coalesce(corinth.membrane_pressure, base.membrane_pressure),
            saaq_delta_q=_coalesce(corinth.saaq_delta_q, base.saaq_delta_q),
            saaq_delta_q_last=_coalesce(corinth.saaq_delta_q_last, base.saaq_delta_q_last),
            saaq_delta_q_legacy=_coalesce(corinth.saaq_delta_q_legacy, base.saaq_delta_q_legacy),
            saaq_delta_q_v15=_coalesce(corinth.saaq_delta_q_v15, base.saaq_delta_q_v15),
        )
        saaq_rule = _coalesce(corinth.saaq_rule, saaq_rule)
        saaq_model_family = _coalesce(corinth.model_family, saaq_model_family)
        if corinth.saaq_delta_q_trajectory:
            saaq_traj = corinth.saaq_delta_q_trajectory
        if corinth.saaq_delta_q_legacy_trajectory:
            saaq_legacy_traj = corinth.saaq_delta_q_legacy_trajectory
        if corinth.saaq_delta_q_v15_trajectory:
            saaq_v15_traj = corinth.saaq_delta_q_v15_trajectory
        if corinth.skipped:
            skipped_note = "skipped corinth-canal artifacts: " + ", ".join(corinth.skipped)
            notes = f"{notes}; {skipped_note}" if notes else skipped_note

    kernel_occupancy = telemetry.kernel_occupancy
    vram_bw = telemetry.vram_bandwidth_gbps
    if myelin:
        kernel_occupancy = myelin.kernel_occupancy if myelin.kernel_occupancy is not None else kernel_occupancy
        vram_bw = myelin.vram_bandwidth_gbps if myelin.vram_bandwidth_gbps is not None else vram_bw

    return TelemetrySnapshot(
        system=telemetry.system,
        gpu_metrics=telemetry.gpu_metrics,
        routing=routing,
        kernel_occupancy=kernel_occupancy,
        vram_bandwidth_gbps=vram_bw,
        notes=notes,
        saaq_rule=saaq_rule,
        saaq_model_family=saaq_model_family,
        saaq_delta_q_trajectory=saaq_traj,
        saaq_delta_q_legacy_trajectory=saaq_legacy_traj,
        saaq_delta_q_v15_trajectory=saaq_v15_traj,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def telemetry_to_dict(telemetry: TelemetrySnapshot) -> dict[str, Any]:
    """Convert a TelemetrySnapshot to a plain dict for JSON/CSV embedding."""
    d: dict[str, Any] = {}
    sys_dict = asdict(telemetry.system)
    for k, v in sys_dict.items():
        d[f"sys_{k}"] = v
    if telemetry.gpu_metrics:
        d["gpu_metrics"] = [asdict(g) for g in telemetry.gpu_metrics]
    routing = telemetry.routing
    d["routing_entropy"] = routing.routing_entropy if routing else None
    d["spike_density"] = routing.spike_density if routing else None
    d["latent_stability"] = routing.latent_stability if routing else None
    d["dv_dt_reductions"] = routing.dv_dt_reductions if routing else None
    d["event_rate"] = routing.event_rate if routing else None
    d["firing_rate"] = routing.firing_rate if routing else None
    d["membrane_pressure"] = routing.membrane_pressure if routing else None
    d["saaq_delta_q"] = routing.saaq_delta_q if routing else None
    d["saaq_delta_q_last"] = routing.saaq_delta_q_last if routing else None
    d["saaq_delta_q_legacy"] = routing.saaq_delta_q_legacy if routing else None
    d["saaq_delta_q_v15"] = routing.saaq_delta_q_v15 if routing else None
    d["saaq_rule"] = telemetry.saaq_rule
    d["saaq_model_family"] = telemetry.saaq_model_family
    d["saaq_delta_q_trajectory"] = (
        list(telemetry.saaq_delta_q_trajectory) if telemetry.saaq_delta_q_trajectory else None
    )
    d["saaq_delta_q_legacy_trajectory"] = (
        list(telemetry.saaq_delta_q_legacy_trajectory)
        if telemetry.saaq_delta_q_legacy_trajectory
        else None
    )
    d["saaq_delta_q_v15_trajectory"] = (
        list(telemetry.saaq_delta_q_v15_trajectory) if telemetry.saaq_delta_q_v15_trajectory else None
    )
    d["kernel_occupancy"] = telemetry.kernel_occupancy
    d["vram_bandwidth_gbps"] = telemetry.vram_bandwidth_gbps
    d["telemetry_notes"] = telemetry.notes
    return d


def write_telemetry_json(path: Path, telemetry: TelemetrySnapshot) -> None:
    """Write the standalone telemetry artifact as standard JSON.

    A benchmark run writes this alongside its main report, and hardware probes
    can legitimately yield a non-finite reading, so it goes through the same
    guard as every other report rather than calling json.dump directly.
    """
    write_json(path, telemetry_to_dict(telemetry))


def _coalesce(new: Any, old: Any) -> Any:
    return new if new is not None else old


def _tuple_floats(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)
