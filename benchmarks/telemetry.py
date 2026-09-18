from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.gpu_telemetry import (
    GPUMetrics,
    GPUPlatform,
    apple_snapshot_notes,
    bandwidth_from_metrics,
    collect_gpu_info,
    collect_gpu_metrics,
    detect_gpu_platform,
)


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
class RoutingMetrics:
    """Neuromorphic / SAAQ routing metrics."""

    routing_entropy: float | None = None
    spike_density: float | None = None
    latent_stability: float | None = None
    dv_dt_reductions: float | None = None
    event_rate: float | None = None


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Complete telemetry collected for a benchmark run or dataset."""

    system: SystemSnapshot
    gpu_metrics: list[GPUMetrics] | None = None
    routing: RoutingMetrics | None = None
    kernel_occupancy: float | None = None
    vram_bandwidth_gbps: float | None = None
    notes: str | None = None


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

    detected = detect_gpu_platform()
    gpu_count, gpu_names, gpu_driver = collect_gpu_info()
    cuda_version = _collect_cuda_version() if detected is GPUPlatform.NVIDIA else None

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


def collect_telemetry_snapshot() -> TelemetrySnapshot:
    system = collect_system_snapshot()
    gpu_metrics = collect_gpu_metrics()
    return TelemetrySnapshot(
        system=system,
        gpu_metrics=gpu_metrics,
        vram_bandwidth_gbps=bandwidth_from_metrics(gpu_metrics),
        notes=apple_snapshot_notes(gpu_metrics),
    )


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
        )

    @staticmethod
    def from_file(path: Path) -> "CorinthCanalArtifact":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return CorinthCanalArtifact.from_dict(raw)


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
    if corinth:
        base = routing or RoutingMetrics()
        routing = RoutingMetrics(
            routing_entropy=corinth.routing_entropy if corinth.routing_entropy is not None else base.routing_entropy,
            spike_density=corinth.spike_density if corinth.spike_density is not None else base.spike_density,
            latent_stability=corinth.latent_stability if corinth.latent_stability is not None else base.latent_stability,
            dv_dt_reductions=corinth.dv_dt_reductions if corinth.dv_dt_reductions is not None else base.dv_dt_reductions,
            event_rate=corinth.event_rate if corinth.event_rate is not None else base.event_rate,
        )

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
        notes=telemetry.notes,
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
    d["routing_entropy"] = telemetry.routing.routing_entropy if telemetry.routing else None
    d["spike_density"] = telemetry.routing.spike_density if telemetry.routing else None
    d["latent_stability"] = telemetry.routing.latent_stability if telemetry.routing else None
    d["dv_dt_reductions"] = telemetry.routing.dv_dt_reductions if telemetry.routing else None
    d["event_rate"] = telemetry.routing.event_rate if telemetry.routing else None
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
    from benchmarks.jsonio import write_json

    write_json(path, telemetry_to_dict(telemetry))
