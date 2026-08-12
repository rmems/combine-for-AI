from __future__ import annotations

import json
import tempfile
from pathlib import Path


from benchmarks.telemetry import (
    CorinthCanalArtifact,
    GPUMetrics,
    MyelinAcceleratorArtifact,
    RoutingMetrics,
    SystemSnapshot,
    TelemetrySnapshot,
    collect_system_snapshot,
    merge_upstream_artifacts,
    telemetry_to_dict,
    write_telemetry_json,
)


SAMPLE_CORINTH = {
    "artifact_version": "0.1.0",
    "experiment_id": "saaq-lambda-001",
    "routing_entropy": 2.718,
    "spike_density": 0.42,
    "event_rate": 150.0,
    "latent_stability": 0.95,
    "dv_dt_reductions": 0.12,
    "raw_path": "experiments/saaq-lambda-001/routing.json",
}

SAMPLE_MYELIN = {
    "artifact_version": "0.2.0",
    "benchmark_id": "myelin-conv2d-001",
    "kernel_occupancy": 0.78,
    "vram_bandwidth_gbps": 900.0,
    "gpu_utilization_percent": 85.0,
    "latency_ms": 4.2,
    "throughput_tops": 125.0,
    "raw_path": "experiments/myelin-conv2d-001/benchmark.json",
}


def test_system_snapshot_fields() -> None:
    snap = collect_system_snapshot()
    assert snap.cpu_count_logical is None or isinstance(snap.cpu_count_logical, int)
    assert isinstance(snap.platform, str)
    assert isinstance(snap.python_version, str)


def test_telemetry_to_dict() -> None:
    system = SystemSnapshot(
        cpu_count_logical=8,
        cpu_count_physical=4,
        memory_total_gb=32.0,
        memory_available_gb=16.0,
        gpu_count=1,
        gpu_names=["NVIDIA GeForce RTX 4090"],
        gpu_driver_version="545.23",
        cuda_version="12.4",
        platform="Linux-6.5-x86_64",
        python_version="3.14.0",
    )
    routing = RoutingMetrics(
        routing_entropy=2.5,
        spike_density=0.3,
        latent_stability=0.95,
        dv_dt_reductions=0.1,
        event_rate=120.0,
    )
    gpu = GPUMetrics(
        index=0,
        name="RTX 4090",
        utilization_percent=85.0,
        memory_used_mb=8000,
        memory_total_mb=24000,
        temperature_c=65,
        power_draw_w=250.0,
        clock_sm_mhz=2000,
        clock_memory_mhz=10500,
    )
    telemetry = TelemetrySnapshot(
        system=system,
        gpu_metrics=[gpu],
        routing=routing,
        kernel_occupancy=0.78,
        vram_bandwidth_gbps=900.0,
        notes="test",
    )
    d = telemetry_to_dict(telemetry)
    assert d["sys_cpu_count_logical"] == 8
    assert d["routing_entropy"] == 2.5
    assert d["spike_density"] == 0.3
    assert d["kernel_occupancy"] == 0.78
    assert d["vram_bandwidth_gbps"] == 900.0
    assert d["telemetry_notes"] == "test"
    assert len(d["gpu_metrics"]) == 1
    assert d["gpu_metrics"][0]["name"] == "RTX 4090"


def test_telemetry_dict_no_gpu() -> None:
    system = SystemSnapshot(
        cpu_count_logical=4,
        cpu_count_physical=2,
        memory_total_gb=None,
        memory_available_gb=None,
        gpu_count=None,
        gpu_names=None,
        gpu_driver_version=None,
        cuda_version=None,
        platform="Linux",
        python_version="3.14.0",
    )
    telemetry = TelemetrySnapshot(system=system)
    d = telemetry_to_dict(telemetry)
    assert d["sys_cpu_count_logical"] == 4
    assert "gpu_metrics" not in d
    assert d["routing_entropy"] is None


def test_write_telemetry_json() -> None:
    system = SystemSnapshot(
        cpu_count_logical=4,
        cpu_count_physical=2,
        memory_total_gb=16.0,
        memory_available_gb=8.0,
        gpu_count=None,
        gpu_names=None,
        gpu_driver_version=None,
        cuda_version=None,
        platform="Linux",
        python_version="3.14.0",
    )
    telemetry = TelemetrySnapshot(system=system, vram_bandwidth_gbps=100.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "telemetry.json"
        write_telemetry_json(path, telemetry)
        assert path.exists()
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["sys_cpu_count_logical"] == 4
        assert loaded["vram_bandwidth_gbps"] == 100.0


def test_corinth_canal_from_dict() -> None:
    artifact = CorinthCanalArtifact.from_dict(SAMPLE_CORINTH)
    assert artifact.artifact_version == "0.1.0"
    assert artifact.experiment_id == "saaq-lambda-001"
    assert artifact.routing_entropy == 2.718
    assert artifact.spike_density == 0.42
    assert artifact.dv_dt_reductions == 0.12


def test_corinth_canal_from_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(SAMPLE_CORINTH, tmp)
        tmp_path = Path(tmp.name)
    try:
        artifact = CorinthCanalArtifact.from_file(tmp_path)
        assert artifact.experiment_id == "saaq-lambda-001"
    finally:
        tmp_path.unlink()


def test_myelin_accelerator_from_dict() -> None:
    artifact = MyelinAcceleratorArtifact.from_dict(SAMPLE_MYELIN)
    assert artifact.artifact_version == "0.2.0"
    assert artifact.benchmark_id == "myelin-conv2d-001"
    assert artifact.kernel_occupancy == 0.78
    assert artifact.vram_bandwidth_gbps == 900.0
    assert artifact.throughput_tops == 125.0


def test_myelin_accelerator_from_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(SAMPLE_MYELIN, tmp)
        tmp_path = Path(tmp.name)
    try:
        artifact = MyelinAcceleratorArtifact.from_file(tmp_path)
        assert artifact.benchmark_id == "myelin-conv2d-001"
    finally:
        tmp_path.unlink()


def test_merge_upstream_corinth_only() -> None:
    system = SystemSnapshot(
        cpu_count_logical=4,
        cpu_count_physical=2,
        memory_total_gb=16.0,
        memory_available_gb=8.0,
        gpu_count=None,
        gpu_names=None,
        gpu_driver_version=None,
        cuda_version=None,
        platform="Linux",
        python_version="3.14.0",
    )
    telemetry = TelemetrySnapshot(system=system)
    corinth = CorinthCanalArtifact.from_dict(SAMPLE_CORINTH)
    merged = merge_upstream_artifacts(telemetry, corinth=corinth)
    assert merged.routing is not None
    assert merged.routing.routing_entropy == 2.718
    assert merged.routing.spike_density == 0.42
    assert merged.kernel_occupancy is None


def test_merge_upstream_myelin_only() -> None:
    system = SystemSnapshot(
        cpu_count_logical=4,
        cpu_count_physical=2,
        memory_total_gb=16.0,
        memory_available_gb=8.0,
        gpu_count=None,
        gpu_names=None,
        gpu_driver_version=None,
        cuda_version=None,
        platform="Linux",
        python_version="3.14.0",
    )
    telemetry = TelemetrySnapshot(system=system)
    myelin = MyelinAcceleratorArtifact.from_dict(SAMPLE_MYELIN)
    merged = merge_upstream_artifacts(telemetry, myelin=myelin)
    assert merged.kernel_occupancy == 0.78
    assert merged.vram_bandwidth_gbps == 900.0
    assert merged.routing is None


def test_merge_upstream_both() -> None:
    system = SystemSnapshot(
        cpu_count_logical=4,
        cpu_count_physical=2,
        memory_total_gb=16.0,
        memory_available_gb=8.0,
        gpu_count=None,
        gpu_names=None,
        gpu_driver_version=None,
        cuda_version=None,
        platform="Linux",
        python_version="3.14.0",
    )
    telemetry = TelemetrySnapshot(system=system)
    corinth = CorinthCanalArtifact.from_dict(SAMPLE_CORINTH)
    myelin = MyelinAcceleratorArtifact.from_dict(SAMPLE_MYELIN)
    merged = merge_upstream_artifacts(telemetry, corinth=corinth, myelin=myelin)
    assert merged.routing is not None
    assert merged.routing.routing_entropy == 2.718
    assert merged.kernel_occupancy == 0.78
    assert merged.vram_bandwidth_gbps == 900.0


def test_corinth_canal_preserves_existing_routing() -> None:
    system = SystemSnapshot(
        cpu_count_logical=4,
        cpu_count_physical=2,
        memory_total_gb=16.0,
        memory_available_gb=8.0,
        gpu_count=None,
        gpu_names=None,
        gpu_driver_version=None,
        cuda_version=None,
        platform="Linux",
        python_version="3.14.0",
    )
    existing = RoutingMetrics(routing_entropy=1.0, spike_density=0.1)
    telemetry = TelemetrySnapshot(system=system, routing=existing)
    partial = {"artifact_version": "0.1.0", "experiment_id": "x", "spike_density": 0.99}
    corinth = CorinthCanalArtifact.from_dict(partial)
    merged = merge_upstream_artifacts(telemetry, corinth=corinth)
    assert merged.routing is not None
    assert merged.routing.routing_entropy == 1.0  # preserved
    assert merged.routing.spike_density == 0.99  # overwritten


def test_load_fixture_files() -> None:
    """Ensure committed fixture artifacts validate successfully."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    corinth_path = fixture_dir / "corinth_canal.sample.json"
    myelin_path = fixture_dir / "myelin_accelerator.sample.json"
    assert corinth_path.exists(), f"missing fixture: {corinth_path}"
    assert myelin_path.exists(), f"missing fixture: {myelin_path}"
    c = CorinthCanalArtifact.from_file(corinth_path)
    assert c.experiment_id == "saaq-lambda-001"
    m = MyelinAcceleratorArtifact.from_file(myelin_path)
    assert m.benchmark_id == "myelin-conv2d-001"
