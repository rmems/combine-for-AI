"""Record builders shared by evidence tests and the conformance fixture generator."""

from __future__ import annotations

from combine_for_ai.evidence import (
    Budget,
    DataProvenance,
    EvidenceRecord,
    ExecutionKind,
    MetricProvenance,
    MetricStatus,
    MetricValue,
    Versions,
    WorkloadTrack,
    unavailable_metric,
)

SHA_A = "a" * 63 + "1"
SHA_B = "b" * 63 + "2"


def ok_metric(name, value, provenance, *, unit="ratio", sha=None, native=None):
    return MetricValue(
        name=name,
        native_name=native or name,
        unit=unit,
        definition=f"{name} definition",
        value=value,
        status=MetricStatus.OK,
        provenance=provenance,
        source_artifact_sha256=sha,
    )


def mock_record() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="mock-1",
        workload_track=WorkloadTrack.LIKELIHOOD,
        architecture="moe",
        execution_kind=ExecutionKind.MOCK,
        data_provenance=DataProvenance.FIXTURE,
        capabilities=("generation", "likelihood"),
        seed=7,
        budget=Budget(max_samples=4),
        metrics=(
            ok_metric("accuracy", 0.5, MetricProvenance.MOCK_COMPUTED),
            ok_metric("throughput", 1200.0, MetricProvenance.PRESET, unit="tokens/s"),
        ),
    )


def metadata_only_record() -> EvidenceRecord:
    reason = "header-only inspection"
    return EvidenceRecord(
        record_id="meta-1",
        execution_kind=ExecutionKind.METADATA_ONLY,
        data_provenance=DataProvenance.UNKNOWN,
        source_artifact_sha256=SHA_A,
        metrics=tuple(
            unavailable_metric(n, status=MetricStatus.UNSUPPORTED, reason=reason)
            for n in ("latency_ms", "accuracy", "throughput", "vram_gb", "energy_j")
        ),
    )


def imported_record(data: DataProvenance, record_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        workload_track=WorkloadTrack.LIKELIHOOD,
        architecture="snn",
        execution_kind=ExecutionKind.IMPORTED,
        data_provenance=data,
        capabilities=("likelihood",),
        versions=Versions(engine="grok-ozempic 0.9", task="lambada@1", scorer="ppl-1", data="r1", model="m@abc", runtime="py3.14"),
        seed=-3,
        environment_identity="env:" + SHA_B,
        source_artifact_sha256=SHA_A,
        metrics=(
            ok_metric("mean_logprob", -1.25e-3, MetricProvenance.IMPORTED, unit="nats", sha=SHA_A, native="mean_ll"),
            unavailable_metric("energy_j", reason="not in source file"),
        ),
        notes="line one | pipe\nline two",
    )


def measured_record() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="meas-1",
        workload_track=WorkloadTrack.GENERATION,
        architecture="dense",
        execution_kind=ExecutionKind.MEASURED,
        data_provenance=DataProvenance.REAL,
        requested_data_provenance=DataProvenance.REAL,
        capabilities=("generation",),
        environment_identity="env:" + SHA_B,
        budget=Budget(max_tokens=512, time_limit_s=30.0),
        metrics=(ok_metric("latency_ms", 12.5, MetricProvenance.MEASURED, unit="ms"),),
    )


ALL_FIXTURES = {
    "mock": mock_record,
    "metadata_only": metadata_only_record,
    "imported_real": lambda: imported_record(DataProvenance.REAL, "imp-real"),
    "imported_synthetic": lambda: imported_record(DataProvenance.SYNTHETIC, "imp-synth"),
    "measured_shaped": measured_record,
}
