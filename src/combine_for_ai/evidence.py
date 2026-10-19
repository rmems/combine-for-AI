"""Versioned evidence envelope for benchmark results.

One record says *how* a result was produced and *how far it can be trusted*.
Five facts stay independent so none can be inferred from another:

* ``workload_track`` (what is asked of the model) vs ``architecture`` (what the
  model is);
* ``execution_kind`` (mock, metadata_only, imported, measured);
* ``data_provenance`` (fixture, synthetic, real, unknown) -- an imported file
  can hold fixture data or real measurements;
* per-metric ``provenance`` (preset, mock_computed, imported, measured,
  unavailable);
* ``capabilities`` the adapter declared. Registering a quantization profile
  establishes none of them.

Missing, unsupported or errored metrics carry ``value=None``: they can never
turn into ``0.0`` or a success. Mock performance presets live only in
``benchmarks.fixture_presets``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)


EVIDENCE_SCHEMA = "combine_for_ai.evidence"
EVIDENCE_SCHEMA_VERSION = 1
#: Pre-envelope payloads (artifact-smoke rows, matrix cells) carry no version.
LEGACY_SCHEMA_VERSION = 0

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

NonEmpty = Annotated[str, StringConstraints(min_length=1)]


class EvidenceError(ValueError):
    """Raised when an evidence payload is invalid or cannot be migrated."""


class CapabilityError(EvidenceError):
    """Raised when an adapter lacks a capability a workload requires."""


class ExecutionKind(str, Enum):
    MOCK = "mock"
    METADATA_ONLY = "metadata_only"
    IMPORTED = "imported"
    MEASURED = "measured"


class DataProvenance(str, Enum):
    FIXTURE = "fixture"
    SYNTHETIC = "synthetic"
    REAL = "real"
    UNKNOWN = "unknown"


class MetricProvenance(str, Enum):
    PRESET = "preset"  # invented constant, mock/fixture only
    MOCK_COMPUTED = "mock_computed"  # derived from mock adapter output
    IMPORTED = "imported"  # read from an external artifact
    MEASURED = "measured"  # measured by this harness
    UNAVAILABLE = "unavailable"


class MetricStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class Capability(str, Enum):
    GENERATION = "generation"
    LIKELIHOOD = "likelihood"
    TEMPORAL = "temporal"
    TOOL_EXECUTION = "tool_execution"


class WorkloadTrack(str, Enum):
    """What is asked of the model. Never encodes architecture."""

    LIKELIHOOD = "likelihood"
    GENERATION = "generation"
    TEMPORAL = "temporal"
    TOOL_EXECUTION = "tool_execution"

    @property
    def required_capability(self) -> Capability:
        return Capability(self.value)


def negotiate_capabilities(
    required: frozenset[Capability] | set[Capability],
    provided: frozenset[Capability] | set[Capability],
) -> tuple[frozenset[Capability], frozenset[Capability]]:
    """Return ``(supported, missing)`` for a workload's required capabilities."""
    required = frozenset(required)
    provided = frozenset(provided)
    return required & provided, required - provided


def require_capabilities(
    required: frozenset[Capability] | set[Capability],
    provided: frozenset[Capability] | set[Capability],
) -> None:
    _, missing = negotiate_capabilities(required, provided)
    if missing:
        names = ", ".join(sorted(cap.value for cap in missing))
        raise CapabilityError(f"adapter does not declare required capabilities: {names}")


class MetricValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: NonEmpty
    native_name: NonEmpty | None = None
    unit: NonEmpty | None = None
    definition: NonEmpty | None = None
    value: float | None = None
    status: MetricStatus
    provenance: MetricProvenance
    source_artifact_sha256: str | None = None
    reason: NonEmpty | None = None

    @field_validator("source_artifact_sha256")
    @classmethod
    def _sha_shape(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("source_artifact_sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _status_rules(self) -> "MetricValue":
        if self.status is MetricStatus.OK:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError(f"metric '{self.name}': ok requires a finite value")
            if self.provenance is MetricProvenance.UNAVAILABLE:
                raise ValueError(f"metric '{self.name}': ok cannot have unavailable provenance")
            for field in ("native_name", "unit", "definition"):
                if getattr(self, field) is None:
                    raise ValueError(f"metric '{self.name}': ok requires {field}")
            if self.provenance is MetricProvenance.IMPORTED and self.source_artifact_sha256 is None:
                raise ValueError(f"metric '{self.name}': imported values require source_artifact_sha256")
        else:
            if self.value is not None:
                raise ValueError(
                    f"metric '{self.name}': status {self.status.value} must not carry a value"
                )
            if self.provenance is not MetricProvenance.UNAVAILABLE:
                raise ValueError(
                    f"metric '{self.name}': status {self.status.value} requires unavailable provenance"
                )
        return self


def unavailable_metric(
    name: str,
    *,
    status: MetricStatus = MetricStatus.UNAVAILABLE,
    reason: str | None = None,
    unit: str | None = None,
    native_name: str | None = None,
    definition: str | None = None,
) -> MetricValue:
    if status is MetricStatus.OK:
        raise ValueError("unavailable_metric cannot produce an ok status")
    return MetricValue(
        name=name,
        native_name=native_name,
        unit=unit,
        definition=definition,
        value=None,
        status=status,
        provenance=MetricProvenance.UNAVAILABLE,
        reason=reason,
    )


class Versions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: NonEmpty | None = None
    task: NonEmpty | None = None
    scorer: NonEmpty | None = None
    data: NonEmpty | None = None
    model: NonEmpty | None = None
    runtime: NonEmpty | None = None


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_samples: int | None = Field(None, ge=0)
    max_tokens: int | None = Field(None, ge=0)
    time_limit_s: float | None = Field(None, ge=0)


_PROVENANCE_FOR_EXECUTION: dict[ExecutionKind, frozenset[MetricProvenance]] = {
    ExecutionKind.MOCK: frozenset({MetricProvenance.PRESET, MetricProvenance.MOCK_COMPUTED}),
    ExecutionKind.IMPORTED: frozenset({MetricProvenance.IMPORTED}),
    ExecutionKind.MEASURED: frozenset({MetricProvenance.MEASURED}),
    ExecutionKind.METADATA_ONLY: frozenset(),
}


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: str = Field(EVIDENCE_SCHEMA, alias="schema")
    schema_version: int = EVIDENCE_SCHEMA_VERSION
    record_id: NonEmpty
    workload_track: WorkloadTrack | None = None
    architecture: NonEmpty | None = None
    execution_kind: ExecutionKind
    data_provenance: DataProvenance
    requested_data_provenance: DataProvenance | None = None
    capabilities: tuple[Capability, ...] = ()
    versions: Versions = Field(default_factory=Versions)
    seed: int | None = None
    budget: Budget = Field(default_factory=Budget)
    environment_identity: NonEmpty | None = None
    source_artifact_sha256: str | None = None
    metrics: tuple[MetricValue, ...] = ()
    notes: NonEmpty | None = None

    @field_validator("source_artifact_sha256")
    @classmethod
    def _sha_shape(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("source_artifact_sha256 must be 64 lowercase hex characters")
        return value

    @field_validator("schema_id")
    @classmethod
    def _schema_name(cls, value: str) -> str:
        if value != EVIDENCE_SCHEMA:
            raise ValueError(f"unknown evidence schema '{value}'")
        return value

    @model_validator(mode="after")
    def _cross_rules(self) -> "EvidenceRecord":
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema_version {self.schema_version}")
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("duplicate metric names in one record")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate capabilities")

        allowed = _PROVENANCE_FOR_EXECUTION[self.execution_kind]
        ok_metrics = [m for m in self.metrics if m.status is MetricStatus.OK]
        if self.execution_kind is ExecutionKind.METADATA_ONLY and ok_metrics:
            raise ValueError(
                "metadata_only records cannot claim metric values (header inspection is not inference)"
            )
        for metric in ok_metrics:
            if metric.provenance not in allowed:
                raise ValueError(
                    f"metric '{metric.name}': provenance {metric.provenance.value} "
                    f"is invalid for execution_kind {self.execution_kind.value}"
                )
        if ok_metrics and self.workload_track is not None:
            if self.workload_track.required_capability not in self.capabilities:
                raise ValueError(
                    f"workload_track {self.workload_track.value} requires the "
                    f"{self.workload_track.required_capability.value} capability, which was not declared"
                )
        if (
            self.requested_data_provenance is not None
            and self.requested_data_provenance is not self.data_provenance
            and ok_metrics
        ):
            raise ValueError(
                f"requested {self.requested_data_provenance.value} data but record holds "
                f"{self.data_provenance.value} data; refusing to substitute"
            )
        if self.execution_kind is ExecutionKind.MEASURED and ok_metrics:
            if self.environment_identity is None:
                raise ValueError("measured records require environment_identity")
        if self.execution_kind is ExecutionKind.MOCK and self.data_provenance is DataProvenance.REAL:
            raise ValueError("mock execution cannot claim real data")
        return self

    def metric(self, name: str) -> MetricValue | None:
        return next((m for m in self.metrics if m.name == name), None)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def parse_evidence(payload: Mapping[str, Any]) -> EvidenceRecord:
    try:
        return EvidenceRecord.model_validate(dict(payload))
    except ValidationError as exc:
        raise EvidenceError(str(exc)) from exc


# --------------------------------------------------------------------------
# Versioned migration. Each entry upgrades one payload from version N to N+1.

_LEGACY_METRICS: dict[str, tuple[str, str, str, MetricProvenance]] = {
    # canonical name: (native_name, unit, definition, provenance of a mock run)
    "accuracy": ("accuracy", "ratio", "fraction of scored records answered correctly", MetricProvenance.MOCK_COMPUTED),
    "perplexity": ("perplexity", "ratio", "exp of mean negative log-likelihood", MetricProvenance.MOCK_COMPUTED),
    "throughput": ("throughput", "tokens/s", "tokens processed per second of fixture time", MetricProvenance.PRESET),
    "latency_ms": ("latency_ms", "ms", "fixture time per record", MetricProvenance.PRESET),
    "vram_gb": ("vram_gb", "GB", "fixture memory preset", MetricProvenance.PRESET),
    "routing_entropy": ("routing_entropy", "nats", "entropy of the output-choice distribution", MetricProvenance.MOCK_COMPUTED),
}
_SENTINEL_FIELDS = frozenset({"accuracy", "routing_entropy"})
_SMOKE_TO_CANONICAL = {
    "perplexity": "perplexity",
    "throughput_tps": "throughput",
    "peak_vram_gb": "vram_gb",
}


def _legacy_metric(name: str, value: Any, *, trusted: bool = True) -> MetricValue:
    native, unit, definition, provenance = _LEGACY_METRICS[name]
    if (
        not trusted
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return unavailable_metric(name, reason="absent, non-finite or ambiguous placeholder in legacy payload", unit=unit)
    return MetricValue(
        name=name,
        native_name=native,
        unit=unit,
        definition=definition,
        value=float(value),
        status=MetricStatus.OK,
        provenance=provenance,
    )


def _legacy_envelope(record_id: str, metrics: list[MetricValue], seed: Any, notes: str) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_id": record_id,
        "execution_kind": ExecutionKind.MOCK.value,
        "data_provenance": DataProvenance.FIXTURE.value,
        "capabilities": [Capability.GENERATION.value, Capability.LIKELIHOOD.value],
        "seed": seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        "metrics": [m.model_dump(mode="json") for m in metrics],
        "notes": notes,
    }


def _migrate_v0(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Legacy artifact-smoke result rows and matrix cells -> envelope v1.

    Every legacy producer used the mock adapter, so values are labelled mock
    (presets for throughput/latency/vram). A failed smoke row keeps every
    metric unavailable instead of surfacing its former ``None``/``0``.
    """
    if isinstance(payload.get("result"), Mapping):
        payload = payload["result"]
    if isinstance(payload.get("metrics"), Mapping):  # matrix cell
        raw = payload["metrics"]
        # ``MetricsAccumulator.summary`` writes accuracy=0.0 (and, with no choices,
        # routing_entropy=0.0) as a fallback when nothing was scored, e.g. language
        # modeling cells. A legacy payload cannot tell that from a real 0% result,
        # so the ambiguous pair is dropped rather than reported as a measurement.
        sentinel = raw.get("accuracy") == 0 and raw.get("routing_entropy") == 0
        metrics = [
            _legacy_metric(name, raw.get(name), trusted=not (sentinel and name in _SENTINEL_FIELDS))
            for name in _LEGACY_METRICS
        ]
        record_id = str(payload.get("cell_id") or payload.get("fingerprint") or "legacy-cell")
        return _legacy_envelope(record_id, metrics, payload.get("seed"), "migrated from legacy matrix cell (v0)")
    if "run_id" in payload and "status" in payload:  # smoke row
        failed = payload.get("status") != "success"
        metrics = [
            _legacy_metric(canonical, payload.get(key), trusted=not failed)
            for key, canonical in _SMOKE_TO_CANONICAL.items()
        ]
        return _legacy_envelope(str(payload["run_id"]), metrics, None, "migrated from legacy artifact-smoke row (v0)")
    raise EvidenceError("unrecognised legacy payload; cannot migrate")


_MIGRATIONS: dict[int, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    LEGACY_SCHEMA_VERSION: _migrate_v0,
}


def load_evidence(payload: Mapping[str, Any]) -> EvidenceRecord:
    """Parse any supported payload version into the current envelope."""
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"evidence payload must be an object, got {type(payload).__name__}")
    # Smoke reports wrap a versioned envelope next to the legacy row; the
    # envelope is authoritative and must never be shadowed by re-migrating the row.
    embedded = payload.get("evidence")
    if "schema_version" not in payload and isinstance(embedded, Mapping):
        return load_evidence(embedded)
    version = payload.get("schema_version", LEGACY_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise EvidenceError(f"invalid schema_version {version!r}")
    if version > EVIDENCE_SCHEMA_VERSION or version < LEGACY_SCHEMA_VERSION:
        raise EvidenceError(f"unsupported evidence schema_version {version}")
    current: Mapping[str, Any] = payload
    while version < EVIDENCE_SCHEMA_VERSION:
        current = _MIGRATIONS[version](current)
        version += 1
    return parse_evidence(current)


__all__ = [
    "Budget",
    "Capability",
    "CapabilityError",
    "DataProvenance",
    "EVIDENCE_SCHEMA",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceError",
    "EvidenceRecord",
    "ExecutionKind",
    "LEGACY_SCHEMA_VERSION",
    "MetricProvenance",
    "MetricStatus",
    "MetricValue",
    "Versions",
    "WorkloadTrack",
    "load_evidence",
    "negotiate_capabilities",
    "parse_evidence",
    "require_capabilities",
    "unavailable_metric",
]
