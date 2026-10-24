"""Validated rendering records. This module does not score or compare them."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


Text = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0)]
Status = Literal[
    "ok", "failed", "skipped", "interrupted", "unsupported", "missing", "unknown"
]

# Required comparison settings. Any other context field is part of the same
# completeness check: nested null, unknown, or empty values isolate the record.
CONTEXT_KEYS = (
    "task",
    "scorer",
    "dataset",
    "split",
    "sample_ids",
    "engine",
    "config",
    "runtime",
    "harness",
    "environment",
    "budgets",
    "preprocessing",
    "seeds",
)


class RenderingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PreparedInterval(RenderingModel):
    low: FiniteFloat
    high: FiniteFloat
    confidence_level: Annotated[FiniteFloat, Field(gt=0, lt=1)]
    method: Text
    experimental_unit: Text
    library: Literal["scipy"]
    library_version: Text
    definition: Text

    @model_validator(mode="after")
    def ordered(self):
        if self.low > self.high:
            raise ValueError("interval low must not exceed high")
        return self


class PreparedMetric(RenderingModel):
    name: Text
    unit: Text
    definition: Text
    scope: Text
    value: FiniteFloat | None = None
    status: Status
    counts: dict[Status, Count] = Field(default_factory=dict)
    interval: PreparedInterval | None = None
    notes: str = ""

    @model_validator(mode="after")
    def available(self):
        if (self.status == "ok") != (self.value is not None):
            raise ValueError(
                "only status=ok may have a value; unavailable metrics must be null"
            )
        if self.interval is not None and self.value is None:
            raise ValueError("an unavailable metric cannot have an interval")
        return self


class EvidenceRecord(RenderingModel):
    workload_track: Literal["language", "agent", "temporal", "compression-analysis"]
    execution_kind: Literal["mock", "metadata-only", "import", "execution", "unknown"]
    input_provenance: Literal["fixture", "actual", "unknown"]
    metric_provenance: Literal[
        "measured", "imported-measurement", "calculated-proxy", "unknown"
    ]
    series: Text
    context: dict[str, Any] = Field(default_factory=dict)
    sample_count: Count | None = None
    repeat_count: Count | None = None
    seed_policy: Text = "unknown"
    block_index: Count | None = None
    metrics: list[PreparedMetric]
    source_artifacts: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def truthful(self):
        _reject_untruthful_provenance(self)
        if len({metric.name for metric in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric names in a record")
        return self


class EvidenceArchive(RenderingModel):
    schema_version: Literal["combine.plot-evidence.v1"]
    run: dict[str, Any] = Field(default_factory=dict)
    records: list[EvidenceRecord]


def _reject_untruthful_provenance(record: EvidenceRecord) -> None:
    if _measured_without_execution(record):
        raise ValueError("measured metrics require execution, not mock/inspection/import")
    if _import_metric_without_import(record):
        raise ValueError("imported-measurement metrics require import execution kind")
    if _mock_claiming_measurement(record):
        raise ValueError("mock/header-only observations cannot claim measurements")


def _measured_without_execution(record: EvidenceRecord) -> bool:
    return record.metric_provenance == "measured" and record.execution_kind != "execution"


def _import_metric_without_import(record: EvidenceRecord) -> bool:
    return (
        record.metric_provenance == "imported-measurement"
        and record.execution_kind != "import"
    )


def _mock_claiming_measurement(record: EvidenceRecord) -> bool:
    header_only = record.execution_kind in {"mock", "metadata-only"}
    allowed = record.metric_provenance in {"unknown", "calculated-proxy"}
    return header_only and not allowed
