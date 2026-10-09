"""Rendering inputs, not a qualification engine or an evaluator (#45/#49).

Prepared summaries retain upstream definitions and intervals. Legacy GOZ1
observations use the existing importer and are explicitly unqualified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from combine_for_ai.goz_import import (
    SCHEMA_MULTIBLOCK_V1,
    SCHEMA_ROUTE_PRESERVATION_V1,
    detect_experiment_kind,
    GozExperimentKind,
    import_multiblock_metrics,
    import_route_preservation,
)

Text = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0)]
Status = Literal[
    "ok", "failed", "skipped", "interrupted", "unsupported", "missing", "unknown"
]


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
        if self.metric_provenance == "measured" and self.execution_kind != "execution":
            raise ValueError(
                "measured metrics require execution, not mock/inspection/import"
            )
        if (
            self.metric_provenance == "imported-measurement"
            and self.execution_kind != "import"
        ):
            raise ValueError(
                "imported-measurement metrics require import execution kind"
            )
        if self.execution_kind in {
            "mock",
            "metadata-only",
        } and self.metric_provenance not in {"unknown", "calculated-proxy"}:
            raise ValueError("mock/header-only observations cannot claim measurements")
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("duplicate metric names in a record")
        return self


class EvidenceArchive(RenderingModel):
    schema_version: Literal["combine.plot-evidence.v1"]
    run: dict[str, Any] = Field(default_factory=dict)
    records: list[EvidenceRecord]


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    """Hash the exact bytes parsed, so source identity cannot race a second read."""
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: input root must be a JSON object")
    return raw, hashlib.sha256(data).hexdigest()


# Use declared GOZ1 units; the route schema's undeclared JS log base remains
# unknown below. Legacy speed/VRAM presets are NOT added to this allowlist.
_GOZ_METRICS = {
    "route_top1_agreement": ("fraction", "Top-1 router agreement with FP16", "router"),
    "route_top2_agreement": (
        "fraction",
        "Top-2 router set agreement with FP16",
        "router",
    ),
    "block_output_cosine": (
        "cosine",
        "Block output cosine similarity with FP16",
        "block",
    ),
    "resid_in_drift": (
        "relative norm",
        "Residual stream drift relative to FP16",
        "block",
    ),
    "expert_load_js": ("bits", "Expert-load Jensen-Shannon divergence", "router"),
}


def _legacy_records(raw: dict[str, Any], path: Path) -> list[EvidenceRecord]:
    run = raw.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("legacy run must be an object")
    if "results" in raw:
        if raw.get("combine_import_schema", raw.get("schema_version", 1)) not in (
            1,
            "1",
            "v1",
            "1.0",
        ):
            raise ValueError("unsupported legacy schema_version")
        if run.get("schema") not in {
            SCHEMA_MULTIBLOCK_V1,
            SCHEMA_ROUTE_PRESERVATION_V1,
        }:
            raise ValueError(
                f"{path}: legacy report lacks a supported GOZ1 schema; prepare combine.plot-evidence.v1 summaries (do not promote mock presets)"
            )
        rows = raw["results"]
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError(f"{path}: results must be an array of objects")
    else:
        kind = detect_experiment_kind(raw)
        if kind is GozExperimentKind.MULTIBLOCK:
            chain = raw["chain"]
            entries = chain["per_block"]
            if any(not isinstance(e, dict) for e in entries):
                raise ValueError("chain.per_block must contain only objects")
            blocks = chain.get("blocks", [e.get("block") for e in entries])
            if not isinstance(blocks, list) or any(
                type(b) is not int or b < 0 for b in blocks
            ):
                raise ValueError("chain.blocks must be nonnegative integers")
            if len(blocks) != len(set(blocks)):
                raise ValueError("duplicate chain.blocks")
            by_block = {}
            for entry in entries:
                block = entry.get("block")
                if type(block) is not int or block < 0 or block in by_block:
                    raise ValueError(
                        "chain.per_block requires unique nonnegative block indices"
                    )
                by_block[block] = entry
            # Missing expected arms/declared blocks stay visible rather than
            # disappearing through the importer's successful-row-only path.
            padded = []
            for block in dict.fromkeys([*blocks, *by_block]):
                entry = dict(by_block.get(block, {"block": block}))
                for arm in ("expert_only", "fp16_control"):
                    if entry.get(arm) is None:
                        entry[arm] = {}
                    if not isinstance(entry[arm], dict):
                        raise ValueError(
                            f"chain.per_block {arm} must be an object or null"
                        )
                padded.append(entry)
            if not padded:
                return []
            imported = import_multiblock_metrics(
                {**raw, "chain": {**chain, "per_block": padded}}, source_path=str(path)
            )
        else:
            imported = import_route_preservation(raw, source_path=str(path))
        rows = [r.to_report_row() for r in imported.rows]
        if kind is GozExperimentKind.MULTIBLOCK:
            for row in rows:
                declared_status = by_block.get(row["block_index"], {}).get(
                    row["arm"], {}
                )
                if isinstance(declared_status, dict) and "status" in declared_status:
                    row["status"] = declared_status["status"]
    provenance = raw.get("provenance") or {}
    certification = raw.get("certification") or {}
    summary = raw.get("summary") or []
    if not isinstance(certification, dict) or not isinstance(summary, list):
        raise ValueError("legacy certification/summary must be an object/array")
    # Only explicit fixture markers count; file names/paths never qualify data.
    fixture = (
        isinstance(provenance, dict) and provenance.get("agent") == "fixture"
    ) or certification.get("basis") == "fixture"
    statuses = {
        item.get("name"): item.get("status")
        for item in summary
        if isinstance(item, dict)
    }
    aliases = {
        "route_top1_agreement": "router_top1_agreement",
        "route_top2_agreement": "router_top2_set_agreement",
        "resid_in_drift": "residual_stream_drift",
        "expert_load_js": "expert_load_js_divergence",
    }
    records = []
    for row in rows:
        metrics = []
        for name, (unit, definition, scope) in _GOZ_METRICS.items():
            if (
                name == "expert_load_js"
                and row.get("schema", run.get("schema")) != SCHEMA_MULTIBLOCK_V1
            ):
                unit = "unknown"
                definition += " (log base undeclared by legacy route schema)"
            value = row.get(name)
            try:
                value = (
                    None if value is None or isinstance(value, bool) else float(value)
                )
            except (TypeError, ValueError, OverflowError):
                value = None
            if value is not None and not math.isfinite(value):
                value = None
            gate = statuses.get(aliases.get(name, name))
            execution_status = row.get("status")
            status = (
                execution_status
                if execution_status not in (None, "ok")
                else "ok"
                if value is not None
                else "missing"
            )
            if status != "ok":
                value = None
            notes = (
                f"Upstream gate status: {gate}"
                if gate
                else "Upstream gate status: unknown"
            )
            if execution_status:
                notes += f"; upstream execution status: {execution_status}"
            metrics.append(
                PreparedMetric(
                    name=name,
                    unit=unit,
                    definition=definition,
                    scope=scope,
                    value=value,
                    status=status,
                    notes=notes,
                )
            )
        records.append(
            EvidenceRecord(
                workload_track="compression-analysis",
                execution_kind="import",
                input_provenance="fixture" if fixture else "unknown",
                metric_provenance="unknown",
                series=str(row.get("arm") or "unknown"),
                context={
                    "model_family": row.get("model_family"),
                    "tokens": row.get("tokens"),
                    "seed": row.get("seed"),
                    "schema": row.get("schema", run.get("schema")),
                    "source_path": row.get("source_path", str(path)),
                    **{key: row[key] for key in _CONTEXT_KEYS if key in row},
                    "declared_context": row.get("context"),
                },
                # Tokens are NOT independent sample/seed counts.
                seed_policy=f"single imported observation; seed={row.get('seed', 'unknown')}",
                block_index=row.get("block_index"),
                metrics=metrics,
            )
        )
    return records


_CONTEXT_KEYS = (
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


def _known(value: Any) -> bool:
    if value is None or value == "unknown" or value == "":
        return False
    if isinstance(value, (dict, list)):
        return bool(value) and all(
            _known(v) for v in (value.values() if isinstance(value, dict) else value)
        )
    return True


def _cohorts(
    records: list[EvidenceRecord], source: str, *, legacy: bool
) -> list[dict[str, Any]]:
    cohorts: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        fields = record.model_dump(
            exclude={"series", "metrics", "block_index", "source_artifacts"}
        )
        complete = all(_known(record.context.get(k)) for k in _CONTEXT_KEYS)
        complete = (
            complete
            and record.sample_count is not None
            and record.repeat_count is not None
            and record.seed_policy != "unknown"
        )
        policy = (
            "legacy observations: unqualified"
            if legacy
            else "compatible prepared summaries"
            if complete
            else "isolated: incomplete context"
        )
        key = json.dumps(fields, sort_keys=True, allow_nan=False)
        if not legacy and not complete:
            key += f":{index}"
        if key not in cohorts:
            cid = hashlib.sha256(f"{source}:{key}".encode()).hexdigest()[:16]
            cohorts[key] = {
                **fields,
                "id": cid,
                "source": source,
                "comparison_policy": policy,
                "metrics": [],
            }
        cohort = cohorts[key]
        for metric in record.metrics:
            definition = metric.model_dump(
                include={"name", "unit", "definition", "scope"}
            )
            match = next(
                (
                    m
                    for m in cohort["metrics"]
                    if all(m[k] == v for k, v in definition.items())
                ),
                None,
            )
            if match is None:
                match = {**definition, "observations": []}
                cohort["metrics"].append(match)
            observation = metric.model_dump(
                exclude={"name", "unit", "definition", "scope"}
            )
            observation.update(
                series=record.series,
                block_index=record.block_index,
                source_artifacts=record.source_artifacts,
            )
            if any(
                o["series"] == record.series and o["block_index"] == record.block_index
                for o in match["observations"]
            ):
                raise ValueError(
                    f"{source}: duplicate observation for {record.series}/{metric.name}/block={record.block_index}; prepare repeats upstream"
                )
            match["observations"].append(observation)
    return list(cohorts.values())


def load_report_evidence(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one archived input is required")
    sources, cohorts = [], []
    seen = set()
    for path in paths:
        path = Path(path)
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate archived input: {path}")
        seen.add(resolved)
        raw, digest = read_json(path)
        legacy = (
            "records" not in raw
            and raw.get("schema_version") != "combine.plot-evidence.v1"
        )
        try:
            records = (
                _legacy_records(raw, path)
                if legacy
                else EvidenceArchive.model_validate(raw).records
            )
            cohorts.extend(_cohorts(records, str(path), legacy=legacy))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid archive {path}: {exc}") from exc
        sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "run": raw.get("run", {}),
                "provenance": raw.get("provenance", {}),
                "legacy": legacy,
            }
        )
    return sources, cohorts
