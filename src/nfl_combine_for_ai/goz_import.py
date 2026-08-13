"""Import grok-ozempic experiment JSON into combine report rows.

Supports:
- Multiblock residual fidelity ``metrics.json`` (chain.per_block)
- Single-block route-preservation reports (pilot + summary)

Does not re-run experiments. Format ownership stays in rmems/grok-ozempic.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


SCHEMA_MULTIBLOCK_V1 = "grok_ozempic.multiblock_metrics.v1"
SCHEMA_ROUTE_PRESERVATION_V1 = "grok_ozempic.route_preservation.v1"


class GozExperimentKind(str, Enum):
    MULTIBLOCK = "multiblock"
    ROUTE_PRESERVATION = "route_preservation"


class GozImportError(ValueError):
    """Raised when an experiment artifact cannot be imported."""


@dataclass(frozen=True)
class ImportedExperimentRow:
    """One combine-normalized experiment row (JSON/CSV friendly)."""

    schema: str
    experiment_kind: str
    source_path: str
    model_family: str | None
    arm: str
    block_index: int | None
    route_top1_agreement: float | None
    route_top2_agreement: float | None
    block_output_cosine: float | None
    resid_in_drift: float | None
    expert_load_js: float | None
    scale_source: str | None
    goz1_version: int | None
    sparsity: float | None
    tokens: int | None
    seed: int | None
    label: str | None
    seconds: float | None
    pack_basename: str | None
    accuracy: float | None = None
    perplexity: float | None = None
    throughput: float | None = None
    latency_ms: float | None = None
    vram_gb: float | None = None
    routing_entropy: float | None = None
    spike_density: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_report_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra", {}) or {}
        # Flatten a few high-value extras without dumping huge arrays.
        for key in (
            "moe_output_cosine",
            "residual_stream_cosine",
            "block_output_drift_relative_norm",
            "mode",
            "decision",
            "experiment_id",
        ):
            if key in extra:
                row[f"extra_{key}"] = extra[key]
        return row


@dataclass(frozen=True)
class GozImportResult:
    kind: GozExperimentKind
    schema: str
    source_path: str
    rows: list[ImportedExperimentRow]
    provenance: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] | None = None

    def to_payload(self, *, run_id: str | None = None) -> dict[str, Any]:
        rid = run_id or _default_run_id()
        return {
            "run": {
                "run_id": rid,
                "run_name": f"goz-import-{self.kind.value}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_path": self.source_path,
                "schema": self.schema,
                "experiment_kind": self.kind.value,
            },
            "provenance": self.provenance,
            "decision": self.decision,
            "benchmark_linkage": {
                "nfl_combine_run_id": rid,
                "grok_ozempic_report_path": self.source_path,
                "grok_ozempic_experiment_id": (
                    (self.provenance or {}).get("issue")
                    or (self.provenance or {}).get("experiment_id")
                ),
            },
            "results": [row.to_report_row() for row in self.rows],
        }


def _default_run_id() -> str:
    return (
        f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}."
        f"{int(time.time() % 1 * 1000):03d}Z-goz-import"
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, "1", "v1", "1.0"})
_SUPPORTED_REPORT_FORMATS = frozenset({"json", "csv"})
_SCALE_RANK = ("pack_v3", "pack_v2", "legacy_oracle")


def _require_schema_version(raw: dict[str, Any]) -> None:
    """Accept missing version as v1; reject unknown explicit versions."""
    ver = raw.get("combine_import_schema", raw.get("schema_version"))
    if ver is None:
        return
    if ver not in _SUPPORTED_SCHEMA_VERSIONS:
        raise GozImportError(f"unsupported combine_import_schema version: {ver!r}")


def detect_experiment_kind(raw: dict[str, Any]) -> GozExperimentKind:
    _require_schema_version(raw)
    if isinstance(raw.get("chain"), dict) and isinstance(
        (raw.get("chain") or {}).get("per_block"), list
    ):
        return GozExperimentKind.MULTIBLOCK
    if isinstance(raw.get("pilot"), dict) and isinstance(raw.get("summary"), list):
        return GozExperimentKind.ROUTE_PRESERVATION
    raise GozImportError(
        "unknown grok-ozempic experiment shape: expected multiblock "
        "(chain.per_block) or route-preservation (pilot + summary)"
    )


def load_experiment_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GozImportError(f"cannot read experiment file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GozImportError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise GozImportError("experiment root must be a JSON object")
    return raw


def _summary_map(summary: list[Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in summary:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        observed = _as_float(item.get("observed"))
        if observed is not None:
            out[str(name)] = observed
    return out


def _prefer_scale_labels(labels: set[str]) -> str | None:
    for preferred in _SCALE_RANK:
        if preferred in labels:
            return preferred
    return next(iter(labels)) if labels else None


def _scale_from_sources_map(sources: Any) -> str | None:
    if not isinstance(sources, dict) or not sources:
        return None
    return _prefer_scale_labels({str(v) for v in sources.values()})


def _scale_from_quant_version(ver: Any) -> str | None:
    if ver == 3:
        return "pack_v3"
    if ver == 2:
        return "pack_v2"
    if ver == 1:
        return "legacy_oracle"
    return None


def _scale_source_from_pack(pack: dict[str, Any] | None) -> str | None:
    if not pack:
        return None
    from_sources = _scale_from_sources_map(pack.get("scale_sources"))
    if from_sources is not None:
        return from_sources
    meta = pack.get("pack_metadata")
    if isinstance(meta, dict):
        return _scale_from_quant_version(meta.get("oz.quantization_version"))
    return None


def _goz1_version_from_pack(pack: dict[str, Any] | None) -> int | None:
    if not pack:
        return None
    versions = pack.get("container_versions")
    if isinstance(versions, list) and versions:
        return _as_int(versions[0])
    meta = pack.get("pack_metadata")
    if isinstance(meta, dict):
        return _as_int(meta.get("oz.quantization_version"))
    return None


def _mean_sparsity(pack: dict[str, Any] | None) -> float | None:
    if not pack:
        return None
    scales = pack.get("ternary_scales")
    if not isinstance(scales, dict) or not scales:
        return None
    vals = [
        s
        for entry in scales.values()
        if isinstance(entry, dict)
        for s in [_as_float(entry.get("sparsity"))]
        if s is not None
    ]
    return sum(vals) / len(vals) if vals else None


def _pack_for_block(chain: dict[str, Any], block: int) -> dict[str, Any] | None:
    packs = chain.get("pack_provenance")
    if not isinstance(packs, list):
        return None
    for pack in packs:
        if isinstance(pack, dict) and pack.get("block") == block:
            return pack
    return None


def _first_present_float(metrics: dict[str, Any], *keys: str) -> float | None:
    """Return the first present key's float value (preserves explicit 0.0)."""
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return _as_float(metrics[key])
    return None


def _row_from_arm_metrics(
    *,
    source_path: str,
    arm: str,
    block: int | None,
    metrics: dict[str, Any],
    tokens: int | None,
    seed: int | None,
    scale_source: str | None,
    goz1_version: int | None,
    sparsity: float | None,
    pack_name: str | None,
    decision: dict[str, Any] | None,
    provenance: dict[str, Any],
) -> ImportedExperimentRow:
    label = metrics.get("label")
    return ImportedExperimentRow(
        schema=SCHEMA_MULTIBLOCK_V1,
        experiment_kind=GozExperimentKind.MULTIBLOCK.value,
        source_path=source_path,
        model_family="grok-1",
        arm=arm,
        block_index=block,
        route_top1_agreement=_as_float(metrics.get("router_top1_agreement")),
        route_top2_agreement=_first_present_float(
            metrics, "router_top2_set_agreement", "router_topk_set_agreement"
        ),
        block_output_cosine=_as_float(metrics.get("block_output_cosine")),
        resid_in_drift=_as_float(metrics.get("residual_drift_relative_norm")),
        expert_load_js=_as_float(metrics.get("expert_load_js_bits")),
        scale_source=scale_source,
        goz1_version=goz1_version,
        sparsity=sparsity,
        tokens=tokens,
        seed=seed,
        label=str(label) if label is not None else arm,
        seconds=_as_float(metrics.get("seconds")),
        pack_basename=str(pack_name) if pack_name else None,
        extra={
            "moe_output_cosine": metrics.get("moe_output_cosine"),
            "residual_stream_cosine": metrics.get("residual_stream_cosine"),
            "block_output_drift_relative_norm": metrics.get(
                "block_output_drift_relative_norm"
            ),
            "decision": (decision or {}).get("decision") if decision else None,
            "experiment_id": provenance.get("issue"),
        },
    )


def _rows_for_block_entry(
    entry: dict[str, Any],
    *,
    chain: dict[str, Any],
    source_path: str,
    arms: tuple[str, ...],
    tokens: int | None,
    seed: int | None,
    decision: dict[str, Any] | None,
    provenance: dict[str, Any],
) -> list[ImportedExperimentRow]:
    block = _as_int(entry.get("block"))
    pack = _pack_for_block(chain, block) if block is not None else None
    pack_name = pack.get("pack") if isinstance(pack, dict) else None
    rows: list[ImportedExperimentRow] = []
    for arm in arms:
        metrics = entry.get(arm)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            _row_from_arm_metrics(
                source_path=source_path,
                arm=arm,
                block=block,
                metrics=metrics,
                tokens=tokens,
                seed=seed,
                scale_source=_scale_source_from_pack(pack),
                goz1_version=_goz1_version_from_pack(pack),
                sparsity=_mean_sparsity(pack),
                pack_name=str(pack_name) if pack_name else None,
                decision=decision,
                provenance=provenance,
            )
        )
    return rows


def import_multiblock_metrics(
    raw: dict[str, Any],
    *,
    source_path: str,
    arms: tuple[str, ...] = ("expert_only", "fp16_control"),
) -> GozImportResult:
    chain = raw.get("chain")
    if not isinstance(chain, dict):
        raise GozImportError("multiblock metrics missing chain object")
    per_block = chain.get("per_block")
    if not isinstance(per_block, list) or not per_block:
        raise GozImportError("multiblock metrics missing chain.per_block")

    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else None
    tokens = _as_int(chain.get("tokens"))
    seed = _as_int(chain.get("token_seed"))
    rows: list[ImportedExperimentRow] = []
    for entry in per_block:
        if isinstance(entry, dict):
            rows.extend(
                _rows_for_block_entry(
                    entry,
                    chain=chain,
                    source_path=source_path,
                    arms=arms,
                    tokens=tokens,
                    seed=seed,
                    decision=decision,
                    provenance=provenance,
                )
            )
    if not rows:
        raise GozImportError(
            f"no rows imported from multiblock metrics (arms={list(arms)})"
        )
    return GozImportResult(
        kind=GozExperimentKind.MULTIBLOCK,
        schema=SCHEMA_MULTIBLOCK_V1,
        source_path=source_path,
        rows=rows,
        provenance=dict(provenance),
        decision=decision,
    )


def _route_scale_source(pilot: dict[str, Any], goz1_version: int | None) -> str | None:
    ternary_scale = str(pilot.get("ternary_scale") or "")
    if "v1" in ternary_scale and "no scale" in ternary_scale:
        return "legacy_oracle"
    return _scale_from_quant_version(goz1_version)


def _summary_status_map(summary: list[Any]) -> dict[str, Any]:
    return {
        str(item.get("name")): item.get("status")
        for item in summary
        if isinstance(item, dict) and item.get("name")
    }


def import_route_preservation(
    raw: dict[str, Any],
    *,
    source_path: str,
) -> GozImportResult:
    pilot = raw.get("pilot")
    summary = raw.get("summary")
    if not isinstance(pilot, dict) or not isinstance(summary, list):
        raise GozImportError("route-preservation report missing pilot/summary")

    observed = _summary_map(summary)
    pack_meta = (
        pilot.get("pack_metadata")
        if isinstance(pilot.get("pack_metadata"), dict)
        else {}
    )
    goz1_version = _as_int(pack_meta.get("oz.quantization_version"))
    mode = str(pilot.get("mode") or "unknown")
    pack_basename = pilot.get("pack_basename")
    row = ImportedExperimentRow(
        schema=SCHEMA_ROUTE_PRESERVATION_V1,
        experiment_kind=GozExperimentKind.ROUTE_PRESERVATION.value,
        source_path=source_path,
        model_family=str(raw.get("model_family") or "grok-1"),
        arm=mode,
        block_index=_as_int(pilot.get("block")),
        route_top1_agreement=observed.get("router_top1_agreement"),
        route_top2_agreement=observed.get("router_top2_set_agreement"),
        block_output_cosine=observed.get("block_output_cosine"),
        resid_in_drift=observed.get("residual_stream_drift"),
        expert_load_js=observed.get("expert_load_js_divergence"),
        scale_source=_route_scale_source(pilot, goz1_version),
        goz1_version=goz1_version,
        sparsity=None,
        tokens=_as_int(pilot.get("tokens")),
        seed=_as_int(pilot.get("seed")),
        label=mode,
        seconds=None,
        pack_basename=str(pack_basename) if pack_basename is not None else None,
        extra={
            "mode": mode,
            "produced_by": raw.get("produced_by"),
            "certification": raw.get("certification"),
            "summary_status": _summary_status_map(summary),
        },
    )
    provenance = {
        "produced_by": raw.get("produced_by"),
        "model_family": raw.get("model_family"),
        "experiment_id": f"route-preservation-block{pilot.get('block')}-{mode}",
    }
    return GozImportResult(
        kind=GozExperimentKind.ROUTE_PRESERVATION,
        schema=SCHEMA_ROUTE_PRESERVATION_V1,
        source_path=source_path,
        rows=[row],
        provenance=provenance,
        decision=None,
    )


def import_goz_experiment(
    path: Path,
    *,
    arms: tuple[str, ...] = ("expert_only", "fp16_control"),
) -> GozImportResult:
    """Load a grok-ozempic experiment JSON and normalize to combine rows."""
    path = Path(path)
    raw = load_experiment_json(path)
    kind = detect_experiment_kind(raw)
    source = str(path)
    if kind is GozExperimentKind.MULTIBLOCK:
        return import_multiblock_metrics(raw, source_path=source, arms=arms)
    return import_route_preservation(raw, source_path=source)


def write_import_reports(
    result: GozImportResult,
    output_dir: Path,
    *,
    formats: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Write JSON/CSV reports under output_dir; return written paths."""
    from benchmarks.reporting import write_csv, write_json

    formats = formats or ["json", "csv"]
    unknown = [f for f in formats if f not in _SUPPORTED_REPORT_FORMATS]
    if unknown:
        raise GozImportError(
            f"unsupported report formats: {unknown}; allowed={sorted(_SUPPORTED_REPORT_FORMATS)}"
        )
    payload = result.to_payload(run_id=run_id)
    rid = payload["run"]["run_id"]
    output_dir = Path(output_dir)
    written: dict[str, Path] = {}
    if "json" in formats:
        jpath = output_dir / "json" / f"{rid}.goz-import.json"
        write_json(jpath, payload)
        written["json"] = jpath
    if "csv" in formats:
        rows = payload["results"]
        if not rows:
            raise GozImportError("no result rows to write")
        cpath = output_dir / "csv" / f"{rid}.goz-import.csv"
        write_csv(cpath, rows)
        written["csv"] = cpath
    if not written:
        raise GozImportError("no report formats selected")
    return written


__all__ = [
    "SCHEMA_MULTIBLOCK_V1",
    "SCHEMA_ROUTE_PRESERVATION_V1",
    "GozExperimentKind",
    "GozImportError",
    "ImportedExperimentRow",
    "GozImportResult",
    "detect_experiment_kind",
    "load_experiment_json",
    "import_multiblock_metrics",
    "import_route_preservation",
    "import_goz_experiment",
    "write_import_reports",
]

