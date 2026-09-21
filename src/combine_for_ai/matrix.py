"""Experiment matrix config, cell expansion, comparison metrics, and progress.

Orchestrates models × quantization methods × datasets without coupling to a
quantization backend. The runner (see ``matrix_runner``) invokes the existing
benchmark harness for each cell.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from benchmarks.datasets import DatasetSpec
from benchmarks.models import QuantizationRegistry, default_quantization_registry


MATRIX_VERSION = "1.0.0"
PROGRESS_VERSION = 1
PROGRESS_FILENAME = "matrix-progress.json"

# Families from corinth-canal's Vultr sprint lineup (RM-105).
CORINTH_CANAL_FAMILIES = (
    "olmoe",
    "qwen3moe",
    "gemma4",
    "deepseek2",
    "llamamoe",
    "zaya",
)

_CELL_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class MatrixError(ValueError):
    """Raised when a matrix config, selection, or progress file is invalid."""


class CellStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MatrixSelect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[str] | None = None
    families: list[str] | None = None
    quantization: list[str] | None = None
    datasets: list[str] | None = None


class MatrixExclude(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    family: str | None = None
    quantization: str | None = None
    dataset: str | None = None

    @model_validator(mode="after")
    def _at_least_one_constraint(self) -> MatrixExclude:
        if not any((self.model, self.family, self.quantization, self.dataset)):
            raise ValueError("exclude entry must set at least one of model/family/quantization/dataset")
        return self


class MatrixModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    family: str
    backend: str = "mock"
    revision: str | None = "local"
    quantization: list[str] | None = None


class MatrixFileConfig(BaseModel):
    """On-disk matrix definition (JSON or TOML)."""

    model_config = ConfigDict(extra="forbid")

    matrix_version: str = MATRIX_VERSION
    matrix_name: str
    comment: str | None = None
    seed: int = 0
    baseline_quantization: str = "fp16"
    models: list[MatrixModelConfig]
    quantization: list[str] = Field(default_factory=lambda: ["fp16"])
    datasets: list[dict[str, Any]]
    select: MatrixSelect | None = None
    exclude: list[MatrixExclude] = Field(default_factory=list)

    @model_validator(mode="after")
    def _non_empty(self) -> MatrixFileConfig:
        if not self.models:
            raise ValueError("matrix config must list at least one model")
        if not self.datasets:
            raise ValueError("matrix config must list at least one dataset")
        return self


@dataclass(frozen=True)
class MatrixModelSpec:
    name: str
    family: str
    backend: str
    revision: str | None
    quantization: tuple[str, ...]


@dataclass(frozen=True)
class MatrixCell:
    model: MatrixModelSpec
    quantization: str
    dataset: DatasetSpec

    @property
    def cell_id(self) -> str:
        return "__".join(
            (
                slugify(self.model.name),
                slugify(self.quantization),
                slugify(self.dataset.name),
            )
        )


@dataclass(frozen=True)
class MatrixSelection:
    models: frozenset[str] | None = None
    families: frozenset[str] | None = None
    quantization: frozenset[str] | None = None
    datasets: frozenset[str] | None = None

    def allows(self, cell: MatrixCell) -> bool:
        pairs = (
            (self.models, cell.model.name),
            (self.families, cell.model.family),
            (self.quantization, cell.quantization),
            (self.datasets, cell.dataset.name),
        )
        return all(_set_allows(allowed, value) for allowed, value in pairs)


@dataclass(frozen=True)
class ExperimentMatrix:
    name: str
    seed: int
    baseline_quantization: str
    cells: tuple[MatrixCell, ...]
    config_path: Path | None
    select: MatrixSelection
    exclude: tuple[MatrixExclude, ...]


@dataclass(frozen=True)
class ComparisonMetrics:
    relative_accuracy_drop: float | None
    compression_ratio: float | None
    throughput_gain: float | None
    vram_savings: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "relative_accuracy_drop": self.relative_accuracy_drop,
            "compression_ratio": self.compression_ratio,
            "throughput_gain": self.throughput_gain,
            "vram_savings": self.vram_savings,
        }


@dataclass(frozen=True)
class CellOutcome:
    cell: MatrixCell
    status: CellStatus
    run_id: str | None
    accuracy: float | None
    perplexity: float | None
    throughput: float | None
    latency_ms: float | None
    vram_gb: float | None
    bits: int | None
    error: str | None = None

    @staticmethod
    def failed(cell: MatrixCell, error: str) -> CellOutcome:
        return CellOutcome(
            cell=cell,
            status=CellStatus.FAILED,
            run_id=None,
            accuracy=None,
            perplexity=None,
            throughput=None,
            latency_ms=None,
            vram_gb=None,
            bits=None,
            error=error,
        )


def slugify(value: str) -> str:
    slug = _CELL_ID_SAFE.sub("-", value.strip()).strip(".-")
    if not slug:
        raise MatrixError(f"cannot derive a filesystem-safe id from {value!r}")
    return slug


def load_matrix_config(path: Path) -> MatrixFileConfig:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MatrixError(f"cannot read matrix config {path}: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            raw = tomllib.loads(text)
        elif suffix == ".json":
            raw = json.loads(text)
        else:
            raise MatrixError(
                f"unsupported matrix config suffix {suffix!r}; use .json or .toml"
            )
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise MatrixError(f"invalid matrix config {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise MatrixError(f"matrix config root must be an object: {path}")
    try:
        return MatrixFileConfig.model_validate(raw)
    except ValidationError as exc:
        raise MatrixError(f"invalid matrix config {path}: {exc}") from exc


def selection_from_config(select: MatrixSelect | None) -> MatrixSelection:
    if select is None:
        return MatrixSelection()
    return MatrixSelection(
        models=_optional_set(select.models),
        families=_optional_set(select.families),
        quantization=_optional_set(select.quantization),
        datasets=_optional_set(select.datasets),
    )


def merge_selections(*selections: MatrixSelection) -> MatrixSelection:
    merged = MatrixSelection()
    for selection in selections:
        merged = MatrixSelection(
            models=_intersect(merged.models, selection.models),
            families=_intersect(merged.families, selection.families),
            quantization=_intersect(merged.quantization, selection.quantization),
            datasets=_intersect(merged.datasets, selection.datasets),
        )
    return merged


def expand_matrix(
    config: MatrixFileConfig,
    *,
    config_path: Path | None = None,
    extra_select: MatrixSelection | None = None,
) -> ExperimentMatrix:
    registry = default_quantization_registry()
    default_quants = _validate_quants(config.quantization, registry)
    _require_known_quant(config.baseline_quantization, registry)
    base_path = config_path.parent if config_path is not None else Path(".")
    select = merge_selections(
        selection_from_config(config.select),
        extra_select or MatrixSelection(),
    )
    selected = _apply_selection(
        _cartesian_cells(config, base_path, registry, default_quants),
        select,
        config.exclude,
    )
    return ExperimentMatrix(
        name=config.matrix_name,
        seed=config.seed,
        baseline_quantization=config.baseline_quantization,
        cells=selected,
        config_path=config_path,
        select=select,
        exclude=tuple(config.exclude),
    )


def _require_known_quant(name: str, registry: QuantizationRegistry) -> None:
    if name not in set(registry.supported_names()):
        raise MatrixError(f"unknown baseline quantization '{name}'")


def _cartesian_cells(
    config: MatrixFileConfig,
    base_path: Path,
    registry: QuantizationRegistry,
    default_quants: tuple[str, ...],
) -> list[MatrixCell]:
    datasets = tuple(_resolve_dataset(raw, base_path) for raw in config.datasets)
    cells: list[MatrixCell] = []
    for model_cfg in config.models:
        cells.extend(_cells_for_model(model_cfg, datasets, registry, default_quants))
    return cells


def _cells_for_model(
    model_cfg: MatrixModelConfig,
    datasets: tuple[DatasetSpec, ...],
    registry: QuantizationRegistry,
    default_quants: tuple[str, ...],
) -> list[MatrixCell]:
    quants = (
        _validate_quants(model_cfg.quantization, registry)
        if model_cfg.quantization is not None
        else default_quants
    )
    model = MatrixModelSpec(
        name=model_cfg.name,
        family=model_cfg.family,
        backend=model_cfg.backend,
        revision=model_cfg.revision,
        quantization=quants,
    )
    return [
        MatrixCell(model=model, quantization=quant, dataset=dataset)
        for quant in quants
        for dataset in datasets
    ]


def _apply_selection(
    cells: list[MatrixCell],
    select: MatrixSelection,
    exclude: list[MatrixExclude],
) -> tuple[MatrixCell, ...]:
    selected = tuple(cell for cell in cells if _keep_cell(cell, select, exclude))
    if not selected:
        raise MatrixError("matrix selection matched no cells")
    cell_ids = [cell.cell_id for cell in selected]
    if len(cell_ids) != len(set(cell_ids)):
        raise MatrixError("duplicate cell ids in expanded matrix")
    return selected


def _keep_cell(
    cell: MatrixCell, select: MatrixSelection, exclude: list[MatrixExclude]
) -> bool:
    if not select.allows(cell):
        return False
    return not _excluded(cell, exclude)


def load_experiment_matrix(
    path: Path,
    *,
    extra_select: MatrixSelection | None = None,
) -> ExperimentMatrix:
    path = Path(path)
    return expand_matrix(
        load_matrix_config(path),
        config_path=path,
        extra_select=extra_select,
    )


def relative_drop(baseline: float | None, treatment: float | None) -> float | None:
    """(baseline - treatment) / baseline, or None when undefined."""
    if baseline is None or treatment is None or baseline == 0:
        return None
    return (baseline - treatment) / baseline


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def comparison_metrics(
    treatment: CellOutcome,
    baseline: CellOutcome | None,
) -> ComparisonMetrics:
    """Compare a cell against the same model+dataset at the baseline quantization."""
    if baseline is None or baseline.status is not CellStatus.COMPLETED:
        return ComparisonMetrics(None, None, None, None)
    if treatment.status is not CellStatus.COMPLETED:
        return ComparisonMetrics(None, None, None, None)
    if treatment.cell.quantization == baseline.cell.quantization:
        return ComparisonMetrics(
            relative_accuracy_drop=0.0,
            compression_ratio=1.0,
            throughput_gain=1.0,
            vram_savings=0.0,
        )
    return ComparisonMetrics(
        relative_accuracy_drop=relative_drop(baseline.accuracy, treatment.accuracy),
        compression_ratio=ratio(_as_float(baseline.bits), _as_float(treatment.bits)),
        throughput_gain=ratio(treatment.throughput, baseline.throughput),
        vram_savings=relative_drop(baseline.vram_gb, treatment.vram_gb),
    )


def index_baselines(
    outcomes: Iterable[CellOutcome],
    baseline_quantization: str,
) -> dict[tuple[str, str], CellOutcome]:
    indexed: dict[tuple[str, str], CellOutcome] = {}
    for outcome in outcomes:
        if outcome.status is not CellStatus.COMPLETED:
            continue
        if outcome.cell.quantization != baseline_quantization:
            continue
        key = (outcome.cell.model.name, outcome.cell.dataset.name)
        indexed[key] = outcome
    return indexed


def family_aggregates(
    rows: list[dict[str, Any]],
    *,
    baseline_quantization: str,
) -> list[dict[str, Any]]:
    groups = _group_completed_rows(rows)
    return [
        _family_summary(family, items, baseline_quantization)
        for family, items in sorted(groups.items())
    ]


def _group_completed_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != CellStatus.COMPLETED.value:
            continue
        family = str(row.get("family") or "unknown")
        groups.setdefault(family, []).append(row)
    return groups


def _family_summary(
    family: str,
    items: list[dict[str, Any]],
    baseline_quantization: str,
) -> dict[str, Any]:
    compared = _non_baseline_rows(items, baseline_quantization)
    return {
        "family": family,
        "cell_count": len(items),
        "model_count": len({row["model"] for row in items}),
        "mean_accuracy": _column_mean(items, "accuracy"),
        "mean_throughput": _column_mean(items, "throughput"),
        "mean_vram_gb": _column_mean(items, "vram_gb"),
        "mean_relative_accuracy_drop": _column_mean(compared, "relative_accuracy_drop"),
        "mean_compression_ratio": _column_mean(compared, "compression_ratio"),
        "mean_throughput_gain": _column_mean(compared, "throughput_gain"),
        "mean_vram_savings": _column_mean(compared, "vram_savings"),
    }


def _non_baseline_rows(
    items: list[dict[str, Any]], baseline_quantization: str
) -> list[dict[str, Any]]:
    return [row for row in items if row.get("quantization") != baseline_quantization]


def _column_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return _mean(row.get(key) for row in rows)


def outcome_to_progress(outcome: CellOutcome) -> dict[str, Any]:
    return {
        "cell_id": outcome.cell.cell_id,
        "status": outcome.status.value,
        "run_id": outcome.run_id,
        "accuracy": outcome.accuracy,
        "perplexity": outcome.perplexity,
        "throughput": outcome.throughput,
        "latency_ms": outcome.latency_ms,
        "vram_gb": outcome.vram_gb,
        "bits": outcome.bits,
        "error": outcome.error,
    }


def outcome_from_progress(cell: MatrixCell, stored: dict[str, Any]) -> CellOutcome:
    status = _parse_status(stored.get("status"))
    return CellOutcome(
        cell=cell,
        status=status,
        run_id=_optional_str(stored.get("run_id")),
        accuracy=_as_float(stored.get("accuracy")),
        perplexity=_as_float(stored.get("perplexity")),
        throughput=_as_float(stored.get("throughput")),
        latency_ms=_as_float(stored.get("latency_ms")),
        vram_gb=_as_float(stored.get("vram_gb")),
        bits=_as_int(stored.get("bits")),
        error=_optional_str(stored.get("error")),
    )


def dataset_payload(spec: DatasetSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": spec.name,
        "split": spec.split,
        "source": spec.source,
    }
    if spec.path is not None:
        payload["path"] = spec.path
    if spec.hf_id is not None:
        payload["hf_id"] = spec.hf_id
    if spec.hf_subset is not None:
        payload["hf_subset"] = spec.hf_subset
    if spec.max_samples is not None:
        payload["max_samples"] = spec.max_samples
    return payload


def _set_allows(allowed: frozenset[str] | None, value: str) -> bool:
    return allowed is None or value in allowed


def _constraint_matches(expected: str | None, actual: str) -> bool:
    return expected is None or expected == actual


def _optional_set(values: list[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(values)


def _intersect(
    left: frozenset[str] | None, right: frozenset[str] | None
) -> frozenset[str] | None:
    if left is None:
        return right
    if right is None:
        return left
    return left & right


def _validate_quants(names: list[str], registry: QuantizationRegistry) -> tuple[str, ...]:
    if not names:
        raise MatrixError("quantization list must not be empty")
    validated: list[str] = []
    known = set(registry.supported_names())
    seen: set[str] = set()
    for name in names:
        if name not in known:
            raise MatrixError(f"unknown quantization '{name}'")
        if name in seen:
            continue
        seen.add(name)
        validated.append(name)
    return tuple(validated)


def _resolve_dataset(raw: dict[str, Any], base_path: Path) -> DatasetSpec:
    spec = DatasetSpec.from_dict(raw)
    if spec.path:
        path = Path(spec.path)
        if not path.is_absolute():
            spec = replace(spec, path=str((base_path / path).resolve()))
    return spec


def _excluded(cell: MatrixCell, rules: list[MatrixExclude]) -> bool:
    return any(_exclude_matches(rule, cell) for rule in rules)


def _exclude_matches(rule: MatrixExclude, cell: MatrixCell) -> bool:
    return all(
        (
            _constraint_matches(rule.model, cell.model.name),
            _constraint_matches(rule.family, cell.model.family),
            _constraint_matches(rule.quantization, cell.quantization),
            _constraint_matches(rule.dataset, cell.dataset.name),
        )
    )


def _parse_status(raw: Any) -> CellStatus:
    try:
        return CellStatus(str(raw))
    except ValueError as exc:
        raise MatrixError(f"invalid cell status {raw!r}") from exc


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)
