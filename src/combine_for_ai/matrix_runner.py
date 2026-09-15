"""Cross-model experiment matrix runner (RM-105)."""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, assert_never

from benchmarks.jsonio import ensure_dir, write_json
from benchmarks.reporting import write_csv
from benchmarks.runner import run_benchmarks_from_config

from combine_for_ai.matrix import (
    PROGRESS_FILENAME,
    PROGRESS_VERSION,
    CellOutcome,
    CellStatus,
    ComparisonMetrics,
    ExperimentMatrix,
    MatrixCell,
    MatrixError,
    comparison_metrics,
    dataset_payload,
    family_aggregates,
    index_baselines,
    outcome_from_progress,
    outcome_to_progress,
)

SUPPORTED_REPORT_FORMATS = frozenset({"json", "csv", "markdown"})

_CELL_COLUMNS = (
    "cell_id",
    "status",
    "model",
    "family",
    "backend",
    "quantization",
    "dataset",
    "run_id",
    "accuracy",
    "perplexity",
    "throughput",
    "latency_ms",
    "vram_gb",
    "bits",
    "relative_accuracy_drop",
    "compression_ratio",
    "throughput_gain",
    "vram_savings",
    "error",
)

_FAMILY_COLUMNS = (
    "family",
    "cell_count",
    "model_count",
    "mean_accuracy",
    "mean_throughput",
    "mean_vram_gb",
    "mean_relative_accuracy_drop",
    "mean_compression_ratio",
    "mean_throughput_gain",
    "mean_vram_savings",
)


class CellExecutor(Protocol):
    def run_cell(
        self,
        cell: MatrixCell,
        *,
        output_dir: Path,
        formats: list[str],
        seed: int,
    ) -> CellOutcome:
        ...


class BenchmarkCellExecutor:
    """Invoke the existing benchmark runner for one matrix cell."""

    def __init__(self, config_base_path: Path) -> None:
        self._base = Path(config_base_path)

    def run_cell(
        self,
        cell: MatrixCell,
        *,
        output_dir: Path,
        formats: list[str],
        seed: int,
    ) -> CellOutcome:
        config = {
            "run_name": cell.cell_id,
            "seed": seed,
            "model": {
                "backend": cell.model.backend,
                "name": cell.model.name,
                "revision": cell.model.revision,
            },
            "quantization": [cell.quantization],
            "datasets": [dataset_payload(cell.dataset)],
        }
        metadata, results = run_benchmarks_from_config(
            config,
            output_dir,
            formats,
            seed,
            config_base_path=self._base,
        )
        if not results:
            return CellOutcome.failed(cell, "benchmark runner produced no results")
        result = results[0]
        metrics = result.metrics
        return CellOutcome(
            cell=cell,
            status=CellStatus.COMPLETED,
            run_id=metadata.run_id,
            accuracy=metrics.accuracy,
            perplexity=metrics.perplexity,
            throughput=metrics.throughput,
            latency_ms=metrics.latency_ms,
            vram_gb=metrics.vram_gb,
            bits=result.quantization.bits,
        )


@dataclass(frozen=True)
class MatrixReport:
    matrix_name: str
    run_id: str
    baseline_quantization: str
    cells: list[dict[str, Any]]
    by_family: list[dict[str, Any]]
    completed: int
    failed: int
    skipped: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "matrix": {
                "name": self.matrix_name,
                "run_id": self.run_id,
                "baseline_quantization": self.baseline_quantization,
                "cell_count": len(self.cells),
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped,
            },
            "cells": self.cells,
            "by_family": self.by_family,
        }


@dataclass(frozen=True)
class MatrixRunOptions:
    formats: list[str] | None = None
    resume: bool = True
    fail_fast: bool = False
    run_id: str | None = None
    seed: int | None = None


class MatrixRunner:
    """Iterate a model × quant × dataset matrix and aggregate comparison reports."""

    def __init__(
        self,
        matrix: ExperimentMatrix,
        output_dir: Path,
        *,
        executor: CellExecutor | None = None,
        options: MatrixRunOptions | None = None,
    ) -> None:
        settings = options or MatrixRunOptions()
        self.matrix = matrix
        self.output_dir = Path(output_dir)
        self.formats = _validate_formats(settings.formats or ["json", "csv", "markdown"])
        self.resume = settings.resume
        self.fail_fast = settings.fail_fast
        self.run_id = settings.run_id or default_matrix_run_id()
        self.seed = matrix.seed if settings.seed is None else settings.seed
        base = matrix.config_path.parent if matrix.config_path is not None else Path(".")
        self.executor = executor or BenchmarkCellExecutor(base)

    @property
    def progress_path(self) -> Path:
        return self.output_dir / PROGRESS_FILENAME

    def run(self) -> MatrixReport:
        ensure_dir(self.output_dir)
        progress = self._load_progress() if self.resume else {"cells": {}}
        stored_cells = progress.get("cells")
        if not isinstance(stored_cells, dict):
            stored_cells = {}

        outcomes: list[CellOutcome] = []
        for cell in self.matrix.cells:
            outcome = self._run_or_resume(cell, stored_cells)
            outcomes.append(outcome)
            stored_cells[cell.cell_id] = outcome_to_progress(outcome)
            self._write_progress(stored_cells)
            if outcome.status is CellStatus.FAILED and self.fail_fast:
                break

        report = build_matrix_report(
            self.matrix,
            outcomes,
            run_id=self.run_id,
        )
        write_matrix_reports(report, self.output_dir, self.formats)
        return report

    def _run_or_resume(
        self,
        cell: MatrixCell,
        stored_cells: dict[str, Any],
    ) -> CellOutcome:
        stored = stored_cells.get(cell.cell_id)
        previous = _stored_status(stored)
        if not _should_execute(previous, resume=self.resume):
            if stored is None:
                raise MatrixError(f"missing stored progress for cell {cell.cell_id}")
            return outcome_from_progress(cell, stored)

        stored_cells[cell.cell_id] = outcome_to_progress(
            CellOutcome(
                cell=cell,
                status=CellStatus.RUNNING,
                run_id=None,
                accuracy=None,
                perplexity=None,
                throughput=None,
                latency_ms=None,
                vram_gb=None,
                bits=None,
            )
        )
        self._write_progress(stored_cells)

        cell_dir = self.output_dir / "cells" / cell.cell_id
        # Individual cell reports stay JSON/CSV; markdown is matrix-level only.
        cell_formats = [fmt for fmt in self.formats if fmt != "markdown"] or ["json"]
        try:
            return self.executor.run_cell(
                cell,
                output_dir=cell_dir,
                formats=cell_formats,
                seed=self.seed,
            )
        except Exception as exc:
            return CellOutcome.failed(cell, str(exc))

    def _load_progress(self) -> dict[str, Any]:
        path = self.progress_path
        if not path.exists():
            return {"cells": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixError(f"cannot read progress file {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise MatrixError(f"progress file root must be an object: {path}")
        cells = payload.get("cells")
        if cells is None:
            return {"cells": {}}
        if not isinstance(cells, dict):
            raise MatrixError(f"progress file cells must be an object: {path}")
        return {"cells": cells}

    def _write_progress(self, cells: dict[str, Any]) -> None:
        payload = {
            "version": PROGRESS_VERSION,
            "matrix_name": self.matrix.name,
            "matrix_run_id": self.run_id,
            "baseline_quantization": self.matrix.baseline_quantization,
            "cells": cells,
        }
        write_json(self.progress_path, payload)


def default_matrix_run_id() -> str:
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now))
    millis = int(now % 1 * 1000)
    return f"{stamp}.{millis:03d}Z-{secrets.token_hex(4)}-matrix"


def build_matrix_report(
    matrix: ExperimentMatrix,
    outcomes: list[CellOutcome],
    *,
    run_id: str,
) -> MatrixReport:
    baselines = index_baselines(outcomes, matrix.baseline_quantization)
    rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    skipped = 0
    for outcome in outcomes:
        match outcome.status:
            case CellStatus.COMPLETED:
                completed += 1
            case CellStatus.FAILED:
                failed += 1
            case CellStatus.SKIPPED:
                skipped += 1
            case CellStatus.PENDING | CellStatus.RUNNING:
                pass
            case _:
                assert_never(outcome.status)
        baseline = baselines.get((outcome.cell.model.name, outcome.cell.dataset.name))
        compared = comparison_metrics(outcome, baseline)
        rows.append(_cell_row(outcome, compared))

    return MatrixReport(
        matrix_name=matrix.name,
        run_id=run_id,
        baseline_quantization=matrix.baseline_quantization,
        cells=rows,
        by_family=family_aggregates(
            rows, baseline_quantization=matrix.baseline_quantization
        ),
        completed=completed,
        failed=failed,
        skipped=skipped,
    )


def write_matrix_reports(
    report: MatrixReport,
    output_dir: Path,
    formats: list[str],
) -> dict[str, Path]:
    formats = _validate_formats(formats)
    written: dict[str, Path] = {}
    _write_json_report(report, output_dir, formats, written)
    _write_csv_reports(report, output_dir, formats, written)
    _write_markdown_report(report, output_dir, formats, written)
    if not written:
        raise MatrixError("no report formats selected")
    return written


def _write_json_report(
    report: MatrixReport,
    output_dir: Path,
    formats: list[str],
    written: dict[str, Path],
) -> None:
    if "json" not in formats:
        return
    path = output_dir / "json" / f"{report.run_id}.matrix.json"
    write_json(path, report.to_payload())
    written["json"] = path


def _write_csv_reports(
    report: MatrixReport,
    output_dir: Path,
    formats: list[str],
    written: dict[str, Path],
) -> None:
    if "csv" not in formats:
        return
    if not report.cells:
        raise MatrixError("no matrix cells to write to csv")
    path = output_dir / "csv" / f"{report.run_id}.matrix.csv"
    write_csv(path, [_ordered_row(row, _CELL_COLUMNS) for row in report.cells])
    written["csv"] = path
    if not report.by_family:
        return
    family_path = output_dir / "csv" / f"{report.run_id}.matrix-family.csv"
    write_csv(
        family_path,
        [_ordered_row(row, _FAMILY_COLUMNS) for row in report.by_family],
    )
    written["csv_family"] = family_path


def _write_markdown_report(
    report: MatrixReport,
    output_dir: Path,
    formats: list[str],
    written: dict[str, Path],
) -> None:
    if "markdown" not in formats:
        return
    path = output_dir / "markdown" / f"{report.run_id}.matrix.md"
    ensure_dir(path.parent)
    path.write_text(_render_markdown(report), encoding="utf-8")
    written["markdown"] = path


def _should_execute(status: CellStatus | None, *, resume: bool) -> bool:
    if status is None:
        return True
    match status:
        case CellStatus.COMPLETED:
            return not resume
        case CellStatus.SKIPPED:
            return True
        case CellStatus.PENDING | CellStatus.RUNNING | CellStatus.FAILED:
            return True
        case _:
            assert_never(status)


def _stored_status(stored: Any) -> CellStatus | None:
    if not isinstance(stored, dict):
        return None
    raw = stored.get("status")
    if raw is None:
        return None
    try:
        return CellStatus(str(raw))
    except ValueError as exc:
        raise MatrixError(f"invalid progress status {raw!r}") from exc


def _validate_formats(formats: list[str]) -> list[str]:
    if not formats:
        raise MatrixError("no report formats selected")
    unknown = [fmt for fmt in formats if fmt not in SUPPORTED_REPORT_FORMATS]
    if unknown:
        raise MatrixError(
            f"unsupported report formats: {unknown}; "
            f"allowed={sorted(SUPPORTED_REPORT_FORMATS)}"
        )
    return formats


def _cell_row(outcome: CellOutcome, compared: ComparisonMetrics) -> dict[str, Any]:
    cell = outcome.cell
    metrics = compared.to_dict()
    return {
        "cell_id": cell.cell_id,
        "status": outcome.status.value,
        "model": cell.model.name,
        "family": cell.model.family,
        "backend": cell.model.backend,
        "quantization": cell.quantization,
        "dataset": cell.dataset.name,
        "run_id": outcome.run_id,
        "accuracy": outcome.accuracy,
        "perplexity": outcome.perplexity,
        "throughput": outcome.throughput,
        "latency_ms": outcome.latency_ms,
        "vram_gb": outcome.vram_gb,
        "bits": outcome.bits,
        **metrics,
        "error": outcome.error,
    }


def _ordered_row(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    return {key: row.get(key) for key in columns}


def _render_markdown(report: MatrixReport) -> str:
    lines = _markdown_header(report)
    lines.extend(_markdown_cell_table(report.cells))
    lines.extend(_markdown_family_table(report.by_family))
    return "\n".join(lines)


def _markdown_header(report: MatrixReport) -> list[str]:
    return [
        f"# Matrix comparison: {report.matrix_name}",
        "",
        f"- run_id: `{report.run_id}`",
        f"- baseline: `{report.baseline_quantization}`",
        f"- cells: {len(report.cells)} "
        f"(completed={report.completed}, failed={report.failed}, skipped={report.skipped})",
        "",
    ]


def _markdown_cell_table(cells: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Side-by-side cells",
        "",
        "| model | family | quant | dataset | acc | rel_drop | compress | tput_gain | vram_save |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    keys = (
        "model",
        "family",
        "quantization",
        "dataset",
        "accuracy",
        "relative_accuracy_drop",
        "compression_ratio",
        "throughput_gain",
        "vram_savings",
    )
    for row in cells:
        lines.append(_markdown_row(row, keys))
    return lines


def _markdown_family_table(families: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Family aggregates", ""]
    if not families:
        lines.extend(["No completed cells to aggregate.", ""])
        return lines
    lines.extend(
        [
            "| family | cells | models | mean_acc | mean_drop | mean_compress | mean_tput_gain | mean_vram_save |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    keys = (
        "family",
        "cell_count",
        "model_count",
        "mean_accuracy",
        "mean_relative_accuracy_drop",
        "mean_compression_ratio",
        "mean_throughput_gain",
        "mean_vram_savings",
    )
    for row in families:
        lines.append(_markdown_row(row, keys))
    lines.append("")
    return lines


def _markdown_row(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    values: list[str] = []
    identity = {"model", "family", "quantization", "dataset", "cell_count", "model_count"}
    for key in keys:
        value = row.get(key)
        if key in identity:
            values.append(str(value))
        else:
            values.append(_md_num(value))
    return "| " + " | ".join(values) + " |"


def _md_num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
