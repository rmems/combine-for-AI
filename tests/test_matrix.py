"""Cross-model experiment matrix orchestration (RM-105)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.datasets import DatasetSpec
from combine_for_ai.matrix import (
    CORINTH_CANAL_FAMILIES,
    CellOutcome,
    CellStatus,
    MatrixCell,
    MatrixError,
    MatrixModelSpec,
    MatrixSelection,
    comparison_metrics,
    family_aggregates,
    load_experiment_matrix,
    load_matrix_config,
    relative_drop,
    ratio,
)
from combine_for_ai.matrix_runner import MatrixRunner, build_matrix_report
from scripts.run_matrix import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSON = REPO_ROOT / "configs" / "matrix" / "corinth_canal.sample.json"
SAMPLE_TOML = REPO_ROOT / "configs" / "matrix" / "corinth_canal.sample.toml"


class RecordingExecutor:
    """Mock adapter stand-in: records cells and returns deterministic metrics."""

    def __init__(self, fail_on: frozenset[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on or frozenset()

    def run_cell(self, cell: MatrixCell, **kwargs: object) -> CellOutcome:
        self.calls.append(cell.cell_id)
        if cell.cell_id in self.fail_on and self.calls.count(cell.cell_id) == 1:
            raise RuntimeError(f"injected failure for {cell.cell_id}")
        bits = {"fp16": 16, "awq": 4, "gptq": 4, "saaq": 2, "ternary": 2}[cell.quantization]
        accuracy = {
            "fp16": 0.90,
            "awq": 0.85,
            "gptq": 0.84,
            "saaq": 0.81,
            "ternary": 0.80,
        }[cell.quantization]
        throughput = {
            "fp16": 100.0,
            "awq": 150.0,
            "gptq": 140.0,
            "saaq": 200.0,
            "ternary": 180.0,
        }[cell.quantization]
        vram = {
            "fp16": 14.0,
            "awq": 10.5,
            "gptq": 11.0,
            "saaq": 5.5,
            "ternary": 6.0,
        }[cell.quantization]
        return CellOutcome(
            cell=cell,
            status=CellStatus.COMPLETED,
            run_id=f"run-{cell.cell_id}",
            accuracy=accuracy,
            perplexity=10.0,
            throughput=throughput,
            latency_ms=5.0,
            vram_gb=vram,
            bits=bits,
        )


def _write_jsonl(path: Path) -> None:
    path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )


def _mini_config(
    tmp_path: Path,
    *,
    models: list[dict[str, object]] | None = None,
    quantization: list[str] | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    dataset = tmp_path / "data.jsonl"
    _write_jsonl(dataset)
    payload: dict[str, object] = {
        "matrix_version": "1.0.0",
        "matrix_name": "mini",
        "seed": 7,
        "baseline_quantization": "fp16",
        "quantization": quantization or ["fp16", "awq"],
        "models": models
        or [
            {"name": "olmoe-1b-7b", "family": "olmoe", "backend": "mock"},
            {"name": "zaya-1", "family": "zaya", "backend": "mock"},
        ],
        "datasets": [
            {
                "name": "lambada",
                "source": "jsonl",
                "path": str(dataset),
                "split": "validation",
                "max_samples": 1,
            }
        ],
    }
    if extra:
        payload.update(extra)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_sample_json_expands_to_22_corinth_canal_cells() -> None:
    matrix = load_experiment_matrix(SAMPLE_JSON)
    assert len(matrix.cells) == 22
    families = {cell.model.family for cell in matrix.cells}
    assert families == set(CORINTH_CANAL_FAMILIES)
    models = {cell.model.name for cell in matrix.cells}
    assert len(models) == 6
    assert matrix.baseline_quantization == "fp16"


def test_sample_toml_matches_json_lineup() -> None:
    json_matrix = load_experiment_matrix(SAMPLE_JSON)
    toml_matrix = load_experiment_matrix(SAMPLE_TOML)
    json_ids = {cell.cell_id for cell in json_matrix.cells}
    toml_ids = {cell.cell_id for cell in toml_matrix.cells}
    assert json_ids == toml_ids
    assert len(toml_ids) == 22


def test_cartesian_product_models_quants_datasets(tmp_path: Path) -> None:
    dataset_a = tmp_path / "a.jsonl"
    dataset_b = tmp_path / "b.jsonl"
    _write_jsonl(dataset_a)
    _write_jsonl(dataset_b)
    config_path = tmp_path / "matrix.json"
    payload = {
        "matrix_name": "product",
        "models": [
            {"name": "m1", "family": "olmoe"},
            {"name": "m2", "family": "zaya"},
        ],
        "quantization": ["fp16", "saaq"],
        "datasets": [
            {"name": "lambada", "source": "jsonl", "path": str(dataset_a)},
            {"name": "piqa", "source": "jsonl", "path": str(dataset_b)},
        ],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    matrix = load_experiment_matrix(config_path)
    assert len(matrix.cells) == 8
    combos = {
        (cell.model.name, cell.quantization, cell.dataset.name) for cell in matrix.cells
    }
    assert ("m1", "fp16", "lambada") in combos
    assert ("m2", "saaq", "piqa") in combos


def test_per_model_and_quant_filtering(tmp_path: Path) -> None:
    path = _mini_config(tmp_path)
    matrix = load_experiment_matrix(
        path,
        extra_select=MatrixSelection(
            models=frozenset({"olmoe-1b-7b"}),
            quantization=frozenset({"awq"}),
        ),
    )
    assert len(matrix.cells) == 1
    assert matrix.cells[0].model.name == "olmoe-1b-7b"
    assert matrix.cells[0].quantization == "awq"


def test_family_filter_and_exclude(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        extra={
            "select": {"families": ["olmoe"]},
            "exclude": [{"quantization": "awq"}],
        },
    )
    matrix = load_experiment_matrix(path)
    assert {cell.model.family for cell in matrix.cells} == {"olmoe"}
    assert {cell.quantization for cell in matrix.cells} == {"fp16"}


def test_empty_selection_raises(tmp_path: Path) -> None:
    path = _mini_config(tmp_path)
    with pytest.raises(MatrixError, match="matched no cells"):
        load_experiment_matrix(
            path, extra_select=MatrixSelection(models=frozenset({"missing"}))
        )


def test_unknown_quantization_raises(tmp_path: Path) -> None:
    path = _mini_config(tmp_path, quantization=["fp16", "not-a-real-quant"])
    with pytest.raises(MatrixError, match="unknown quantization"):
        load_experiment_matrix(path)


def test_comparison_metric_formulas() -> None:
    assert relative_drop(0.90, 0.81) == pytest.approx(0.1)
    assert ratio(16.0, 4.0) == pytest.approx(4.0)
    assert relative_drop(14.0, 10.5) == pytest.approx((14.0 - 10.5) / 14.0)
    assert ratio(150.0, 100.0) == pytest.approx(1.5)
    assert relative_drop(0.0, 0.1) is None
    assert ratio(1.0, 0.0) is None


def test_comparison_metrics_against_baseline(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        models=[{"name": "olmoe-1b-7b", "family": "olmoe"}],
        quantization=["fp16", "awq"],
    )
    matrix = load_experiment_matrix(path)
    report = MatrixRunner(
        matrix,
        tmp_path / "out",
        executor=RecordingExecutor(),
        formats=["json"],
        resume=False,
        run_id="cmp",
    ).run()
    by_quant = {row["quantization"]: row for row in report.cells}
    fp16 = by_quant["fp16"]
    awq = by_quant["awq"]
    assert fp16["relative_accuracy_drop"] == 0.0
    assert fp16["compression_ratio"] == 1.0
    assert fp16["throughput_gain"] == 1.0
    assert fp16["vram_savings"] == 0.0
    assert awq["relative_accuracy_drop"] == pytest.approx((0.90 - 0.85) / 0.90)
    assert awq["compression_ratio"] == pytest.approx(16 / 4)
    assert awq["throughput_gain"] == pytest.approx(150 / 100)
    assert awq["vram_savings"] == pytest.approx((14.0 - 10.5) / 14.0)


def test_family_level_aggregation(tmp_path: Path) -> None:
    path = _mini_config(tmp_path)
    matrix = load_experiment_matrix(path)
    report = MatrixRunner(
        matrix,
        tmp_path / "out",
        executor=RecordingExecutor(),
        formats=["json"],
        resume=False,
        run_id="fam",
    ).run()
    families = {row["family"]: row for row in report.by_family}
    assert set(families) == {"olmoe", "zaya"}
    assert families["olmoe"]["cell_count"] == 2
    assert families["olmoe"]["model_count"] == 1
    assert families["olmoe"]["mean_relative_accuracy_drop"] == pytest.approx(
        (0.90 - 0.85) / 0.90
    )


def test_resume_skips_completed_cells(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        models=[{"name": "olmoe-1b-7b", "family": "olmoe"}],
        quantization=["fp16", "awq"],
    )
    matrix = load_experiment_matrix(path)
    fail_id = [cell.cell_id for cell in matrix.cells if cell.quantization == "awq"][0]
    output = tmp_path / "out"
    first = RecordingExecutor(fail_on=frozenset({fail_id}))
    report1 = MatrixRunner(
        matrix,
        output,
        executor=first,
        formats=["json"],
        resume=True,
        run_id="r1",
    ).run()
    assert report1.failed == 1
    assert report1.completed == 1
    assert len(first.calls) == 2

    second = RecordingExecutor()
    report2 = MatrixRunner(
        matrix,
        output,
        executor=second,
        formats=["json"],
        resume=True,
        run_id="r2",
    ).run()
    assert report2.completed == 2
    assert report2.failed == 0
    assert second.calls == [fail_id]


def test_fresh_reruns_completed_cells(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        models=[{"name": "olmoe-1b-7b", "family": "olmoe"}],
        quantization=["fp16"],
    )
    matrix = load_experiment_matrix(path)
    output = tmp_path / "out"
    MatrixRunner(
        matrix,
        output,
        executor=RecordingExecutor(),
        formats=["json"],
        resume=False,
        run_id="a",
    ).run()
    rerun = RecordingExecutor()
    MatrixRunner(
        matrix,
        output,
        executor=rerun,
        formats=["json"],
        resume=False,
        run_id="b",
    ).run()
    assert len(rerun.calls) == 1


def test_reports_include_individual_and_aggregate(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        models=[{"name": "olmoe-1b-7b", "family": "olmoe"}],
        quantization=["fp16", "saaq"],
    )
    matrix = load_experiment_matrix(path)
    output = tmp_path / "out"
    report = MatrixRunner(
        matrix,
        output,
        executor=RecordingExecutor(),
        formats=["json", "csv", "markdown"],
        resume=False,
        run_id="side-by-side",
    ).run()
    json_path = output / "json" / "side-by-side.matrix.json"
    csv_path = output / "csv" / "side-by-side.matrix.csv"
    family_csv = output / "csv" / "side-by-side.matrix-family.csv"
    md_path = output / "markdown" / "side-by-side.matrix.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert family_csv.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["matrix"]["cell_count"] == 2
    assert {row["quantization"] for row in payload["cells"]} == {"fp16", "saaq"}
    markdown = md_path.read_text(encoding="utf-8")
    assert "Side-by-side cells" in markdown
    assert "Family aggregates" in markdown
    assert "olmoe" in markdown
    assert report.completed == 2


def test_end_to_end_with_mock_model_adapter(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        models=[{"name": "toy-olmoe", "family": "olmoe", "backend": "mock"}],
        quantization=["fp16", "saaq"],
    )
    matrix = load_experiment_matrix(path)
    output = tmp_path / "out"
    report = MatrixRunner(
        matrix,
        output,
        formats=["json", "csv"],
        resume=False,
        run_id="mock-e2e",
        seed=42,
    ).run()
    assert report.completed == 2
    assert report.failed == 0
    cell_dirs = list((output / "cells").iterdir())
    assert len(cell_dirs) == 2
    json_reports = list((output / "cells").glob("*/json/*.json"))
    assert len(json_reports) == 2
    saaq = next(row for row in report.cells if row["quantization"] == "saaq")
    assert saaq["relative_accuracy_drop"] is not None
    assert saaq["compression_ratio"] == pytest.approx(16 / 2)


def test_sample_matrix_orchestrates_all_families(tmp_path: Path) -> None:
    matrix = load_experiment_matrix(SAMPLE_JSON)
    report = MatrixRunner(
        matrix,
        tmp_path / "out",
        executor=RecordingExecutor(),
        formats=["json", "markdown"],
        resume=False,
        run_id="sprint",
    ).run()
    assert report.completed == 22
    assert report.failed == 0
    assert {row["family"] for row in report.by_family} == set(CORINTH_CANAL_FAMILIES)
    assert (tmp_path / "out" / "matrix-progress.json").exists()


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0


def test_cli_filters_and_runs(tmp_path: Path) -> None:
    path = _mini_config(tmp_path)
    output = tmp_path / "out"
    code = main(
        [
            "--config",
            str(path),
            "--output-dir",
            str(output),
            "--models",
            "olmoe-1b-7b",
            "--quant-methods",
            "fp16",
            "--formats",
            "json",
            "--run-id",
            "cli-filter",
            "--fresh",
        ]
    )
    assert code == 0
    payload = json.loads((output / "json" / "cli-filter.matrix.json").read_text())
    assert payload["matrix"]["cell_count"] == 1
    assert payload["cells"][0]["model"] == "olmoe-1b-7b"


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text("matrix_name: x\n", encoding="utf-8")
    with pytest.raises(MatrixError, match="unsupported matrix config suffix"):
        load_matrix_config(path)


def test_comparison_without_baseline_is_null() -> None:
    cell = MatrixCell(
        model=MatrixModelSpec(
            name="m",
            family="olmoe",
            backend="mock",
            revision="local",
            quantization=("saaq",),
        ),
        quantization="saaq",
        dataset=DatasetSpec(name="lambada", source="jsonl"),
    )
    treatment = CellOutcome(
        cell=cell,
        status=CellStatus.COMPLETED,
        run_id="t",
        accuracy=0.8,
        perplexity=1.0,
        throughput=200.0,
        latency_ms=1.0,
        vram_gb=5.5,
        bits=2,
    )
    metrics = comparison_metrics(treatment, None)
    assert metrics.relative_accuracy_drop is None
    assert metrics.compression_ratio is None


def test_family_aggregates_ignore_failed_rows() -> None:
    rows = [
        {
            "family": "olmoe",
            "model": "a",
            "status": "completed",
            "quantization": "saaq",
            "accuracy": 0.8,
            "throughput": 10.0,
            "vram_gb": 5.0,
            "relative_accuracy_drop": 0.1,
            "compression_ratio": 4.0,
            "throughput_gain": 1.5,
            "vram_savings": 0.2,
        },
        {
            "family": "olmoe",
            "model": "a",
            "status": "failed",
            "quantization": "awq",
            "accuracy": None,
        },
    ]
    grouped = family_aggregates(rows, baseline_quantization="fp16")
    assert len(grouped) == 1
    assert grouped[0]["cell_count"] == 1
    assert grouped[0]["mean_relative_accuracy_drop"] == pytest.approx(0.1)


def test_build_matrix_report_counts_failed_cells(tmp_path: Path) -> None:
    path = _mini_config(
        tmp_path,
        models=[{"name": "olmoe-1b-7b", "family": "olmoe"}],
        quantization=["fp16"],
    )
    matrix = load_experiment_matrix(path)
    failed = CellOutcome.failed(matrix.cells[0], "boom")
    report = build_matrix_report(matrix, [failed], run_id="fail")
    assert report.failed == 1
    assert report.completed == 0
    assert report.cells[0]["status"] == "failed"
