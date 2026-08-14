from __future__ import annotations

import json
from pathlib import Path

import pytest

from combine_for_ai.compare import (
    CompareError,
    build_comparison,
    load_experiment_rows,
    write_comparison_reports,
)
from combine_for_ai.goz_import import import_goz_experiment, write_import_reports


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_load_from_raw_experiment() -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    rows = load_experiment_rows([path])
    assert len(rows) == 4  # 2 blocks x 2 arms
    arms = {r["arm"] for r in rows}
    assert arms == {"expert_only", "fp16_control"}


def test_load_from_goz_import_report(tmp_path: Path) -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    imported = import_goz_experiment(path)
    write_import_reports(imported, tmp_path, run_id="cmp-src", formats=["json"])
    report = tmp_path / "json" / "cmp-src.goz-import.json"
    rows = load_experiment_rows([report])
    assert len(rows) == 4


def test_build_comparison_deltas() -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    rows = load_experiment_rows([path])
    result = build_comparison(rows)
    assert "expert_only" in result.arms
    assert "fp16_control" in result.arms
    assert len(result.by_block) == 2

    b0 = next(r for r in result.by_block if r["block_index"] == 0)
    assert b0["baseline_route_top1_agreement"] == pytest.approx(0.999)
    assert b0["treatment_route_top1_agreement"] == pytest.approx(1.0)
    assert b0["delta_route_top1_agreement"] == pytest.approx(1.0 - 0.999)
    assert b0["treatment_resid_in_drift"] == pytest.approx(0.0)
    assert b0["baseline_block_output_cosine"] == pytest.approx(0.9999)

    b1 = next(r for r in result.by_block if r["block_index"] == 1)
    assert b1["treatment_resid_in_drift"] == pytest.approx(0.277)
    assert b1["delta_resid_in_drift"] == pytest.approx(0.277 - 0.0)


def test_write_comparison_reports(tmp_path: Path) -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    rows = load_experiment_rows([path])
    result = build_comparison(rows)
    written = write_comparison_reports(
        result, tmp_path, run_id="cmp-test", formats=["json", "csv", "markdown"]
    )
    assert written["json"].is_file()
    assert written["csv"].is_file()
    assert written["markdown"].is_file()
    payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert len(payload["by_block"]) == 2
    md = written["markdown"].read_text(encoding="utf-8")
    assert "Hybrid quant comparison" in md
    assert "Δtop1" in md


def test_empty_inputs_fail() -> None:
    with pytest.raises(CompareError, match="at least one"):
        load_experiment_rows([])


def test_cli_main(tmp_path: Path) -> None:
    from scripts.compare_runs import main

    src = FIXTURES / "goz_multiblock_metrics.sample.json"
    rc = main(
        [
            "--input",
            str(src),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "cli-cmp",
        ]
    )
    assert rc == 0
    assert (tmp_path / "json" / "cli-cmp.compare.json").is_file()
    assert (tmp_path / "markdown" / "cli-cmp.compare.md").is_file()
