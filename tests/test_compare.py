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

    b0 = next((r for r in result.by_block if r["block_index"] == 0), None)
    assert b0 is not None, "block_index=0 not found in result.by_block"
    assert b0["baseline_route_top1_agreement"] == pytest.approx(0.999)
    assert b0["treatment_route_top1_agreement"] == pytest.approx(1.0)
    assert b0["delta_route_top1_agreement"] == pytest.approx(1.0 - 0.999)
    assert b0["treatment_resid_in_drift"] == pytest.approx(0.0)
    assert b0["baseline_block_output_cosine"] == pytest.approx(0.9999)

    b1 = next((r for r in result.by_block if r["block_index"] == 1), None)
    assert b1 is not None, "block_index=1 not found in result.by_block"
    assert b1["treatment_resid_in_drift"] == pytest.approx(0.277)
    assert b1["delta_resid_in_drift"] == pytest.approx(0.277 - 0.0)


def test_duplicate_block_arm_raises() -> None:
    rows = [
        {"arm": "fp16_control", "block_index": 0, "route_top1_agreement": 1.0},
        {"arm": "fp16_control", "block_index": 0, "route_top1_agreement": 0.9},
        {"arm": "expert_only", "block_index": 0, "route_top1_agreement": 0.8},
    ]
    with pytest.raises(CompareError, match="duplicate row"):
        build_comparison(rows)


def test_missing_arm_raises() -> None:
    rows = [{"arm": "expert_only", "block_index": 0, "route_top1_agreement": 0.8}]
    with pytest.raises(CompareError, match="both arms"):
        build_comparison(rows)


def test_incomplete_block_pairs_are_skipped() -> None:
    rows = [
        {"arm": "fp16_control", "block_index": 0},
        {"arm": "expert_only", "block_index": 1},
        {"arm": "fp16_control", "block_index": 2, "route_top1_agreement": 1.0},
        {"arm": "expert_only", "block_index": 2, "route_top1_agreement": 0.9},
    ]
    result = build_comparison(rows)
    assert [row["block_index"] for row in result.by_block] == [2]


def test_pairs_without_comparable_metrics_raise() -> None:
    rows = [
        {"arm": "fp16_control", "block_index": 0, "label": "baseline"},
        {"arm": "expert_only", "block_index": 0, "label": "treatment"},
    ]
    with pytest.raises(CompareError, match="no baseline/treatment block pairs"):
        build_comparison(rows)


def test_incompatible_pairing_context_raises() -> None:
    rows = [
        {"arm": "fp16_control", "block_index": 0, "tokens": 128, "seed": 1},
        {"arm": "expert_only", "block_index": 0, "tokens": 256, "seed": 1},
    ]
    with pytest.raises(CompareError, match="incompatible comparison context"):
        build_comparison(rows)


def test_identical_comparison_arms_raise() -> None:
    with pytest.raises(CompareError, match="must be different"):
        build_comparison([], baseline_arm="fp16_control", treatment_arm="fp16_control")


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
    assert "Baseline arm: fp16_control" in md
    assert "Treatment arm: expert_only" in md
    assert "d_top1" in md
    assert "top2_base" in md
    assert "0.9990" in md


def test_reports_validate_all_formats_before_writing(tmp_path: Path) -> None:
    rows = load_experiment_rows([FIXTURES / "goz_multiblock_metrics.sample.json"])
    result = build_comparison(rows)
    with pytest.raises(CompareError, match="unsupported report format"):
        write_comparison_reports(result, tmp_path, run_id="partial", formats=["json", "bogus"])
    assert not (tmp_path / "json" / "partial.compare.json").exists()


def test_comparison_csv_escapes_formula_metadata(tmp_path: Path) -> None:
    rows = [
        {"arm": "fp16_control", "block_index": 0, "label": "=evil()", "route_top1_agreement": 1.0},
        {"arm": "expert_only", "block_index": 0, "label": "+evil()", "route_top1_agreement": 0.9},
    ]
    result = build_comparison(rows)
    written = write_comparison_reports(result, tmp_path, run_id="csv-safe", formats=["csv"])
    csv_text = written["csv"].read_text(encoding="utf-8")
    assert "'=evil()" in csv_text
    assert "'+evil()" in csv_text


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


def test_cli_loads_custom_selected_arms(tmp_path: Path) -> None:
    from scripts.compare_runs import main

    payload = json.loads((FIXTURES / "goz_multiblock_metrics.sample.json").read_text())
    for block in payload["chain"]["per_block"]:
        block["z_control"] = block.pop("fp16_control")
        block["a_quant"] = block.pop("expert_only")
    source = tmp_path / "custom-arms.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    rc = main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "custom-arms",
            "--baseline-arm",
            "z_control",
            "--treatment-arm",
            "a_quant",
        ]
    )
    assert rc == 0
    markdown = (tmp_path / "markdown" / "custom-arms.compare.md").read_text()
    assert "Baseline arm: z_control" in markdown
    assert "Treatment arm: a_quant" in markdown
