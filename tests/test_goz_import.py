from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nfl_combine_for_ai.goz_import import (
    SCHEMA_MULTIBLOCK_V1,
    SCHEMA_ROUTE_PRESERVATION_V1,
    GozExperimentKind,
    GozImportError,
    detect_experiment_kind,
    import_goz_experiment,
    write_import_reports,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_detect_multiblock_kind() -> None:
    raw = json.loads((FIXTURES / "goz_multiblock_metrics.sample.json").read_text())
    assert detect_experiment_kind(raw) is GozExperimentKind.MULTIBLOCK


def test_detect_route_preservation_kind() -> None:
    raw = json.loads((FIXTURES / "goz_route_preservation.sample.json").read_text())
    assert detect_experiment_kind(raw) is GozExperimentKind.ROUTE_PRESERVATION


def test_import_multiblock_rows() -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    result = import_goz_experiment(path)
    assert result.kind is GozExperimentKind.MULTIBLOCK
    assert result.schema == SCHEMA_MULTIBLOCK_V1
    # 2 blocks x 2 arms
    assert len(result.rows) == 4

    b0_expert = next(
        r for r in result.rows if r.block_index == 0 and r.arm == "expert_only"
    )
    assert b0_expert.route_top1_agreement == pytest.approx(1.0)
    assert b0_expert.route_top2_agreement == pytest.approx(1.0)
    assert b0_expert.block_output_cosine == pytest.approx(0.963572)
    assert b0_expert.resid_in_drift == pytest.approx(0.0)
    assert b0_expert.expert_load_js == pytest.approx(0.0)
    assert b0_expert.scale_source == "pack_v2"
    assert b0_expert.goz1_version == 3
    assert b0_expert.sparsity == pytest.approx(0.65)
    assert b0_expert.tokens == 128
    assert b0_expert.seed == 20260806
    assert b0_expert.pack_basename == "block_000-attention_plus_expert.goz1"

    b1_expert = next(
        r for r in result.rows if r.block_index == 1 and r.arm == "expert_only"
    )
    assert b1_expert.route_top1_agreement == pytest.approx(0.887695)
    assert b1_expert.resid_in_drift == pytest.approx(0.277)


def test_import_multiblock_expert_only_four_blocks() -> None:
    """Compatibility fixture shaped like #68 expert-only chain (4 blocks)."""
    path = FIXTURES / "goz_multiblock_expert_only_4block.sample.json"
    result = import_goz_experiment(path, arms=("expert_only",))
    assert len(result.rows) == 4
    by_block = {r.block_index: r for r in result.rows}
    assert by_block[0].route_top1_agreement == pytest.approx(1.0)
    assert by_block[0].resid_in_drift == pytest.approx(0.0)
    assert by_block[3].route_top1_agreement == pytest.approx(0.52832)
    assert by_block[3].resid_in_drift == pytest.approx(0.498)
    assert by_block[3].block_output_cosine == pytest.approx(0.839144)
    assert result.decision is not None
    assert result.decision.get("decision") == 3


def test_preserve_zero_top2_agreement() -> None:
    raw = {
        "chain": {
            "tokens": 8,
            "token_seed": 1,
            "per_block": [
                {
                    "block": 0,
                    "expert_only": {
                        "router_top1_agreement": 0.5,
                        "router_top2_set_agreement": 0.0,
                        "router_topk_set_agreement": 0.9,
                        "block_output_cosine": 0.8,
                        "residual_drift_relative_norm": 0.1,
                    },
                }
            ],
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "z.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        row = import_goz_experiment(path, arms=("expert_only",)).rows[0]
        assert row.route_top2_agreement == pytest.approx(0.0)


def test_reject_unsupported_schema_version() -> None:
    raw = {
        "combine_import_schema": "v99",
        "chain": {"per_block": [{"block": 0, "expert_only": {}}]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad-ver.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(GozImportError, match="unsupported combine_import_schema"):
            import_goz_experiment(path)


def test_reject_unsupported_report_format() -> None:
    path = FIXTURES / "goz_route_preservation.sample.json"
    result = import_goz_experiment(path)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(GozImportError, match="unsupported report formats"):
            write_import_reports(result, Path(tmp), formats=["xml"])


def test_import_multiblock_arm_filter() -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    result = import_goz_experiment(path, arms=("expert_only",))
    assert len(result.rows) == 2
    assert all(r.arm == "expert_only" for r in result.rows)


def test_import_route_preservation() -> None:
    path = FIXTURES / "goz_route_preservation.sample.json"
    result = import_goz_experiment(path)
    assert result.kind is GozExperimentKind.ROUTE_PRESERVATION
    assert result.schema == SCHEMA_ROUTE_PRESERVATION_V1
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.block_index == 0
    assert row.arm == "attention_only"
    assert row.route_top1_agreement == pytest.approx(0.6396484375)
    assert row.route_top2_agreement == pytest.approx(0.421142578125)
    assert row.block_output_cosine == pytest.approx(0.8492034016757434)
    assert row.resid_in_drift == pytest.approx(0.5341102824225851)
    assert row.expert_load_js == pytest.approx(0.11021330045422195)
    assert row.scale_source == "legacy_oracle"
    assert row.goz1_version == 1
    assert row.tokens == 256


def test_unknown_shape_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        with pytest.raises(GozImportError, match="unknown"):
            import_goz_experiment(path)


def test_write_import_reports() -> None:
    path = FIXTURES / "goz_multiblock_metrics.sample.json"
    result = import_goz_experiment(path, arms=("expert_only",))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        written = write_import_reports(
            result, out, formats=["json", "csv"], run_id="test-run-1"
        )
        assert written["json"].is_file()
        assert written["csv"].is_file()
        payload = json.loads(written["json"].read_text(encoding="utf-8"))
        assert payload["run"]["run_id"] == "test-run-1"
        assert payload["benchmark_linkage"]["grok_ozempic_report_path"]
        assert len(payload["results"]) == 2
        assert payload["results"][0]["route_top1_agreement"] == pytest.approx(1.0)
        csv_text = written["csv"].read_text(encoding="utf-8")
        assert "route_top1_agreement" in csv_text
        assert "expert_only" in csv_text


def test_cli_main(tmp_path: Path) -> None:
    from scripts.import_goz_experiment import main

    src = FIXTURES / "goz_route_preservation.sample.json"
    out = tmp_path / "reports"
    rc = main(
        [
            "--input",
            str(src),
            "--output-dir",
            str(out),
            "--run-id",
            "cli-test",
        ]
    )
    assert rc == 0
    assert (out / "json" / "cli-test.goz-import.json").is_file()
    assert (out / "csv" / "cli-test.goz-import.csv").is_file()
