"""Public archived-report contract: no fabricated values or unsafe comparisons."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from combine_for_ai.plot_reports import generate_reports

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = Path(__file__).parents[1] / "configs" / "plots.sample.json"


def archive(tmp_path, records):
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps({"schema_version": "combine.plot-evidence.v1", "records": records})
    )
    return path


def fixture_records():
    return json.loads((FIXTURES / "plot_evidence.sample.json").read_text())["records"]


def report(path, tmp_path, *, plots=False):
    written = generate_reports([path], tmp_path / "reports", CONFIG, plots=plots)
    return written, json.loads(written["json"].read_text())


def test_prepared_values_intervals_counts_and_source_hash_survive(tmp_path):
    source = FIXTURES / "plot_evidence.sample.json"
    written, payload = report(source, tmp_path)
    assert (
        payload["sources"][0]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert payload["sources"][0]["run"]["url"] == "https://example.org/runs/fixture"
    assert payload["plot_config"]["schema_version"] == "combine.plot-config.v1"
    language = next(c for c in payload["cohorts"] if c["workload_track"] == "language")
    accuracy = next(m for m in language["metrics"] if m["name"] == "accuracy")
    assert accuracy["observations"][0]["value"] == 0.75
    assert accuracy["observations"][0]["interval"]["low"] == 0.31
    assert accuracy["observations"][0]["interval"]["high"] == 0.91
    assert accuracy["observations"][0]["counts"] == {"ok": 3, "failed": 1}
    text = written["markdown"].read_text()
    for expected in (
        "fixture",
        "mock",
        "failed",
        "unsupported",
        "0.31",
        "0.91",
        "paired seed",
        "scipy",
        "MiB",
        "calculated-proxy",
    ):
        assert expected in text
    assert not list((tmp_path / "reports").rglob("*.png"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("dataset", "other@v1"),
        ("scorer", "other@v1"),
        ("config", {"batch_size": 8}),
        ("environment", {"hardware": "different"}),
        ("seeds", [13, 17]),
    ],
)
def test_incompatible_contexts_are_separate_not_averaged(tmp_path, key, value):
    a = fixture_records()[0]
    b = copy.deepcopy(a)
    b["series"] = "model-b"
    b["context"][key] = value
    b["metrics"][0]["value"] = 0.5
    _, payload = report(archive(tmp_path, [a, b]), tmp_path)
    assert len(payload["cohorts"]) == 2
    values = [c["metrics"][0]["observations"][0]["value"] for c in payload["cohorts"]]
    assert sorted(values) == [0.5, 0.75]


def test_complete_compatible_context_can_share_a_plot_but_not_an_average(tmp_path):
    a = fixture_records()[0]
    b = copy.deepcopy(a)
    b["series"] = "model-b"
    b["metrics"][0]["value"] = 0.5
    _, payload = report(archive(tmp_path, [a, b]), tmp_path)
    assert len(payload["cohorts"]) == 1
    assert [
        o["value"] for o in payload["cohorts"][0]["metrics"][0]["observations"]
    ] == [0.75, 0.5]


def test_incomplete_context_and_distinct_sources_never_merge(tmp_path):
    a = fixture_records()[0]
    a["context"].pop("sample_ids")
    b = copy.deepcopy(a)
    b["series"] = "model-b"
    _, payload = report(archive(tmp_path, [a, b]), tmp_path)
    assert len(payload["cohorts"]) == 2
    assert all(
        c["comparison_policy"] == "isolated: incomplete context"
        for c in payload["cohorts"]
    )
    other = tmp_path / "other.json"
    other.write_bytes((tmp_path / "archive.json").read_bytes())
    result = generate_reports(
        [tmp_path / "archive.json", other], tmp_path / "two", CONFIG, plots=False
    )
    assert len(json.loads(result["json"].read_text())["cohorts"]) == 4


def test_empty_partial_and_failed_rows_remain_visible(tmp_path):
    written, payload = report(archive(tmp_path, []), tmp_path)
    assert payload["cohorts"] == []
    assert "No observations" in written["markdown"].read_text()
    record = fixture_records()[0]
    record["metrics"] = [
        {
            "name": "error",
            "unit": "fraction",
            "definition": "Task error",
            "scope": "task",
            "status": "failed",
        },
        {
            "name": "latency",
            "unit": "ms",
            "definition": "Request time",
            "scope": "request",
            "status": "missing",
        },
        {
            "name": "zero",
            "unit": "count",
            "definition": "Actual zero",
            "scope": "task",
            "status": "ok",
            "value": 0,
        },
    ]
    written, payload = report(archive(tmp_path, [record]), tmp_path)
    observations = [m["observations"][0] for m in payload["cohorts"][0]["metrics"]]
    assert [o["value"] for o in observations] == [None, None, 0]
    assert [o["status"] for o in observations] == ["failed", "missing", "ok"]
    assert "unavailable" in written["markdown"].read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"execution_kind": "mock", "metric_provenance": "measured"},
        {"execution_kind": "metadata-only", "metric_provenance": "measured"},
        {
            "metrics": [
                {
                    "name": "latency",
                    "unit": "ms",
                    "definition": "Request time",
                    "scope": "request",
                    "status": "unsupported",
                    "value": 10,
                }
            ]
        },
    ],
)
def test_invalid_evidence_fails_closed_before_writing(tmp_path, change):
    record = fixture_records()[0]
    record.update(change)
    with pytest.raises(ValueError):
        report(archive(tmp_path, [record]), tmp_path)
    assert not (tmp_path / "reports").exists()


def test_missing_unknown_or_malformed_input_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="missing.json"):
        report(tmp_path / "missing.json", tmp_path)
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": "future", "records": []}')
    with pytest.raises(ValueError, match="schema"):
        report(path, tmp_path)
    path.write_text('{"schema_version": "combine.plot-evidence.v1", "records": [null]}')
    with pytest.raises(ValueError):
        report(path, tmp_path)
    path.write_text(
        json.dumps(
            {
                "combine_import_schema": 2,
                "run": {"schema": "grok_ozempic.multiblock_metrics.v1"},
                "results": [],
            }
        )
    )
    with pytest.raises(ValueError, match="version"):
        report(path, tmp_path)


def test_legacy_goz1_raw_and_import_reports_keep_route_and_residual_views(tmp_path):
    from combine_for_ai.goz_import import import_goz_experiment, write_import_reports

    source = FIXTURES / "goz_multiblock_metrics.sample.json"
    imported = import_goz_experiment(source)
    legacy = write_import_reports(imported, tmp_path / "legacy", run_id="legacy")[
        "json"
    ]
    for path in (source, legacy):
        written, payload = report(path, tmp_path)
        cohort = payload["cohorts"][0]
        assert cohort["input_provenance"] == "fixture"
        assert cohort["execution_kind"] == "import"
        assert cohort["comparison_policy"] == "legacy observations: unqualified"
        route = next(
            m for m in cohort["metrics"] if m["name"] == "route_top1_agreement"
        )
        drift = next(m for m in cohort["metrics"] if m["name"] == "resid_in_drift")
        assert [o["value"] for o in route["observations"]] == [
            1.0,
            0.999,
            0.887695,
            0.999,
        ]
        assert [o["value"] for o in drift["observations"]] == [0, 0, 0.277, 0]
        assert "unqualified" in written["markdown"].read_text()


def test_legacy_route_threshold_failure_does_not_erase_observation(tmp_path):
    written, payload = report(FIXTURES / "goz_route_preservation.sample.json", tmp_path)
    route = next(
        m
        for m in payload["cohorts"][0]["metrics"]
        if m["name"] == "route_top1_agreement"
    )
    assert route["observations"][0]["value"] == 0.6396484375
    assert "fail" in route["observations"][0]["notes"]
    assert "fail" in written["markdown"].read_text()
    divergence = next(
        m for m in payload["cohorts"][0]["metrics"] if m["name"] == "expert_load_js"
    )
    assert divergence["unit"] == "unknown"  # route schema does not declare a log base


def test_legacy_normalized_rows_from_different_origins_stay_separate(tmp_path):
    from combine_for_ai.goz_import import import_goz_experiment

    imported = import_goz_experiment(FIXTURES / "goz_multiblock_metrics.sample.json")
    raw = imported.to_payload(run_id="mixed")
    raw["results"][1]["source_path"] = "another-experiment.json"
    source = tmp_path / "mixed.json"
    source.write_text(json.dumps(raw))
    _, payload = report(source, tmp_path)
    assert len(payload["cohorts"]) == 2
    assert {c["context"]["source_path"] for c in payload["cohorts"]} == {
        str(FIXTURES / "goz_multiblock_metrics.sample.json"),
        "another-experiment.json",
    }


def test_legacy_explicit_v1_and_missing_arms_are_not_dropped(tmp_path):
    raw = json.loads((FIXTURES / "goz_multiblock_metrics.sample.json").read_text())
    raw["schema_version"] = 1
    del raw["chain"]["per_block"][1]["expert_only"]
    raw["chain"]["blocks"].append(2)  # declared block never produced results
    source = tmp_path / "partial.json"
    source.write_text(json.dumps(raw))
    written, payload = report(source, tmp_path)
    route = next(
        m
        for m in payload["cohorts"][0]["metrics"]
        if m["name"] == "route_top1_agreement"
    )
    by_arm_block = {(o["series"], o["block_index"]): o for o in route["observations"]}
    assert len(by_arm_block) == 6
    assert by_arm_block[("expert_only", 0)]["value"] == 1
    for arm, block in (("expert_only", 1), ("expert_only", 2), ("fp16_control", 2)):
        assert by_arm_block[(arm, block)]["value"] is None
        assert by_arm_block[(arm, block)]["status"] == "missing"
    assert "missing=3" in written["markdown"].read_text()


def test_empty_legacy_and_regeneration_cannot_leave_stale_csv(tmp_path):
    source = tmp_path / "legacy.json"
    raw = json.loads((FIXTURES / "goz_multiblock_metrics.sample.json").read_text())
    source.write_text(json.dumps(raw))
    written, _ = report(source, tmp_path)
    assert written["csv"].exists()
    raw["chain"]["per_block"] = []
    raw["chain"]["blocks"] = []
    source.write_text(json.dumps(raw))
    written, payload = report(source, tmp_path)
    assert payload["cohorts"] == []
    assert "No observations" in written["markdown"].read_text()
    assert not (tmp_path / "reports" / "csv" / "plot-report.csv").exists()


def test_csv_preserves_source_hash_and_raw_references(tmp_path):
    import csv

    record = fixture_records()[0]
    record["source_artifacts"] = [{"path": "raw.json", "sha256": "a" * 64}]
    source = archive(tmp_path, [record])
    written, payload = report(source, tmp_path)
    row = next(csv.DictReader(written["csv"].open()))
    assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(row["source_artifacts"]) == [
        {"path": "raw.json", "sha256": "a" * 64}
    ]
    assert (
        payload["cohorts"][0]["metrics"][0]["observations"][0]["source_artifacts"][0][
            "path"
        ]
        == "raw.json"
    )


def test_malformed_legacy_blocks_are_rejected_not_silently_removed(tmp_path):
    raw = json.loads((FIXTURES / "goz_multiblock_metrics.sample.json").read_text())
    raw["chain"]["per_block"].append(None)
    source = tmp_path / "malformed.json"
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="per_block"):
        report(source, tmp_path)


def test_repeated_input_is_rejected_not_counted_as_independent_evidence(tmp_path):
    source = archive(tmp_path, fixture_records())
    with pytest.raises(ValueError, match="duplicate"):
        generate_reports(
            [source, source.parent / "." / source.name],
            tmp_path / "reports",
            CONFIG,
            plots=False,
        )


def test_failed_legacy_arm_never_becomes_successful_observation(tmp_path):
    raw = json.loads((FIXTURES / "goz_multiblock_metrics.sample.json").read_text())
    raw["chain"]["per_block"][1]["expert_only"]["status"] = "failed"
    raw["chain"]["per_block"][1]["fp16_control"]["status"] = "ok"
    del raw["chain"]["per_block"][1]["fp16_control"]["expert_load_js_bits"]
    source = tmp_path / "failed.json"
    source.write_text(json.dumps(raw))
    written, payload = report(source, tmp_path)
    route = next(
        m
        for m in payload["cohorts"][0]["metrics"]
        if m["name"] == "route_top1_agreement"
    )
    failed = next(
        o
        for o in route["observations"]
        if o["series"] == "expert_only" and o["block_index"] == 1
    )
    assert failed["status"] == "failed"
    assert failed["value"] is None
    assert "failed=1" in written["markdown"].read_text()
    load = next(
        m for m in payload["cohorts"][0]["metrics"] if m["name"] == "expert_load_js"
    )
    missing = next(
        o
        for o in load["observations"]
        if o["series"] == "fp16_control" and o["block_index"] == 1
    )
    assert missing["status"] == "missing"
    assert missing["value"] is None


def test_headless_plots_and_regeneration_are_deterministic(tmp_path):
    pytest.importorskip("matplotlib")
    source = FIXTURES / "plot_evidence.sample.json"
    written, payload = report(source, tmp_path, plots=True)
    pngs = list((tmp_path / "reports" / "plots").glob("*.png"))
    assert len(pngs) == 6  # memory/energy unavailable, never plotted at zero
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in pngs)
    before = {p: p.read_bytes() for p in [*pngs, *written.values()]}
    report(source, tmp_path, plots=True)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert "plots/" in written["markdown"].read_text()
    assert (
        payload["plot_config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    )


def test_prepared_interval_outside_estimate_preserves_confidence_precision(tmp_path):
    from importlib.util import find_spec

    record = fixture_records()[0]
    record["metrics"] = record["metrics"][:1]
    record["metrics"][0]["interval"].update(low=0.2, high=0.3, confidence_level=0.955)
    written, payload = report(
        archive(tmp_path, [record]), tmp_path, plots=find_spec("matplotlib") is not None
    )
    assert payload["cohorts"][0]["metrics"][0]["observations"][0]["value"] == 0.75
    assert "95.5%" in written["markdown"].read_text()
