"""Every JSON report must be readable by a conforming parser.

``NaN`` and ``Infinity`` are not part of JSON. Python's ``json.dump`` emits them
as bare literals by default, which strict parsers reject, so a single degenerate
metric upstream can make a whole report unreadable to its consumers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.jsonio import json_safe, write_json


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _strict_load(path: Path) -> object:
    """Parse like a conforming JSON parser: bare NaN/Infinity are an error."""

    def reject(constant: str) -> object:
        raise ValueError(f"not valid JSON: bare {constant}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_metric_is_written_as_null(tmp_path: Path, value: float) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"results": [{"block_output_cosine": value}]})

    payload = _strict_load(path)
    assert payload == {"results": [{"block_output_cosine": None}]}


def test_non_finite_values_are_replaced_at_every_depth(tmp_path: Path) -> None:
    path = tmp_path / "nested.json"
    write_json(
        path,
        {
            "run": {"telemetry": {"vram_bandwidth_gbps": float("inf")}},
            "results": [
                {"metrics": {"cosine": float("nan"), "drift": 0.5}},
                {"metrics": {"cosine": 1.0, "history": [float("nan"), 2.0]}},
            ],
        },
    )

    payload = _strict_load(path)
    assert payload["run"]["telemetry"]["vram_bandwidth_gbps"] is None
    assert payload["results"][0]["metrics"]["cosine"] is None
    assert payload["results"][0]["metrics"]["drift"] == 0.5
    assert payload["results"][1]["metrics"]["history"] == [None, 2.0]


def test_finite_values_are_untouched(tmp_path: Path) -> None:
    original = {
        "run_id": "abc",
        "count": 3,
        "ratio": 0.125,
        "flag": True,
        "missing": None,
        "names": ["a", "b"],
    }
    path = tmp_path / "plain.json"
    write_json(path, original)
    assert _strict_load(path) == original


def test_json_safe_leaves_non_float_values_alone() -> None:
    assert json_safe({"a": [1, "b", None, True]}) == {"a": [1, "b", None, True]}
    assert json_safe(float("nan")) is None
    assert json_safe(0.0) == 0.0


def test_import_report_of_a_nan_metric_is_valid_json(tmp_path: Path) -> None:
    """End-to-end: a NaN cosine upstream must not corrupt the import report."""
    from combine_for_ai.goz_import import import_goz_experiment, write_import_reports

    payload = json.loads(
        (FIXTURES / "goz_multiblock_metrics.sample.json").read_text(encoding="utf-8")
    )
    payload["chain"]["per_block"][0]["expert_only"]["block_output_cosine"] = float("nan")
    source = tmp_path / "nan-metrics.json"
    # json.dumps writes the NaN as a bare literal, matching a real upstream file.
    source.write_text(json.dumps(payload), encoding="utf-8")

    written = write_import_reports(
        import_goz_experiment(source), tmp_path, run_id="nan-import", formats=["json"]
    )
    report = _strict_load(written["json"])
    cosines = [row["block_output_cosine"] for row in report["results"]]
    assert None in cosines, "the NaN cosine should have been written as null"


def test_benchmark_report_of_a_nan_metric_is_valid_json(tmp_path: Path) -> None:
    """End-to-end: the benchmark writer goes through the same guard."""
    from benchmarks.reporting import write_json as writer  # re-exported

    writer(
        tmp_path / "json" / "run.json",
        {"run": {"run_id": "r"}, "results": [{"perplexity": float("inf")}]},
    )
    report = _strict_load(tmp_path / "json" / "run.json")
    assert report["results"][0]["perplexity"] is None


def test_reporting_still_re_exports_the_writers() -> None:
    """`benchmarks.reporting` is the import site three modules already use."""
    from benchmarks import jsonio, reporting

    assert reporting.write_json is jsonio.write_json
    assert reporting.json_safe is jsonio.json_safe


def test_standalone_telemetry_artifact_is_valid_json(tmp_path: Path) -> None:
    """A benchmark run writes this file too, and it bypassed the guard.

    Hardware probes legitimately return a non-finite reading, and
    `write_telemetry_json` called `json.dump` directly, so `<run>.telemetry.json`
    could be invalid even when the main report was clean.
    """
    from benchmarks.telemetry import (
        RoutingMetrics,
        SystemSnapshot,
        TelemetrySnapshot,
        write_telemetry_json,
    )

    snapshot = TelemetrySnapshot(
        system=SystemSnapshot(
            cpu_count_logical=8,
            cpu_count_physical=4,
            memory_total_gb=float("nan"),
            memory_available_gb=16.0,
            gpu_count=0,
            gpu_names=None,
            gpu_driver_version=None,
            cuda_version=None,
            platform="linux",
            python_version="3.14",
        ),
        routing=RoutingMetrics(routing_entropy=float("inf")),
        kernel_occupancy=float("-inf"),
        vram_bandwidth_gbps=float("nan"),
    )

    path = tmp_path / "telemetry" / "run.telemetry.json"
    write_telemetry_json(path, snapshot)

    payload = _strict_load(path)
    assert payload["sys_memory_total_gb"] is None
    assert payload["sys_memory_available_gb"] == 16.0
    assert payload["routing_entropy"] is None
    assert payload["kernel_occupancy"] is None
    assert payload["vram_bandwidth_gbps"] is None


def test_benchmark_run_writes_no_invalid_json_anywhere(tmp_path: Path) -> None:
    """Every JSON file a run produces must parse strictly, telemetry included."""
    from benchmarks.runner import run_benchmarks

    config = Path("configs/benchmark.sample.json")
    run_benchmarks(config, tmp_path, ["json", "csv"], 42)

    written = sorted(tmp_path.rglob("*.json"))
    assert written, "the run produced no JSON at all"
    for path in written:
        _strict_load(path)  # raises if the file contains a bare constant
