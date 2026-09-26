from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.cli import parse_args
from benchmarks.corinth_canal import (
    LATENT_CSV_HEADER,
    classify_json_artifact,
    load_corinth_canal,
    parse_latent_telemetry_csv,
    parse_run_manifest,
    parse_summary,
    parse_tick_telemetry,
    try_load_corinth_canal,
)
from benchmarks.metrics import MetricsAccumulator, SaaqMetricOverlay
from benchmarks.runner import run_benchmarks
from benchmarks.telemetry import (
    CorinthCanalArtifact,
    SystemSnapshot,
    TelemetrySnapshot,
    merge_upstream_artifacts,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUN_DIR = FIXTURES / "corinth_canal"
CANONICAL_LATENT_CSV = "latent_telemetry.csv"


def test_dual_saaq_csv_header_matches_corinth_canal() -> None:
    csv_path = RUN_DIR / "latent_telemetry.csv"
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == LATENT_CSV_HEADER


def test_parse_latent_telemetry_csv_dual_trajectories() -> None:
    series = parse_latent_telemetry_csv(RUN_DIR / "latent_telemetry.csv")
    assert series.row_count == 4
    assert series.firing_rate_mean == pytest.approx(7.3875)
    assert series.membrane_dv_dt_mean == pytest.approx(0.4375)
    assert series.membrane_pressure_mean == pytest.approx(0.03645833333333333)
    assert series.routing_entropy_mean == pytest.approx(0.8175)
    assert series.delta_q_mean == pytest.approx(0.235891)
    assert series.delta_q_last == pytest.approx(0.296199)
    assert series.delta_q_legacy_mean == pytest.approx(0.1160645)
    assert series.delta_q_v15_mean == pytest.approx(0.235891)
    assert series.delta_q_trajectory == (0.146087, 0.226744, 0.274534, 0.296199)
    assert series.delta_q_legacy_trajectory == (0.077833, 0.117057, 0.128203, 0.141165)
    assert series.delta_q_v15_trajectory == series.delta_q_trajectory


def test_parse_latent_telemetry_csv_legacy_columns_only(tmp_path: Path) -> None:
    path = tmp_path / "latent_telemetry.csv"
    path.write_text(
        "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,"
        "saaq_delta_q_prev,saaq_delta_q_target\n"
        "1,12.0,2.4,0.5,0.0,0.1\n",
        encoding="utf-8",
    )
    series = parse_latent_telemetry_csv(path)
    assert series.firing_rate_mean == pytest.approx(12.0)
    assert series.membrane_pressure_mean == pytest.approx(0.2)
    assert series.delta_q_mean == pytest.approx(0.1)
    assert series.delta_q_legacy_mean is None
    assert series.delta_q_v15_mean is None
    assert series.delta_q_legacy_trajectory == ()


def test_parse_summary_and_manifest() -> None:
    summary = parse_summary(RUN_DIR / "summary.json")
    manifest = parse_run_manifest(RUN_DIR / "run_manifest.json")
    assert classify_json_artifact(summary) == "summary"
    assert classify_json_artifact(manifest) == "manifest"
    assert summary["saaq_rule"] == "SaaqV1_5SqrtRate"
    assert summary["metrics"]["latent_rows"] == 4
    assert manifest["saaq_dual_emit"] is True
    assert "latent_telemetry.csv" in manifest["generated_files"]


def test_parse_tick_telemetry() -> None:
    ticks = parse_tick_telemetry(RUN_DIR / "tick_telemetry.txt")
    assert ticks.tick_count == 4
    assert ticks.mean_elapsed_us == pytest.approx(42.5)
    assert ticks.last_best_walker == 3


def test_load_run_directory() -> None:
    run = load_corinth_canal(RUN_DIR)
    assert run.experiment_id == "saaq-lambda-001"
    assert run.saaq_rule == "SaaqV1_5SqrtRate"
    assert run.saaq_dual_emit is True
    assert run.model_family == "Olmoe"
    assert run.model_slug == "olmoe_baseline"
    assert run.firing_rate == pytest.approx(7.3875)
    assert run.membrane_pressure == pytest.approx(0.03645833333333333)
    assert run.saaq_delta_q_last == pytest.approx(0.296199)
    assert run.ticks_completed == 4
    assert run.latent_rows == 4
    assert run.skipped == ()
    assert run.saaq_delta_q_trajectory[-1] == pytest.approx(0.296199)
    assert run.spike_density is None
    assert run.event_rate is None


def test_load_summary_in_run_dir_pulls_siblings() -> None:
    run = load_corinth_canal(RUN_DIR / "summary.json")
    assert run.firing_rate == pytest.approx(7.3875)
    assert run.run_manifest is not None
    assert run.run_manifest["saaq_primary_rule"] == "SaaqV1_5SqrtRate"


def test_missing_path_is_skipped() -> None:
    assert try_load_corinth_canal(FIXTURES / "does-not-exist") is None
    assert CorinthCanalArtifact.try_from_path(FIXTURES / "does-not-exist") is None


def test_partial_run_directory_skips_missing_files(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        (RUN_DIR / "summary.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    run = load_corinth_canal(tmp_path)
    assert run.experiment_id == "saaq-lambda-001"
    assert run.saaq_rule == "SaaqV1_5SqrtRate"
    assert run.firing_rate is None
    assert CANONICAL_LATENT_CSV in run.skipped
    assert "tick_telemetry.txt" in run.skipped
    assert "run_manifest.json" in run.skipped


def test_malformed_sibling_does_not_discard_csv(tmp_path: Path) -> None:
    (tmp_path / "latent_telemetry.csv").write_text(
        (RUN_DIR / "latent_telemetry.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text("{not-json", encoding="utf-8")
    run = try_load_corinth_canal(tmp_path)
    assert run is not None
    assert run.firing_rate == pytest.approx(7.3875)
    assert "summary.json" in run.skipped


def test_dual_trajectories_stay_aligned_on_partial_row(tmp_path: Path) -> None:
    path = tmp_path / "latent_telemetry.csv"
    path.write_text(
        LATENT_CSV_HEADER
        + "\n"
        + "1,6.5,1.2,0.85,0.0,0.1,65,250,70,45,0.0,0.2,0.0,0.3\n"
        + "2,7.0,0.8,0.82,0.1,nan,66,255,71,46,0.2,0.25,0.3,0.35\n"
        + "3,8.0,-0.4,0.79,0.2,0.4,67,260,72,47,0.25,0.3,0.35,0.45\n",
        encoding="utf-8",
    )
    series = parse_latent_telemetry_csv(path)
    assert series.row_count == 3
    assert series.delta_q_trajectory == pytest.approx((0.1, 0.4))
    assert series.delta_q_legacy_trajectory == pytest.approx((0.2, 0.3))
    assert series.delta_q_v15_trajectory == pytest.approx((0.3, 0.45))
    assert series.delta_q_last == pytest.approx(0.4)


def test_unreadable_json_is_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{not-json", encoding="utf-8")
    assert try_load_corinth_canal(bad) is None


def test_metrics_accumulator_maps_saaq_overlay() -> None:
    accumulator = MetricsAccumulator()
    accumulator.apply_saaq(
        SaaqMetricOverlay(
            firing_rate=7.3875,
            membrane_pressure=0.0365,
            saaq_delta_q=0.235891,
            saaq_delta_q_last=0.296199,
            saaq_delta_q_legacy=0.1160645,
            saaq_delta_q_v15=0.235891,
            saaq_rule="SaaqV1_5SqrtRate",
            spike_density=0.3078,
        )
    )
    summary = accumulator.summary(total_time_s=1.0, vram_gb=14.0)
    assert summary.accuracy == pytest.approx(0.0)
    assert summary.firing_rate == pytest.approx(7.3875)
    assert summary.membrane_pressure == pytest.approx(0.0365)
    assert summary.saaq_delta_q == pytest.approx(0.235891)
    assert summary.saaq_rule == "SaaqV1_5SqrtRate"
    assert summary.spike_density == pytest.approx(0.3078)


def test_artifact_to_overlay_and_merge() -> None:
    artifact = CorinthCanalArtifact.from_path(RUN_DIR)
    overlay = artifact.to_saaq_overlay()
    assert overlay.firing_rate == pytest.approx(7.3875)

    telemetry = TelemetrySnapshot(
        system=SystemSnapshot(
            cpu_count_logical=4,
            cpu_count_physical=2,
            memory_total_gb=16.0,
            memory_available_gb=8.0,
            gpu_count=None,
            gpu_names=None,
            gpu_driver_version=None,
            cuda_version=None,
            platform="Linux",
            python_version="3.14.0",
        )
    )
    merged = merge_upstream_artifacts(telemetry, corinth=artifact)
    assert merged.routing is not None
    assert merged.routing.firing_rate == pytest.approx(7.3875)
    assert merged.routing.membrane_pressure == pytest.approx(0.03645833333333333)
    assert merged.routing.saaq_delta_q_v15 == pytest.approx(0.235891)
    assert merged.saaq_rule == "SaaqV1_5SqrtRate"
    assert merged.saaq_model_family == "Olmoe"
    assert merged.saaq_delta_q_trajectory is not None
    assert len(merged.saaq_delta_q_trajectory) == 4


def test_cli_corinth_canal_dir_flag() -> None:
    args = parse_args(
        [
            "--config",
            "configs/benchmark.sample.json",
            "--corinth-canal-dir",
            str(RUN_DIR),
        ]
    )
    assert args.corinth_canal_dir == Path(str(RUN_DIR))


def test_benchmark_merges_saaq_into_json_and_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "benchmark.json"
    config = {
        "run_name": "saaq-overlay",
        "seed": 1,
        "model": {"backend": "mock", "name": "toy-model", "revision": "local"},
        "quantization": ["fp16"],
        "datasets": [
            {
                "name": "smoke",
                "source": "jsonl",
                "path": str(dataset_path),
                "split": "validation",
                "max_samples": 1,
            }
        ],
        "telemetry": {"corinth_canal_dir": str(RUN_DIR)},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "reports"

    run_benchmarks(config_path, output_dir, ["json", "csv"], seed_override=7)

    json_files = list((output_dir / "json").glob("*.json"))
    csv_files = list((output_dir / "csv").glob("*.csv"))
    assert len(json_files) == 1
    report = json.loads(json_files[0].read_text(encoding="utf-8"))
    telemetry = report["run"]["telemetry"]
    assert telemetry["firing_rate"] == pytest.approx(7.3875)
    assert telemetry["membrane_pressure"] == pytest.approx(0.03645833333333333)
    assert telemetry["saaq_delta_q"] == pytest.approx(0.235891)
    assert telemetry["saaq_rule"] == "SaaqV1_5SqrtRate"
    assert telemetry["saaq_delta_q_trajectory"] == [
        0.146087,
        0.226744,
        0.274534,
        0.296199,
    ]
    row = report["results"][0]
    assert row["firing_rate"] == pytest.approx(7.3875)
    assert row["membrane_pressure"] == pytest.approx(0.03645833333333333)
    assert row["saaq_delta_q"] == pytest.approx(0.235891)
    assert row["saaq_delta_q_legacy"] == pytest.approx(0.1160645)
    assert row["saaq_delta_q_v15"] == pytest.approx(0.235891)
    assert row["telemetry_firing_rate"] == pytest.approx(7.3875)
    assert row["telemetry_saaq_rule"] == "SaaqV1_5SqrtRate"

    with csv_files[0].open("r", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert float(csv_row["firing_rate"]) == pytest.approx(7.3875)
    assert float(csv_row["saaq_delta_q_last"]) == pytest.approx(0.296199)
    assert csv_row["saaq_rule"] == "SaaqV1_5SqrtRate"
    assert json.loads(csv_row["telemetry_saaq_delta_q_trajectory"]) == [
        0.146087,
        0.226744,
        0.274534,
        0.296199,
    ]


def test_benchmark_cli_dir_overrides_missing_config(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "benchmark.json"
    config = {
        "run_name": "saaq-cli",
        "seed": 1,
        "model": {"backend": "mock", "name": "toy-model", "revision": "local"},
        "quantization": ["fp16"],
        "datasets": [
            {
                "name": "smoke",
                "source": "jsonl",
                "path": str(dataset_path),
                "split": "validation",
                "max_samples": 1,
            }
        ],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "reports"
    run_benchmarks(
        config_path,
        output_dir,
        ["json"],
        seed_override=5,
        corinth_canal_dir=RUN_DIR,
    )
    report = json.loads(next((output_dir / "json").glob("*.json")).read_text(encoding="utf-8"))
    assert report["run"]["telemetry"]["firing_rate"] == pytest.approx(7.3875)
    assert report["results"][0]["saaq_rule"] == "SaaqV1_5SqrtRate"


def test_missing_corinth_dir_does_not_break_benchmark(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "benchmark.json"
    config = {
        "run_name": "saaq-missing",
        "seed": 1,
        "model": {"backend": "mock", "name": "toy-model", "revision": "local"},
        "quantization": ["fp16"],
        "datasets": [
            {
                "name": "smoke",
                "source": "jsonl",
                "path": str(dataset_path),
                "split": "validation",
                "max_samples": 1,
            }
        ],
        "telemetry": {"corinth_canal_dir": str(tmp_path / "absent-run")},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "reports"
    run_benchmarks(config_path, output_dir, ["json"], seed_override=3)
    report = json.loads(next((output_dir / "json").glob("*.json")).read_text(encoding="utf-8"))
    assert report["run"]["telemetry"]["firing_rate"] is None
    assert report["results"][0]["firing_rate"] is None
    assert report["results"][0]["accuracy"] is not None


def test_header_only_csv_has_zero_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(LATENT_CSV_HEADER + "\n", encoding="utf-8")
    series = parse_latent_telemetry_csv(path)
    assert series.row_count == 0
    assert series.delta_q_mean is None


def test_non_finite_csv_cells_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "nan.csv"
    path.write_text(
        "timestamp_ms,avg_pop_firing_rate_hz,membrane_dv_dt,routing_entropy,"
        "saaq_delta_q_target,saaq_delta_q_legacy_target,saaq_delta_q_v15_target\n"
        "1,nan,0.1,0.2,inf,0.3,-inf\n",
        encoding="utf-8",
    )
    series = parse_latent_telemetry_csv(path)
    assert series.row_count == 1
    assert series.firing_rate_mean is None
    assert series.delta_q_mean is None
    assert series.membrane_dv_dt_mean == pytest.approx(0.1)
    assert series.delta_q_legacy_mean is None
    assert series.delta_q_legacy_trajectory == ()
    assert series.delta_q_v15_trajectory == ()


def test_invalid_tick_value_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ticks.txt"
    path.write_text(
        "tick= best_walker=1 elapsed_us=2\n"
        "tick=3 best_walker=1 elapsed_us=4\n",
        encoding="utf-8",
    )
    ticks = parse_tick_telemetry(path)
    assert ticks.tick_count == 1
    assert ticks.mean_elapsed_us == pytest.approx(4.0)


def test_legacy_json_accepts_serialized_aliases(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "x",
                "saaq_delta_q_legacy_trajectory": [0.1, 0.2],
                "saaq_delta_q_v15_trajectory": [0.3],
                "saaq_model_family": "Olmoe",
                "saaq_primary_rule": "SaaqV1_5SqrtRate",
            }
        ),
        encoding="utf-8",
    )
    run = load_corinth_canal(path)
    assert run.saaq_delta_q_legacy_trajectory == (0.1, 0.2)
    assert run.saaq_delta_q_v15_trajectory == (0.3,)
    assert run.model_family == "Olmoe"
    assert run.saaq_primary_rule == "SaaqV1_5SqrtRate"
