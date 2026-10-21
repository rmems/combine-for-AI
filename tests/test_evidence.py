from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from combine_for_ai.evidence import (
    Capability,
    CapabilityError,
    DataProvenance,
    EvidenceError,
    EvidenceRecord,
    ExecutionKind,
    MetricProvenance,
    MetricStatus,
    MetricValue,
    WorkloadTrack,
    load_evidence,
    negotiate_capabilities,
    parse_evidence,
    require_capabilities,
    unavailable_metric,
)
from combine_for_ai.evidence_io import (
    parse_evidence_markdown,
    read_evidence_csv,
    read_evidence_json,
    read_evidence_markdown,
    render_evidence_markdown,
    write_evidence_csv,
    write_evidence_json,
    write_evidence_markdown,
)
from evidence_builders import ALL_FIXTURES, SHA_A, ok_metric, imported_record, mock_record

FIXTURES = Path(__file__).parent / "fixtures" / "evidence"


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_conformance_fixture_matches_builder(name: str) -> None:
    on_disk = json.loads((FIXTURES / f"{name}.v1.json").read_text())
    assert load_evidence(on_disk) == ALL_FIXTURES[name]()


def test_imported_real_and_synthetic_differ_only_in_data_provenance() -> None:
    real = imported_record(DataProvenance.REAL, "x")
    synth = imported_record(DataProvenance.SYNTHETIC, "x")
    assert real.execution_kind is synth.execution_kind is ExecutionKind.IMPORTED
    assert real.metrics == synth.metrics
    assert real.data_provenance is not synth.data_provenance


@pytest.mark.parametrize(
    "status", [MetricStatus.UNAVAILABLE, MetricStatus.UNSUPPORTED, MetricStatus.ERROR]
)
def test_non_ok_states_never_carry_a_value(status: MetricStatus) -> None:
    assert unavailable_metric("accuracy", status=status).value is None
    with pytest.raises(ValueError):
        MetricValue(name="a", value=0.0, status=status, provenance=MetricProvenance.UNAVAILABLE)
    with pytest.raises(ValueError):
        MetricValue(name="a", status=status, provenance=MetricProvenance.MEASURED)


@pytest.mark.parametrize("bad", [None, math.nan, math.inf])
def test_ok_requires_finite_value(bad) -> None:
    with pytest.raises(ValueError):
        ok_metric("a", bad, MetricProvenance.MEASURED)


def test_ok_requires_native_name_unit_definition_and_import_hash() -> None:
    with pytest.raises(ValueError):
        MetricValue(name="a", value=1.0, status=MetricStatus.OK, provenance=MetricProvenance.MEASURED)
    with pytest.raises(ValueError):
        ok_metric("a", 1.0, MetricProvenance.IMPORTED)  # no source hash
    with pytest.raises(ValueError):
        ok_metric("a", 1.0, MetricProvenance.IMPORTED, sha="XYZ")


def test_metadata_only_cannot_claim_metrics() -> None:
    kwargs = {"record_id": "r", "execution_kind": ExecutionKind.METADATA_ONLY, "data_provenance": DataProvenance.UNKNOWN}
    for provenance in (MetricProvenance.MEASURED, MetricProvenance.PRESET):
        with pytest.raises(ValueError, match="metadata_only"):
            EvidenceRecord(**kwargs, metrics=(ok_metric("latency_ms", 1.0, provenance),))


def test_preset_provenance_is_mock_only() -> None:
    preset = ok_metric("throughput", 1.0, MetricProvenance.PRESET)
    with pytest.raises(ValueError, match="provenance"):
        EvidenceRecord(
            record_id="r",
            execution_kind=ExecutionKind.MEASURED,
            data_provenance=DataProvenance.REAL,
            environment_identity="e",
            metrics=(preset,),
        )


def test_mock_cannot_claim_real_data_and_measured_needs_environment() -> None:
    with pytest.raises(ValueError, match="real data"):
        EvidenceRecord(record_id="r", execution_kind=ExecutionKind.MOCK, data_provenance=DataProvenance.REAL)
    with pytest.raises(ValueError, match="environment_identity"):
        EvidenceRecord(
            record_id="r",
            execution_kind=ExecutionKind.MEASURED,
            data_provenance=DataProvenance.REAL,
            metrics=(ok_metric("a", 1.0, MetricProvenance.MEASURED),),
        )


def test_toy_data_never_substitutes_for_real_request() -> None:
    with pytest.raises(ValueError, match="substitute"):
        EvidenceRecord(
            record_id="r",
            workload_track=WorkloadTrack.LIKELIHOOD,
            execution_kind=ExecutionKind.MOCK,
            data_provenance=DataProvenance.FIXTURE,
            requested_data_provenance=DataProvenance.REAL,
            capabilities=("likelihood",),
            metrics=(ok_metric("a", 1.0, MetricProvenance.MOCK_COMPUTED),),
        )


def test_track_requires_declared_capability_not_profile_registration() -> None:
    with pytest.raises(ValueError, match="capability"):
        EvidenceRecord(
            record_id="r",
            workload_track=WorkloadTrack.TOOL_EXECUTION,
            execution_kind=ExecutionKind.MOCK,
            data_provenance=DataProvenance.FIXTURE,
            capabilities=("generation", "likelihood"),
            metrics=(ok_metric("a", 1.0, MetricProvenance.MOCK_COMPUTED),),
        )


def test_capability_negotiation() -> None:
    have = {Capability.LIKELIHOOD}
    supported, missing = negotiate_capabilities({Capability.LIKELIHOOD, Capability.TEMPORAL}, have)
    assert supported == {Capability.LIKELIHOOD} and missing == {Capability.TEMPORAL}
    with pytest.raises(CapabilityError, match="temporal"):
        require_capabilities({Capability.TEMPORAL}, have)
    require_capabilities({Capability.LIKELIHOOD}, have)


def test_architecture_and_track_are_separate_fields() -> None:
    record = mock_record()
    assert record.architecture == "moe" and record.workload_track is WorkloadTrack.LIKELIHOOD
    assert "architecture" not in {t.value for t in WorkloadTrack}


def test_unknown_or_future_schema_is_rejected() -> None:
    payload = mock_record().to_payload()
    for version in (2, -1, "1", True):
        with pytest.raises(EvidenceError):
            load_evidence({**payload, "schema_version": version})
    with pytest.raises(EvidenceError):
        parse_evidence({**payload, "surprise": 1})


# --- legacy migration -------------------------------------------------------


def test_legacy_smoke_row_migrates_with_preset_labels() -> None:
    record = load_evidence(json.loads((FIXTURES / "legacy_smoke_row.v0.json").read_text()))
    assert record.execution_kind is ExecutionKind.MOCK
    assert record.data_provenance is DataProvenance.FIXTURE
    assert record.metric("throughput").provenance is MetricProvenance.PRESET
    assert record.metric("vram_gb").provenance is MetricProvenance.PRESET
    assert record.metric("perplexity").provenance is MetricProvenance.MOCK_COMPUTED
    assert record.metric("perplexity").value == 3.25


def test_legacy_failed_row_keeps_every_metric_unavailable() -> None:
    record = load_evidence(json.loads((FIXTURES / "legacy_smoke_failed.v0.json").read_text()))
    assert record.metrics and all(m.status is MetricStatus.UNAVAILABLE and m.value is None for m in record.metrics)


def test_legacy_failed_row_ignores_stale_numbers() -> None:
    row = {"run_id": "r", "status": "failed", "throughput_tps": 0.0, "perplexity": 1.0}
    assert all(m.value is None for m in load_evidence(row).metrics)


def test_legacy_matrix_cell_migrates_and_nan_is_unavailable() -> None:
    cell = {
        "cell_id": "c1",
        "seed": 5,
        "metrics": {"accuracy": 0.75, "perplexity": float("nan"), "throughput": 100.0, "vram_gb": 5.0, "latency_ms": 2.0, "routing_entropy": 0.5},
    }
    record = load_evidence(cell)
    assert record.record_id == "c1" and record.seed == 5
    assert record.metric("accuracy").value == 0.75
    assert record.metric("perplexity").status is MetricStatus.UNAVAILABLE
    assert record.metric("vram_gb").provenance is MetricProvenance.PRESET


def test_unrecognised_legacy_payload_is_an_error() -> None:
    with pytest.raises(EvidenceError):
        load_evidence({"hello": "world"})


# --- round trips ------------------------------------------------------------


def _all_records() -> list[EvidenceRecord]:
    return [build() for build in ALL_FIXTURES.values()]


def test_json_round_trip(tmp_path: Path) -> None:
    write_evidence_json(tmp_path / "e.json", _all_records())
    assert read_evidence_json(tmp_path / "e.json") == _all_records()


def test_csv_round_trip_preserves_hashes_units_and_negative_values(tmp_path: Path) -> None:
    write_evidence_csv(tmp_path / "e.csv", _all_records())
    back = read_evidence_csv(tmp_path / "e.csv")
    assert back == _all_records()
    real = next(r for r in back if r.record_id == "imp-real")
    metric = real.metric("mean_logprob")
    assert metric.value == -1.25e-3 and metric.source_artifact_sha256 == SHA_A
    assert metric.unit == "nats" and metric.native_name == "mean_ll"
    assert real.seed == -3


def test_markdown_round_trip_with_pipes_and_newlines(tmp_path: Path) -> None:
    text = render_evidence_markdown(_all_records())
    assert parse_evidence_markdown(text) == _all_records()
    write_evidence_markdown(tmp_path / "e.md", _all_records())
    assert read_evidence_markdown(tmp_path / "e.md") == _all_records()


def test_independent_provenance_survives_every_format(tmp_path: Path) -> None:
    records = _all_records()
    write_evidence_csv(tmp_path / "e.csv", records)
    write_evidence_json(tmp_path / "e.json", records)
    write_evidence_markdown(tmp_path / "e.md", records)
    for back in (read_evidence_csv(tmp_path / "e.csv"), read_evidence_json(tmp_path / "e.json"), read_evidence_markdown(tmp_path / "e.md")):
        by_id = {r.record_id: r for r in back}
        assert by_id["imp-real"].data_provenance is DataProvenance.REAL
        assert by_id["imp-synth"].data_provenance is DataProvenance.SYNTHETIC
        assert by_id["imp-real"].execution_kind is by_id["imp-synth"].execution_kind
        assert by_id["meta-1"].metric("energy_j").status is MetricStatus.UNSUPPORTED
        assert by_id["meta-1"].metric("energy_j").value is None


def test_tampered_csv_zero_for_unsupported_metric_is_rejected(tmp_path: Path) -> None:
    write_evidence_csv(tmp_path / "e.csv", [ALL_FIXTURES["metadata_only"]()])
    text = (tmp_path / "e.csv").read_text()
    assert ",,unsupported,unavailable," in text
    (tmp_path / "e.csv").write_text(text.replace(",,unsupported,unavailable,", ",0.0,unsupported,unavailable,", 1))
    with pytest.raises(EvidenceError, match="must not carry a value"):
        read_evidence_csv(tmp_path / "e.csv")


def test_malformed_metric_value_is_an_evidence_error(tmp_path: Path) -> None:
    write_evidence_csv(tmp_path / "e.csv", [mock_record()])
    text = (tmp_path / "e.csv").read_text().replace("0.5", "abc", 1)
    (tmp_path / "e.csv").write_text(text)
    with pytest.raises(EvidenceError, match="malformed value"):
        read_evidence_csv(tmp_path / "e.csv")


def test_csv_guard_prefixes_are_real_control_characters() -> None:
    from combine_for_ai.csvio import CSV_FORMULA_PREFIXES

    assert "\t" in CSV_FORMULA_PREFIXES and "\r" in CSV_FORMULA_PREFIXES
    assert "\\t" not in CSV_FORMULA_PREFIXES


def test_legacy_language_modeling_cell_does_not_report_placeholder_zeros() -> None:
    record = load_evidence(json.loads((FIXTURES / "legacy_lm_cell.v0.json").read_text()))
    for name in ("accuracy", "routing_entropy"):
        metric = record.metric(name)
        assert metric.status is MetricStatus.UNAVAILABLE and metric.value is None
    assert record.metric("perplexity").value == 12.3


def test_legacy_cell_with_real_nonzero_accuracy_keeps_zero_entropy() -> None:
    cell = {"cell_id": "c", "metrics": {"accuracy": 1.0, "routing_entropy": 0.0, "perplexity": 2.0}}
    record = load_evidence(cell)
    assert record.metric("accuracy").value == 1.0
    assert record.metric("routing_entropy").value == 0.0


def test_markdown_round_trip_with_unicode_line_separators() -> None:
    record = imported_record(DataProvenance.REAL, "sep").model_copy(
        update={"notes": "a\x0bb\x0cc\x1cd\x85e\u2028f\u2029g"}
    )
    assert parse_evidence_markdown(render_evidence_markdown([record])) == [record]


def test_csv_round_trip_with_values_that_start_with_an_apostrophe(tmp_path: Path) -> None:
    for notes in ("'-x", "'=x", "'plain", "'", "''", "-x", "=1+1"):
        record = mock_record().model_copy(update={"notes": notes})
        write_evidence_csv(tmp_path / "e.csv", [record])
        assert read_evidence_csv(tmp_path / "e.csv") == [record], notes


def test_sha256_rejects_trailing_newline() -> None:
    with pytest.raises(ValueError):
        ok_metric("a", 1.0, MetricProvenance.IMPORTED, sha=SHA_A + "\n")


@pytest.mark.parametrize("bad", [[], "text", 3, None])
def test_load_evidence_rejects_non_object_payloads(bad) -> None:
    with pytest.raises(EvidenceError, match="must be an object"):
        load_evidence(bad)


def test_read_evidence_json_wraps_non_object_records(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"records": [1]}))
    (tmp_path / "b.json").write_text(json.dumps([1, 2]))
    for name in ("a.json", "b.json"):
        with pytest.raises(EvidenceError):
            read_evidence_json(tmp_path / name)


def test_evidence_csv_writer_lives_in_the_installed_package() -> None:
    import combine_for_ai.evidence_io as io_module
    from benchmarks.reporting import write_csv as reporting_write_csv
    from combine_for_ai.csvio import write_csv

    assert io_module.write_csv is write_csv
    assert reporting_write_csv is write_csv


def test_records_sharing_a_record_id_stay_separate_in_csv_and_markdown(tmp_path: Path) -> None:
    empty = ALL_FIXTURES["metadata_only"]().model_copy(update={"record_id": "dup", "metrics": ()})
    with_metrics = mock_record().model_copy(update={"record_id": "dup"})
    records = [empty, with_metrics, empty]
    write_evidence_csv(tmp_path / "e.csv", records)
    assert read_evidence_csv(tmp_path / "e.csv") == records
    assert parse_evidence_markdown(render_evidence_markdown(records)) == records


def test_legacy_oversized_integer_becomes_unavailable_not_a_crash() -> None:
    cell = {"cell_id": "c", "metrics": {"accuracy": 0.5, "perplexity": 10**309, "throughput": -(10**400)}}
    record = load_evidence(cell)
    assert record.metric("accuracy").value == 0.5
    for name in ("perplexity", "throughput"):
        assert record.metric(name).status is MetricStatus.UNAVAILABLE
