from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from benchmarks.fixture_presets import MOCK_PERFORMANCE_PRESETS, mock_performance
from benchmarks.models import (
    MockModelAdapter,
    ModelSpec,
    QuantizationProfile,
    build_model_adapter,
    default_quantization_registry,
)
from combine_for_ai.manifest import GOZ1_MAGIC
from combine_for_ai.evidence import Capability, ExecutionKind, MetricStatus, load_evidence


def test_default_profiles_carry_no_performance_numbers() -> None:
    registry = default_quantization_registry()
    for name in registry.supported_names():
        profile = registry.get(name)
        assert profile.speed_tps is None and profile.vram_gb is None


def test_mock_presets_cover_default_profiles_and_are_mock_only() -> None:
    registry = default_quantization_registry()
    for name in registry.supported_names():
        assert mock_performance(registry.get(name)) == MOCK_PERFORMANCE_PRESETS[name]
        for kind in (ExecutionKind.MEASURED, ExecutionKind.IMPORTED, ExecutionKind.METADATA_ONLY):
            with pytest.raises(ValueError, match="fixture-only"):
                mock_performance(registry.get(name), kind)


def test_unknown_profile_without_override_has_no_preset() -> None:
    profile = QuantizationProfile("x", "fp8", "x", 8, True, "n")
    with pytest.raises(ValueError, match="no mock performance preset"):
        mock_performance(profile)


def test_mock_adapter_declares_capabilities_independent_of_profile() -> None:
    profile = default_quantization_registry().get("saaq")
    adapter = build_model_adapter(ModelSpec("mock", "m"), profile)
    assert isinstance(adapter, MockModelAdapter)
    assert adapter.execution_kind is ExecutionKind.MOCK
    assert adapter.capabilities == {Capability.GENERATION, Capability.LIKELIHOOD}
    assert Capability.TEMPORAL not in adapter.capabilities
    assert Capability.TOOL_EXECUTION not in adapter.capabilities


def test_header_only_goz1_smoke_claims_no_performance(tmp_path: Path) -> None:
    from benchmarks.artifact_smoke import run_artifact_smoke
    manifest = tmp_path / "manifest.json"
    pack = tmp_path / "pack.goz1"
    pack.write_bytes(struct.pack("<IIQQ", GOZ1_MAGIC, 3, 0, 0))
    manifest.write_text(
        json.dumps({"model_name": "goz-demo", "source_artifact": {"format": "goz1", "path": str(pack)}}),
        encoding="utf-8",
    )
    payload = run_artifact_smoke(manifest, tmp_path / "out", ["json", "csv"])
    assert payload["result"]["status"] == "success"
    assert payload["result"]["throughput_tps"] is None
    assert payload["result"]["peak_vram_gb"] is None
    record = load_evidence(payload["evidence"])
    for name in ("latency_ms", "accuracy", "throughput", "energy_j", "vram_gb"):
        metric = record.metric(name)
        assert metric.value is None and metric.status is MetricStatus.UNSUPPORTED
    written = json.loads(next((tmp_path / "out" / "json").glob("*.json")).read_text())
    assert written["evidence"] == payload["evidence"]


def test_failed_smoke_evidence_is_error_not_zero(tmp_path: Path) -> None:
    from benchmarks.artifact_smoke import run_artifact_smoke

    payload = run_artifact_smoke(tmp_path / "missing.json", tmp_path / "out", ["json"])
    record = load_evidence(payload["evidence"])
    assert record.execution_kind is ExecutionKind.METADATA_ONLY
    assert all(m.status is MetricStatus.ERROR and m.value is None for m in record.metrics)


def test_empty_failure_reason_still_yields_an_error_reason() -> None:
    from benchmarks.artifact_smoke import _smoke_evidence

    record = load_evidence(_smoke_evidence("r", None, perplexity=None, max_samples=None, failure_reason=""))
    assert all(m.status is MetricStatus.ERROR and m.reason == "artifact smoke failed" for m in record.metrics)


def test_smoke_report_file_reads_back_its_embedded_envelope(tmp_path: Path) -> None:
    from benchmarks.artifact_smoke import run_artifact_smoke
    from combine_for_ai.evidence_io import read_evidence_json

    payload = run_artifact_smoke(tmp_path / "missing.json", tmp_path / "out", ["json"])
    [record] = read_evidence_json(next((tmp_path / "out" / "json").glob("*.json")))
    assert record == load_evidence(payload["evidence"])
    assert record.execution_kind is ExecutionKind.METADATA_ONLY
    assert record.capabilities == ()
    assert all(m.status is MetricStatus.ERROR for m in record.metrics)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_mock_overrides_must_be_positive_and_finite(bad: float) -> None:
    for kwargs in ({"speed_tps": bad, "vram_gb": 1.0}, {"speed_tps": 1.0, "vram_gb": bad}):
        profile = QuantizationProfile("x", "fp8", "x", 8, True, "n", **kwargs)
        with pytest.raises(ValueError, match="finite and > 0"):
            mock_performance(profile)


def test_smoke_data_provenance_distinguishes_bundled_from_supplied(tmp_path: Path) -> None:
    from benchmarks.artifact_smoke import run_artifact_smoke
    from combine_for_ai.evidence import DataProvenance

    artifact = tmp_path / "m.gguf"
    artifact.write_text("ok", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"model_name": "g", "source_artifact": {"format": "gguf", "path": str(artifact)}}),
        encoding="utf-8",
    )
    bundled = run_artifact_smoke(manifest, tmp_path / "a", ["json"])
    assert load_evidence(bundled["evidence"]).data_provenance is DataProvenance.FIXTURE
    sample = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "lambada.sample.jsonl"
    supplied = run_artifact_smoke(manifest, tmp_path / "b", ["json"], dataset_path=sample)
    assert load_evidence(supplied["evidence"]).data_provenance is DataProvenance.UNKNOWN


def test_smoke_evidence_records_requested_budget_and_artifact_hash(tmp_path: Path) -> None:
    import hashlib

    from benchmarks.artifact_smoke import run_artifact_smoke

    artifact = tmp_path / "m.gguf"
    artifact.write_bytes(b"model-bytes" * 1000)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"model_name": "g", "source_artifact": {"format": "gguf", "path": str(artifact)}}),
        encoding="utf-8",
    )
    payload = run_artifact_smoke(manifest, tmp_path / "out", ["json"], max_samples=50)
    record = load_evidence(payload["evidence"])
    assert record.budget.max_samples == 50  # requested, not the dataset-limited count
    assert record.source_artifact_sha256 == hashlib.sha256(b"model-bytes" * 1000).hexdigest()


def test_smoke_evidence_leaves_hash_unset_without_a_local_file(tmp_path: Path) -> None:
    from benchmarks.artifact_smoke import _file_sha256

    assert _file_sha256(None) is None
    assert _file_sha256(str(tmp_path / "missing.bin")) is None
