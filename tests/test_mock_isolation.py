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

    record = load_evidence(_smoke_evidence("r", None, perplexity=None, sample_count=0, failure_reason=""))
    assert all(m.status is MetricStatus.ERROR and m.reason == "artifact smoke failed" for m in record.metrics)
