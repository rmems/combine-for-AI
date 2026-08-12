from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.artifact_smoke import run_artifact_smoke, select_artifact_for_smoke
from nfl_combine_for_ai.manifest import (
    GOZ1_MAGIC,
    ArtifactFormat,
    ArtifactStatus,
    GeneratedArtifact,
    Goz1Metadata,
    ModelManifest,
    ScaleSource,
    SourceArtifact,
    default_scale_source_for_version,
    dispatch_artifact,
    load_manifest,
    load_manifest_from_string,
    sniff_goz1_header,
    write_minimal_goz1_fixture,
)


def test_goz1_format_in_schema() -> None:
    manifest = load_manifest_from_string(
        json.dumps(
            {
                "model_name": "goz1-demo",
                "source_artifact": {"format": "safetensors", "hf_repo_id": "x/y"},
                "generated_artifacts": [
                    {
                        "format": "goz1",
                        "status": "success",
                        "path": "out.goz1",
                        "goz1": {
                            "container_version": 3,
                            "scale_source": "pack_v3",
                            "gif_threshold": 0.05,
                        },
                    }
                ],
            }
        )
    )
    assert manifest.generated_artifacts[0].format == ArtifactFormat.GOZ1
    assert manifest.generated_artifacts[0].goz1 is not None
    assert manifest.generated_artifacts[0].goz1.container_version == 3
    assert manifest.generated_artifacts[0].goz1.scale_source == ScaleSource.PACK_V3


def test_dispatch_generated_goz1() -> None:
    manifest = ModelManifest(
        model_name="goz1-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.SAFETENSORS),
        generated_artifacts=[
            GeneratedArtifact(
                format=ArtifactFormat.GOZ1,
                status=ArtifactStatus.SUCCESS,
                goz1=Goz1Metadata(container_version=3),
            )
        ],
    )
    assert dispatch_artifact(manifest) == "generated_goz1"


def test_dispatch_source_goz1() -> None:
    manifest = ModelManifest(
        model_name="goz1-source",
        source_artifact=SourceArtifact(format=ArtifactFormat.GOZ1, path="x.goz1"),
    )
    assert dispatch_artifact(manifest) == "goz1"


def test_sniff_goz1_header_valid_v3() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_minimal_goz1_fixture(
            Path(tmpdir) / "sample.goz1", version=3, tensor_count=12, meta_count=2
        )
        header = sniff_goz1_header(path)
        assert header.valid
        assert header.version == 3
        assert header.tensor_count == 12
        assert header.meta_count == 2
        assert header.error is None
        assert default_scale_source_for_version(3) == ScaleSource.PACK_V3


def test_sniff_goz1_header_bad_magic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.goz1"
        path.write_bytes(struct.pack("<IIQQ", 0xDEADBEEF, 3, 0, 0))
        header = sniff_goz1_header(path)
        assert not header.valid
        assert header.error is not None
        assert "magic" in header.error.lower()


def test_sniff_goz1_header_unsupported_version() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "v99.goz1"
        path.write_bytes(struct.pack("<IIQQ", GOZ1_MAGIC, 99, 0, 0))
        header = sniff_goz1_header(path)
        assert not header.valid
        assert "version" in (header.error or "").lower()


def test_select_artifact_goz1_missing_path_fails_closed() -> None:
    """GOZ1 success with missing path must not fall through to HF."""
    manifest = load_manifest_from_string(
        json.dumps(
            {
                "model_name": "missing-goz1",
                "source_artifact": {
                    "format": "safetensors",
                    "hf_repo_id": "xai-org/grok-1",
                },
                "generated_artifacts": [
                    {
                        "format": "goz1",
                        "status": "success",
                        "path": "artifacts/does-not-exist.goz1",
                    }
                ],
            }
        )
    )
    selection = select_artifact_for_smoke(manifest, Path.cwd())
    assert selection.status == "failed"
    assert selection.runtime_format == "generated_goz1"
    assert selection.quantization_name == "saaq"
    assert selection.failure_reason is not None
    assert "does not exist" in selection.failure_reason


def test_select_artifact_prefers_goz1_over_awq() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        goz1 = write_minimal_goz1_fixture(tmp / "model.goz1", version=3, tensor_count=4)
        awq = tmp / "model-awq"
        awq.write_text("ok", encoding="utf-8")
        manifest = load_manifest_from_string(
            json.dumps(
                {
                    "model_name": "prefer-goz1",
                    "source_artifact": {"format": "safetensors", "path": str(tmp / "src")},
                    "generated_artifacts": [
                        {"format": "awq", "status": "success", "path": str(awq)},
                        {"format": "goz1", "status": "success", "path": str(goz1)},
                    ],
                }
            )
        )
        selection = select_artifact_for_smoke(manifest, tmp)
        assert selection.status == "success"
        assert selection.runtime_format == "generated_goz1"
        assert selection.quantization_name == "saaq"
        assert selection.generated_format == "goz1"


def test_select_artifact_goz1_invalid_header_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bad = tmp / "bad.goz1"
        bad.write_bytes(b"not-a-goz1-header!!!!!")
        manifest = load_manifest_from_string(
            json.dumps(
                {
                    "model_name": "bad-goz1",
                    "source_artifact": {"format": "safetensors"},
                    "generated_artifacts": [
                        {"format": "goz1", "status": "success", "path": str(bad)}
                    ],
                }
            )
        )
        selection = select_artifact_for_smoke(manifest, tmp)
        assert selection.status == "failed"
        assert selection.runtime_format == "generated_goz1"
        assert selection.failure_reason


def test_run_artifact_smoke_goz1_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pack = write_minimal_goz1_fixture(tmp / "model.goz1", version=3, tensor_count=8)
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "model_name": "goz1-smoke",
                    "model_family": "grok",
                    "source_artifact": {
                        "format": "safetensors",
                        "hf_repo_id": "xai-org/grok-1",
                    },
                    "generated_artifacts": [
                        {
                            "format": "goz1",
                            "status": "success",
                            "path": str(pack),
                            "quantization_method": "saaq",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output_dir = tmp / "reports"
        payload = run_artifact_smoke(manifest_path, output_dir, ["json", "csv"])
        assert payload["result"]["status"] == "success"
        assert payload["result"]["runtime_format"] == "generated_goz1"
        assert payload["result"]["quantization"] == "saaq"
        assert payload["result"]["goz1_version"] == 3
        assert payload["result"]["goz1_tensor_count"] == 8
        assert payload["result"]["goz1_scale_source"] == "pack_v3"
        assert list((output_dir / "json").glob("*.artifact-smoke.json"))


def test_load_goz1_sample_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "configs" / "manifests" / "goz1.sample.json")
    assert manifest.model_family == "grok"
    assert dispatch_artifact(manifest) == "generated_goz1"
    gen = manifest.generated_artifacts[0]
    assert gen.format == ArtifactFormat.GOZ1
    assert gen.goz1 is not None
    assert gen.goz1.container_version == 3


def test_invalid_goz1_scale_source() -> None:
    with pytest.raises(ValidationError):
        load_manifest_from_string(
            json.dumps(
                {
                    "model_name": "bad",
                    "source_artifact": {"format": "goz1"},
                    "generated_artifacts": [
                        {
                            "format": "goz1",
                            "status": "success",
                            "goz1": {"scale_source": "not_a_source"},
                        }
                    ],
                }
            )
        )
