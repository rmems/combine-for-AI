from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from nfl_combine_for_ai.manifest import (
    ArtifactFormat,
    ArtifactStatus,
    GeneratedArtifact,
    ModelManifest,
    SourceArtifact,
    dispatch_artifact,
    load_manifest,
    load_manifest_from_string,
)


VALID_GGUF_MANIFEST = {
    "manifest_version": "1.0.0",
    "model_name": "test-gguf",
    "source_artifact": {
        "format": "gguf",
        "path": "models/test.gguf",
        "checksum_sha256": "abcd1234",
        "parameter_count": 7_000_000_000,
    },
    "generated_artifacts": [],
    "backend_compatibility": {"gguf": True, "awq": False, "gptq": False, "myelin_accelerator": False},
}


VALID_SAFETENSORS_MANIFEST = {
    "manifest_version": "1.0.0",
    "model_name": "test-safetensors",
    "source_artifact": {
        "format": "safetensors",
        "path": "models/test",
        "hf_repo_id": "org/model",
        "hf_revision": "main",
        "parameter_count": 8_000_000_000,
    },
    "generated_artifacts": [
        {
            "format": "awq",
            "status": "success",
            "path": "models/test-awq",
            "quantization_method": "awq",
            "bits": 4,
            "group_size": 128,
            "backend_compatibility": ["vllm"],
        }
    ],
    "backend_compatibility": {"gguf": False, "awq": True, "gptq": False, "myelin_accelerator": False},
}


def test_load_valid_gguf_manifest() -> None:
    manifest = load_manifest_from_string(json.dumps(VALID_GGUF_MANIFEST))
    assert manifest.model_name == "test-gguf"
    assert manifest.source_artifact.format == ArtifactFormat.GGUF
    assert manifest.source_artifact.parameter_count == 7_000_000_000
    assert manifest.generated_artifacts == []


def test_load_valid_safetensors_manifest() -> None:
    manifest = load_manifest_from_string(json.dumps(VALID_SAFETENSORS_MANIFEST))
    assert manifest.model_name == "test-safetensors"
    assert manifest.source_artifact.format == ArtifactFormat.SAFETENSORS
    assert len(manifest.generated_artifacts) == 1
    gen = manifest.generated_artifacts[0]
    assert gen.status == ArtifactStatus.SUCCESS
    assert gen.bits == 4


def test_load_from_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(VALID_GGUF_MANIFEST, tmp)
        tmp_path = Path(tmp.name)
    try:
        manifest = load_manifest(tmp_path)
        assert manifest.model_name == "test-gguf"
    finally:
        tmp_path.unlink()


def test_invalid_manifest_missing_model_name() -> None:
    bad = {"manifest_version": "1.0.0", "source_artifact": {"format": "gguf"}}
    with pytest.raises(ValidationError):
        load_manifest_from_string(json.dumps(bad))


def test_invalid_manifest_bad_format() -> None:
    bad = {
        "manifest_version": "1.0.0",
        "model_name": "bad",
        "source_artifact": {"format": "not_a_format"},
    }
    with pytest.raises(ValidationError):
        load_manifest_from_string(json.dumps(bad))


def test_invalid_manifest_bad_status() -> None:
    bad = {
        "manifest_version": "1.0.0",
        "model_name": "bad",
        "source_artifact": {"format": "gguf"},
        "generated_artifacts": [{"format": "awq", "status": "unknown"}],
    }
    with pytest.raises(ValidationError):
        load_manifest_from_string(json.dumps(bad))


def test_dispatch_gguf() -> None:
    manifest = ModelManifest(
        model_name="gguf-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.GGUF),
    )
    assert dispatch_artifact(manifest) == "gguf"


def test_dispatch_safetensors() -> None:
    manifest = ModelManifest(
        model_name="hf-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.SAFETENSORS),
    )
    assert dispatch_artifact(manifest) == "safetensors_hf"


def test_dispatch_hf() -> None:
    manifest = ModelManifest(
        model_name="hf-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.HF),
    )
    assert dispatch_artifact(manifest) == "safetensors_hf"


def test_dispatch_generated_success() -> None:
    manifest = ModelManifest(
        model_name="awq-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.SAFETENSORS),
        generated_artifacts=[
            GeneratedArtifact(format=ArtifactFormat.AWQ, status=ArtifactStatus.SUCCESS)
        ],
    )
    assert dispatch_artifact(manifest) == "generated_awq"


def test_dispatch_generated_planned() -> None:
    manifest = ModelManifest(
        model_name="planned-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.SAFETENSORS),
        generated_artifacts=[
            GeneratedArtifact(format=ArtifactFormat.GPTQ, status=ArtifactStatus.PLANNED)
        ],
    )
    assert dispatch_artifact(manifest) == "generated_gptq"


def test_dispatch_unknown() -> None:
    manifest = ModelManifest(
        model_name="unknown-model",
        source_artifact=SourceArtifact(format=ArtifactFormat.ONNX),
    )
    assert dispatch_artifact(manifest) == "unknown"


def test_load_example_manifests() -> None:
    """Ensure all committed example manifests validate successfully."""
    manifest_dir = Path(__file__).resolve().parents[1] / "configs" / "manifests"
    paths = list(manifest_dir.glob("*.json"))
    assert paths, f"no manifest files found in {manifest_dir}"
    for path in paths:
        manifest = load_manifest(path)
        assert manifest.manifest_version == "1.0.0"
        assert manifest.model_name
        assert manifest.source_artifact.format
