from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pydantic import ValidationError

from combine_for_ai.manifest import (
    ArtifactFormat,
    ArtifactStatus,
    ModelManifest,
    load_manifest,
)


def test_load_yaml_manifest() -> None:
    """Test loading a YAML manifest."""
    yaml_content = """
manifest_version: "1.0.0"
model_name: "Test-YAML-Model"
model_family: "test"

source_artifact:
  format: "gguf"
  path: "models/test.gguf"
  checksum_sha256: "test1234"
  parameter_count: 7000000000

generated_artifacts:
  - format: "awq"
    status: "planned"
    quantization_method: "awq"
    bits: 4

backend_compatibility:
  gguf: true
  awq: true
  gptq: false
  myelin_accelerator: false
"""
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = Path(tmp.name)
    
    try:
        manifest = load_manifest(tmp_path)
        
        assert isinstance(manifest, ModelManifest)
        assert manifest.manifest_version == "1.0.0"
        assert manifest.model_name == "Test-YAML-Model"
        assert manifest.model_family == "test"
        
        assert manifest.source_artifact.format == ArtifactFormat.GGUF
        assert manifest.source_artifact.path == "models/test.gguf"
        assert manifest.source_artifact.checksum_sha256 == "test1234"
        assert manifest.source_artifact.parameter_count == 7000000000
        
        assert len(manifest.generated_artifacts) == 1
        gen = manifest.generated_artifacts[0]
        assert gen.format == ArtifactFormat.AWQ
        assert gen.status == ArtifactStatus.PLANNED
        assert gen.quantization_method == "awq"
        assert gen.bits == 4
        
        assert manifest.backend_compatibility is not None
        assert manifest.backend_compatibility.gguf is True
        assert manifest.backend_compatibility.awq is True
        assert manifest.backend_compatibility.gptq is False
        assert manifest.backend_compatibility.myelin_accelerator is False
        
    finally:
        tmp_path.unlink()


def test_yaml_and_json_equivalence() -> None:
    """Test that YAML and JSON manifests produce same result."""
    yaml_content = """
manifest_version: "1.0.0"
model_name: "Equivalent-Model"
source_artifact:
  format: "gguf"
  path: "models/eq.gguf"
"""
    
    json_content = """
{
  "manifest_version": "1.0.0",
  "model_name": "Equivalent-Model",
  "source_artifact": {
    "format": "gguf",
    "path": "models/eq.gguf"
  }
}
"""
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_yaml:
        tmp_yaml.write(yaml_content)
        yaml_path = Path(tmp_yaml.name)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_json:
        tmp_json.write(json_content)
        json_path = Path(tmp_json.name)
    
    try:
        yaml_manifest = load_manifest(yaml_path)
        json_manifest = load_manifest(json_path)
        
        # Both should produce identical manifests
        assert yaml_manifest.model_name == json_manifest.model_name
        assert yaml_manifest.source_artifact.format == json_manifest.source_artifact.format
        assert yaml_manifest.source_artifact.path == json_manifest.source_artifact.path
        
        # Check model validation
        assert isinstance(yaml_manifest, ModelManifest)
        assert isinstance(json_manifest, ModelManifest)
        
    finally:
        yaml_path.unlink()
        json_path.unlink()


def test_invalid_yaml_manifest() -> None:
    """Test that invalid YAML manifests raise validation errors."""
    invalid_yaml = """
manifest_version: "1.0.0"
# Missing model_name
source_artifact:
  format: "invalid_format"  # Invalid format value
  path: "models/test.gguf"
"""
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(invalid_yaml)
        tmp_path = Path(tmp.name)
    
    try:
        with pytest.raises(ValidationError):
            load_manifest(tmp_path)
    finally:
        tmp_path.unlink()


def test_yaml_with_comments() -> None:
    """Test that YAML with comments loads correctly."""
    yaml_with_comments = """
# This is a YAML manifest with comments
manifest_version: "1.0.0"
model_name: "Comment-Model"  # Model name comment
source_artifact:
  format: "gguf"
  path: "models/comment.gguf"
  # Optional field commented out
  # parameter_count: 7000000000
"""
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_with_comments)
        tmp_path = Path(tmp.name)
    
    try:
        manifest = load_manifest(tmp_path)
        
        assert manifest.model_name == "Comment-Model"
        assert manifest.source_artifact.format == ArtifactFormat.GGUF
        assert manifest.source_artifact.path == "models/comment.gguf"
        assert manifest.source_artifact.parameter_count is None  # Commented out
        
    finally:
        tmp_path.unlink()


def test_file_extension_agnostic() -> None:
    """Test that load_manifest works regardless of file extension."""
    yaml_content = """
manifest_version: "1.0.0"
model_name: "Extension-Test"
source_artifact:
  format: "gguf"
  path: "models/ext.gguf"
"""
    
    # Test with .yml extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = Path(tmp.name)
    
    try:
        manifest = load_manifest(tmp_path)
        assert manifest.model_name == "Extension-Test"
    finally:
        tmp_path.unlink()
    
    # Test with .yaml extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = Path(tmp.name)
    
    try:
        manifest = load_manifest(tmp_path)
        assert manifest.model_name == "Extension-Test"
    finally:
        tmp_path.unlink()


def test_yaml_without_pyyaml_fallback() -> None:
    """Test YAML loading when pyyaml is not available (should fallback to JSON)."""
    pytest.skip("Skipping because pyyaml is installed and mocking is complex")