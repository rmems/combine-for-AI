from __future__ import annotations

import json
import struct
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError


# Little-endian ASCII "GOZ1" — format SoT: rmems/grok-ozempic/docs/goz1-format.md
GOZ1_MAGIC = int.from_bytes(b"GOZ1", "little")
GOZ1_SUPPORTED_VERSIONS = frozenset({1, 2, 3})
GOZ1_ROW_SENTINEL = 0x5CA1E021


class ArtifactFormat(str, Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    HF = "hf"
    AWQ = "awq"
    GPTQ = "gptq"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    MYELIN = "myelin"
    GOZ1 = "goz1"


class ArtifactStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PLANNED = "planned"
    SKIPPED = "skipped"


class ScaleSource(str, Enum):
    """How ternary reconstruction scales were obtained."""

    PACK_V2 = "pack_v2"
    PACK_V3 = "pack_v3"
    LEGACY_ORACLE = "legacy_oracle"
    UNKNOWN = "unknown"


class SourceArtifact(BaseModel):
    format: ArtifactFormat
    path: str | None = None
    hf_repo_id: str | None = None
    hf_revision: str | None = None
    url: str | None = None
    checksum_sha256: str | None = Field(None, alias="checksum_sha256")
    parameter_count: int | None = None
    moe_layout: dict[str, Any] | None = None
    notes: str | None = None


class Goz1Metadata(BaseModel):
    """Optional GOZ1 pack metadata carried on generated artifacts or standalone fields."""

    container_version: int | None = None
    packing_scheme: str | None = None
    gif_threshold: float | None = None
    preserved_tensor_count: int | None = None
    ternary_tensor_count: int | None = None
    tensor_count: int | None = None
    scale_source: ScaleSource | None = None
    notes: str | None = None


class GeneratedArtifact(BaseModel):
    format: ArtifactFormat
    status: ArtifactStatus
    path: str | None = None
    checksum_sha256: str | None = Field(None, alias="checksum_sha256")
    quantization_method: str | None = None
    calibration_dataset: str | None = None
    bits: int | None = None
    group_size: int | None = None
    backend_compatibility: list[str] | None = None
    notes: str | None = None
    goz1: Goz1Metadata | None = None


class BackendCompatibility(BaseModel):
    gguf: bool = False
    awq: bool = False
    gptq: bool = False
    myelin_accelerator: bool = False
    goz1: bool = False


class SAAQMetadata(BaseModel):
    routing_entropy: float | None = None
    spike_density: float | None = None
    experiment_id: str | None = None
    route_top1_agreement: float | None = None
    route_top2_agreement: float | None = None
    block_output_cosine: float | None = None
    resid_in_drift: float | None = None


class BenchmarkLinkage(BaseModel):
    """Links a model manifest to a combine-for-AI run.

    Accepts legacy ``nfl_combine_*`` keys for older magere handoff files.
    """

    model_config = ConfigDict(populate_by_name=True)

    combine_run_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("combine_run_id", "nfl_combine_run_id"),
        serialization_alias="combine_run_id",
    )
    combine_config_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("combine_config_path", "nfl_combine_config_path"),
        serialization_alias="combine_config_path",
    )
    grok_ozempic_report_path: str | None = None
    grok_ozempic_experiment_id: str | None = None


class ModelManifest(BaseModel):
    manifest_version: str = "1.0.0"
    model_name: str
    model_family: str | None = None
    source_artifact: SourceArtifact
    generated_artifacts: list[GeneratedArtifact] = []
    backend_compatibility: BackendCompatibility | None = None
    saaq_metadata: SAAQMetadata | None = None
    benchmark_linkage: BenchmarkLinkage | None = None


class Goz1HeaderInfo(BaseModel):
    """Lightweight header sniff result (no full table/payload parse)."""

    magic: str = "GOZ1"
    version: int
    tensor_count: int
    meta_count: int
    path: str
    valid: bool = True
    error: str | None = None
    # False when counts declare tables but the file is only a header (truncated pack).
    layout_plausible: bool = True
    file_size: int | None = None


def _parse_raw(content: str, path: Path | None = None) -> Any:
    """Parse JSON or YAML. Prefer YAML for .yml/.yaml; otherwise try JSON then YAML."""
    suffix = path.suffix.lower() if path is not None else ""
    if suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import yaml

        return yaml.safe_load(content)


def load_manifest(path: Path) -> ModelManifest:
    """Load and validate a model manifest from a JSON or YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read()
    raw = _parse_raw(content, path)
    return ModelManifest.model_validate(raw)


def load_manifest_from_string(text: str) -> ModelManifest:
    """Load and validate a model manifest from a JSON (or YAML) string."""
    raw = _parse_raw(text)
    return ModelManifest.model_validate(raw)


def dispatch_artifact(manifest: ModelManifest) -> str:
    """Return a dispatch tag based on generated artifacts, then source format."""
    for gen in manifest.generated_artifacts:
        if gen.status in (ArtifactStatus.SUCCESS, ArtifactStatus.PARTIAL, ArtifactStatus.PLANNED):
            return f"generated_{gen.format.value}"
    source_format = manifest.source_artifact.format
    if source_format == ArtifactFormat.GGUF:
        return "gguf"
    if source_format in (ArtifactFormat.SAFETENSORS, ArtifactFormat.HF):
        return "safetensors_hf"
    if source_format == ArtifactFormat.GOZ1:
        return "goz1"
    return "unknown"


def _goz1_header_fail(
    path: str,
    *,
    error: str,
    version: int = 0,
    tensor_count: int = 0,
    meta_count: int = 0,
    file_size: int | None = None,
) -> Goz1HeaderInfo:
    return Goz1HeaderInfo(
        version=version,
        tensor_count=tensor_count,
        meta_count=meta_count,
        path=path,
        valid=False,
        error=error,
        layout_plausible=False,
        file_size=file_size,
    )


def _validate_goz1_header_fields(
    path_s: str,
    *,
    file_size: int,
    magic_u32: int,
    version: int,
    tensor_count: int,
    meta_count: int,
) -> Goz1HeaderInfo:
    """Validate unpacked GOZ1 header fields (fail-closed)."""
    common = dict(
        version=version,
        tensor_count=tensor_count,
        meta_count=meta_count,
        file_size=file_size,
    )
    if magic_u32 != GOZ1_MAGIC:
        return _goz1_header_fail(
            path_s,
            error=f"bad GOZ1 magic 0x{magic_u32:08x} (expected GOZ1)",
            **common,
        )
    if version not in GOZ1_SUPPORTED_VERSIONS:
        return _goz1_header_fail(
            path_s,
            error=(
                f"unsupported GOZ1 version {version} "
                f"(supported: {sorted(GOZ1_SUPPORTED_VERSIONS)})"
            ),
            **common,
        )
    if (tensor_count > 0 or meta_count > 0) and file_size <= 24:
        return _goz1_header_fail(
            path_s,
            error=(
                "GOZ1 pack truncated: header declares "
                f"tensor_count={tensor_count} meta_count={meta_count} "
                f"but file is only {file_size} bytes"
            ),
            **common,
        )
    return Goz1HeaderInfo(
        path=path_s,
        valid=True,
        error=None,
        layout_plausible=True,
        **common,
    )


def sniff_goz1_header(path: Path) -> Goz1HeaderInfo:
    """Sniff GOZ1 magic/version/counts; reject truncated packs. SoT: grok-ozempic goz1-format."""
    path = Path(path)
    path_s = str(path)
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        return _goz1_header_fail(path_s, error=f"cannot read GOZ1 file: {exc}")
    if len(header) < 24:
        return _goz1_header_fail(
            path_s,
            error=f"GOZ1 header too short ({len(header)} bytes; need 24)",
            file_size=file_size,
        )
    magic_u32, version, tensor_count, meta_count = struct.unpack("<IIQQ", header)
    return _validate_goz1_header_fields(
        path_s,
        file_size=file_size,
        magic_u32=magic_u32,
        version=version,
        tensor_count=tensor_count,
        meta_count=meta_count,
    )


def write_minimal_goz1_fixture(
    path: Path,
    *,
    version: int = 3,
    tensor_count: int = 0,
    meta_count: int = 0,
    pad_body: bool = True,
) -> Path:
    """Write a minimal GOZ1 test file.

    When ``pad_body`` is True and counts are nonzero, append a single padding
    byte so the pack is not rejected as header-only truncated. Real tensor
    tables are not written — this is only for header/selection unit tests.
    """
    if version not in GOZ1_SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported fixture version {version}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<IIQQ", GOZ1_MAGIC, version, tensor_count, meta_count))
        if pad_body and (tensor_count > 0 or meta_count > 0):
            handle.write(b"\x00")
    return path


def default_scale_source_for_version(version: int) -> ScaleSource:
    if version >= 3:
        return ScaleSource.PACK_V3
    if version == 2:
        return ScaleSource.PACK_V2
    return ScaleSource.LEGACY_ORACLE


__all__ = [
    "GOZ1_MAGIC",
    "GOZ1_SUPPORTED_VERSIONS",
    "GOZ1_ROW_SENTINEL",
    "ArtifactFormat",
    "ArtifactStatus",
    "ScaleSource",
    "SourceArtifact",
    "Goz1Metadata",
    "GeneratedArtifact",
    "BackendCompatibility",
    "SAAQMetadata",
    "BenchmarkLinkage",
    "ModelManifest",
    "Goz1HeaderInfo",
    "load_manifest",
    "load_manifest_from_string",
    "dispatch_artifact",
    "sniff_goz1_header",
    "write_minimal_goz1_fixture",
    "default_scale_source_for_version",
    "ValidationError",
]
