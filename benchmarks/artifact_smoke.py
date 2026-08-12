from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.datasets import DatasetSpec, JsonlDatasetLoader
from benchmarks.metrics import MetricsAccumulator
from benchmarks.models import ModelSpec, build_model_adapter, default_quantization_registry
from benchmarks.reporting import write_csv, write_json
from benchmarks.runner import build_metadata
from nfl_combine_for_ai.manifest import (
    ArtifactFormat,
    ArtifactStatus,
    GeneratedArtifact,
    ModelManifest,
    default_scale_source_for_version,
    load_manifest,
    sniff_goz1_header,
)


@dataclass(frozen=True)
class ArtifactSelection:
    status: str
    source_format: str
    runtime_format: str | None
    quantization_name: str | None
    generated_format: str | None
    artifact_path: str | None
    failure_reason: str | None = None


def _resolve_path(base_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


_SUPPORTED_SMOKE_QUANTS = frozenset({"fp16", "awq", "gptq", "gguf", "ternary", "saaq"})


def _quantization_for_goz1(artifact: GeneratedArtifact | None) -> str | None:
    """Map GOZ1 manifest quantization metadata to a registry profile name.

    Returns None when the declared method is present but not supported.
    """
    method = (artifact.quantization_method if artifact else None) or ""
    method = method.strip().lower()
    if method in ("", "saaq", "saaq_v1", "saaq1", "saaq_1_5", "saaq1.5"):
        return "saaq"
    if method in ("ternary", "ternary_snn", "trit"):
        return "ternary"
    if method in _SUPPORTED_SMOKE_QUANTS:
        return method
    return None


def _quantization_for_generated(
    fmt: ArtifactFormat, artifact: GeneratedArtifact | None = None
) -> str | None:
    if fmt == ArtifactFormat.GOZ1:
        return _quantization_for_goz1(artifact)
    return fmt.value


def _goz1_failed_selection(
    *,
    source_format: str,
    artifact_path: str | None,
    failure_reason: str,
    quantization_name: str | None = "saaq",
) -> ArtifactSelection:
    return ArtifactSelection(
        status="failed",
        source_format=source_format,
        runtime_format="generated_goz1",
        quantization_name=quantization_name,
        generated_format="goz1",
        artifact_path=artifact_path,
        failure_reason=failure_reason,
    )


def _select_goz1_generated(
    artifact: GeneratedArtifact,
    resolved: Path,
    source_format: str,
) -> ArtifactSelection:
    """Validate one GOZ1 generated artifact path and return success or failure."""
    quant = _quantization_for_goz1(artifact)
    if quant is None:
        return _goz1_failed_selection(
            source_format=source_format,
            artifact_path=str(resolved),
            failure_reason=(
                "unsupported GOZ1 quantization_method: "
                f"{artifact.quantization_method!r}"
            ),
            quantization_name=None,
        )
    header = sniff_goz1_header(resolved)
    if not header.valid:
        return _goz1_failed_selection(
            source_format=source_format,
            artifact_path=str(resolved),
            failure_reason=header.error or "invalid GOZ1 header",
            quantization_name=quant,
        )
    return ArtifactSelection(
        status="success",
        source_format=source_format,
        runtime_format="generated_goz1",
        quantization_name=quant,
        generated_format="goz1",
        artifact_path=str(resolved),
    )


def _runnable_generated_artifact(
    base_path: Path,
    generated: list[GeneratedArtifact],
    source_format: ArtifactFormat | None = None,
    preferred_formats: tuple[ArtifactFormat, ...] = (
        ArtifactFormat.GOZ1,
        ArtifactFormat.AWQ,
        ArtifactFormat.GPTQ,
    ),
) -> ArtifactSelection | None:
    """Prefer GOZ1, then AWQ/GPTQ, when status is success/partial and path exists.

    A claimed GOZ1 success/partial pack fails closed: missing path, bad header,
    truncated layout, or unsupported quantization_method never falls through to
    AWQ/GPTQ or HF. AWQ/GPTQ still skip missing paths so planned quants can fall
    back to source formats when no GOZ1 was claimed.
    """
    goz1_failure: ArtifactSelection | None = None
    src = source_format.value if source_format else "unknown"

    for fmt in preferred_formats:
        for artifact in generated:
            if artifact.status not in (ArtifactStatus.SUCCESS, ArtifactStatus.PARTIAL):
                continue
            if artifact.format != fmt:
                continue
            resolved = _resolve_path(base_path, artifact.path)
            if not resolved or not resolved.exists() or not resolved.is_file():
                if fmt == ArtifactFormat.GOZ1:
                    goz1_failure = _goz1_failed_selection(
                        source_format=src,
                        artifact_path=str(resolved) if resolved else artifact.path,
                        failure_reason=(
                            "GOZ1 generated artifact path does not exist or is not a file: "
                            f"{artifact.path!r}"
                        ),
                        quantization_name=_quantization_for_goz1(artifact) or "saaq",
                    )
                continue
            if fmt == ArtifactFormat.GOZ1:
                return _select_goz1_generated(artifact, resolved, src)

            return ArtifactSelection(
                status="success",
                source_format=src,
                runtime_format=f"generated_{artifact.format.value}",
                quantization_name=_quantization_for_generated(artifact.format, artifact),
                generated_format=artifact.format.value,
                artifact_path=str(resolved),
            )

        # After scanning all GOZ1 entries, do not fall through to AWQ/GPTQ.
        if fmt == ArtifactFormat.GOZ1 and goz1_failure is not None:
            return goz1_failure

    return goz1_failure


def select_artifact_for_smoke(manifest: ModelManifest, base_path: Path) -> ArtifactSelection:
    generated = _runnable_generated_artifact(
        base_path, manifest.generated_artifacts, manifest.source_artifact.format
    )
    if generated is not None:
        return generated

    source = manifest.source_artifact
    source_path = _resolve_path(base_path, source.path)

    if source.format == ArtifactFormat.GOZ1:
        if source_path and source_path.is_file():
            header = sniff_goz1_header(source_path)
            if not header.valid:
                return ArtifactSelection(
                    status="failed",
                    source_format=source.format.value,
                    runtime_format="goz1",
                    quantization_name="saaq",
                    generated_format=None,
                    artifact_path=str(source_path),
                    failure_reason=header.error or "invalid GOZ1 header",
                )
            return ArtifactSelection(
                status="success",
                source_format=source.format.value,
                runtime_format="goz1",
                quantization_name="saaq",
                generated_format=None,
                artifact_path=str(source_path),
            )
        return ArtifactSelection(
            status="failed",
            source_format=source.format.value,
            runtime_format=None,
            quantization_name=None,
            generated_format=None,
            artifact_path=str(source_path) if source_path else None,
            failure_reason="GOZ1 source artifact path does not exist",
        )

    if source.format == ArtifactFormat.GGUF:
        if source_path and source_path.is_file():
            return ArtifactSelection(
                status="success",
                source_format=source.format.value,
                runtime_format="gguf",
                quantization_name="gguf",
                generated_format=None,
                artifact_path=str(source_path),
            )
        return ArtifactSelection(
            status="failed",
            source_format=source.format.value,
            runtime_format=None,
            quantization_name=None,
            generated_format=None,
            artifact_path=str(source_path) if source_path else None,
            failure_reason="GGUF source artifact path does not exist",
        )

    if source.format in (ArtifactFormat.SAFETENSORS, ArtifactFormat.HF):
        local_exists = source_path and source_path.exists()
        if local_exists or source.hf_repo_id:
            return ArtifactSelection(
                status="success",
                source_format=source.format.value,
                runtime_format="safetensors_hf",
                quantization_name="fp16",
                generated_format=None,
                artifact_path=str(source_path) if local_exists else None,
            )
        return ArtifactSelection(
            status="failed",
            source_format=source.format.value,
            runtime_format=None,
            quantization_name=None,
            generated_format=None,
            artifact_path=str(source_path) if source_path else None,
            failure_reason="HF/Safetensors source is not loadable: missing local path and hf_repo_id",
        )

    return ArtifactSelection(
        status="failed",
        source_format=source.format.value,
        runtime_format=None,
        quantization_name=None,
        generated_format=None,
        artifact_path=str(source_path) if source_path else None,
        failure_reason=f"Unsupported source format for smoke run: {source.format.value}",
    )


def _default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "datasets" / "lambada.sample.jsonl"


def _resolve_dataset_path(manifest_path: Path, dataset_path: Path | None) -> tuple[Path, str]:
    if dataset_path is None:
        return _default_dataset_path(), "lambada-smoke"

    if dataset_path.is_absolute():
        return dataset_path, dataset_path.stem

    return (manifest_path.parent / dataset_path).resolve(), dataset_path.stem


def _load_smoke_dataset(dataset_path: Path, max_samples: int, dataset_name: str) -> tuple[str, list[Any]]:
    loader = JsonlDatasetLoader()
    spec = DatasetSpec(
        name=dataset_name,
        source="jsonl",
        path=str(dataset_path),
        split="validation",
        max_samples=max_samples,
    )
    loaded = loader.load(spec)
    return loaded.spec.name, loaded.records


def _peak_vram_gb_from_metadata(metadata_vram_gb: float, gpu_memory_mb: list[int | None]) -> float | None:
    return metadata_vram_gb


def _goz1_report_fields(selection: ArtifactSelection) -> dict[str, Any]:
    """Attach GOZ1 header metadata to smoke report rows when applicable."""
    path = selection.artifact_path
    is_goz1 = selection.generated_format == "goz1" or selection.runtime_format in {
        "goz1",
        "generated_goz1",
    }
    if not is_goz1 or not path:
        return {}
    header = sniff_goz1_header(Path(path))
    if not header.valid:
        return {
            "goz1_version": header.version or None,
            "goz1_tensor_count": None,
            "goz1_meta_count": None,
            "goz1_scale_source": None,
            "goz1_header_error": header.error,
        }
    return {
        "goz1_version": header.version,
        "goz1_tensor_count": header.tensor_count,
        "goz1_meta_count": header.meta_count,
        "goz1_scale_source": default_scale_source_for_version(header.version).value,
        "goz1_header_error": None,
    }


def _build_failure_payload(
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
    dataset_name: str,
) -> dict[str, Any]:
    row = _build_failure_row(metadata, manifest, manifest_path, selection, dataset_name=dataset_name)
    return {
        "run": {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "timestamp": metadata.timestamp,
        },
        "result": row,
    }


def _build_success_row(
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
    profile: Any,
    dataset_name: str,
    sample_count: int,
    perplexity: float,
) -> dict[str, Any]:
    gpu_names = metadata.telemetry.system.gpu_names or []
    gpu_metrics = metadata.telemetry.gpu_metrics or []
    peak_vram_gb = _peak_vram_gb_from_metadata(
        profile.vram_gb,
        [metric.memory_used_mb for metric in gpu_metrics],
    )
    return {
        "run_id": metadata.run_id,
        "status": "success",
        "timestamp": metadata.timestamp,
        "model_id": manifest.model_name,
        "model_family": manifest.model_family,
        "manifest_path": str(manifest_path),
        "source_format": selection.source_format,
        "runtime_format": selection.runtime_format,
        "generated_format": selection.generated_format,
        "quantization": selection.quantization_name,
        "artifact_path": selection.artifact_path,
        "gpu": gpu_names[0] if gpu_names else None,
        "cuda": metadata.telemetry.system.cuda_version,
        "throughput_tps": profile.speed_tps,
        "generation_tokens": 512,
        "peak_vram_gb": peak_vram_gb,
        "perplexity": perplexity,
        "dataset": dataset_name,
        "sample_count": sample_count,
        "failure_reason": None,
    }


def _build_failure_row(
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
    dataset_name: str,
) -> dict[str, Any]:
    gpu_names = metadata.telemetry.system.gpu_names or []
    return {
        "run_id": metadata.run_id,
        "status": "failed",
        "timestamp": metadata.timestamp,
        "model_id": manifest.model_name,
        "model_family": manifest.model_family,
        "manifest_path": str(manifest_path),
        "source_format": selection.source_format,
        "runtime_format": selection.runtime_format,
        "generated_format": selection.generated_format,
        "quantization": selection.quantization_name,
        "artifact_path": selection.artifact_path,
        "gpu": gpu_names[0] if gpu_names else None,
        "cuda": metadata.telemetry.system.cuda_version,
        "throughput_tps": None,
        "generation_tokens": 512,
        "peak_vram_gb": None,
        "perplexity": None,
        "dataset": dataset_name,
        "sample_count": 0,
        "failure_reason": selection.failure_reason,
    }


def _fallback_run_payload(manifest_path: Path, failure_reason: str) -> dict[str, Any]:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{int(time.time() % 1 * 1000):03d}Z-artifact-smoke-failed"
    row = {
        "run_id": run_id,
        "status": "failed",
        "timestamp": timestamp,
        "model_id": None,
        "model_family": None,
        "manifest_path": str(manifest_path),
        "source_format": None,
        "runtime_format": None,
        "generated_format": None,
        "quantization": None,
        "artifact_path": None,
        "gpu": None,
        "cuda": None,
        "throughput_tps": None,
        "generation_tokens": 512,
        "peak_vram_gb": None,
        "perplexity": None,
        "dataset": "lambada-smoke",
        "sample_count": 0,
        "failure_reason": failure_reason,
    }
    return {
        "run": {
            "run_id": run_id,
            "run_name": "artifact-smoke-failed",
            "timestamp": timestamp,
        },
        "result": row,
    }


def _write_fallback_failure_outputs(
    output_dir: Path,
    formats: list[str],
    manifest_path: Path,
    failure_reason: str,
) -> dict[str, Any]:
    payload = _fallback_run_payload(manifest_path, failure_reason)
    row = payload["result"]
    run_id = payload["run"]["run_id"]
    if "json" in formats:
        write_json(output_dir / "json" / f"{run_id}.artifact-smoke.json", payload)
    if "csv" in formats:
        write_csv(output_dir / "csv" / f"{run_id}.artifact-smoke.csv", [row])
    return payload


def _write_failure_outputs(
    output_dir: Path,
    formats: list[str],
    metadata: Any,
    manifest: ModelManifest,
    manifest_path: Path,
    selection: ArtifactSelection,
    dataset_name: str = "lambada-smoke",
) -> dict[str, Any]:
    row = _build_failure_row(
        metadata,
        manifest,
        manifest_path,
        selection,
        dataset_name=dataset_name,
    )
    payload = {
        "run": {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "timestamp": metadata.timestamp,
        },
        "result": row,
    }
    run_id = metadata.run_id
    if "json" in formats:
        write_json(output_dir / "json" / f"{run_id}.artifact-smoke.json", payload)
    if "csv" in formats:
        write_csv(output_dir / "csv" / f"{run_id}.artifact-smoke.csv", [row])
    return payload


def run_artifact_smoke(
    manifest_path: Path,
    output_dir: Path,
    formats: list[str],
    dataset_path: Path | None = None,
    max_samples: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        manifest = load_manifest(manifest_path)
        metadata = build_metadata(f"artifact-smoke-{manifest.model_name}", seed)
        selection = select_artifact_for_smoke(manifest, manifest_path.parent)
    except Exception as exc:
        return _write_fallback_failure_outputs(output_dir, formats, manifest_path, str(exc))

    if selection.status != "success":
        return _write_failure_outputs(output_dir, formats, metadata, manifest, manifest_path, selection)

    dataset_name = "lambada-smoke"
    try:
        dataset, dataset_name = _resolve_dataset_path(manifest_path, dataset_path)
        _, records = _load_smoke_dataset(dataset, max_samples, dataset_name)
        if not records:
            raise ValueError(f"No records loaded from dataset: {dataset}")
        profile = default_quantization_registry().get(selection.quantization_name or "fp16")
        if not profile.supported:
            raise ValueError(f"Unsupported quantization profile: {selection.quantization_name}")
        adapter = build_model_adapter(
            ModelSpec(backend="mock", name=manifest.model_name, revision=manifest.source_artifact.hf_revision),
            profile,
        )

        accumulator = MetricsAccumulator()

        rng = random.Random(seed)
        for record in records:
            prediction = adapter.predict(record, rng)
            accumulator.add(record, prediction)

        total_time = accumulator.token_count / profile.speed_tps if profile.speed_tps else 0.0
        metrics = accumulator.summary(total_time, profile.vram_gb)
        row = _build_success_row(
            metadata,
            manifest,
            manifest_path,
            selection,
            profile,
            dataset_name,
            len(records),
            metrics.perplexity,
        )
        goz1_fields = _goz1_report_fields(selection)
        if goz1_fields:
            row.update(goz1_fields)
        payload = {
            "run": {
                "run_id": metadata.run_id,
                "run_name": metadata.run_name,
                "timestamp": metadata.timestamp,
                "gpu": metadata.telemetry.system.gpu_names,
                "cuda": metadata.telemetry.system.cuda_version,
            },
            "result": row,
        }

        run_id = metadata.run_id
        if "json" in formats:
            write_json(output_dir / "json" / f"{run_id}.artifact-smoke.json", payload)
        if "csv" in formats:
            write_csv(output_dir / "csv" / f"{run_id}.artifact-smoke.csv", [row])
        return payload
    except Exception as exc:
        failed = ArtifactSelection(
            status="failed",
            source_format=selection.source_format,
            runtime_format=selection.runtime_format,
            quantization_name=selection.quantization_name,
            generated_format=selection.generated_format,
            artifact_path=selection.artifact_path,
            failure_reason=str(exc),
        )
        return _write_failure_outputs(
            output_dir,
            formats,
            metadata,
            manifest,
            manifest_path,
            failed,
            dataset_name=dataset_name,
        )
