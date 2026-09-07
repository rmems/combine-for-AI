from __future__ import annotations

import json
import os
import platform
import random
import secrets
import shutil
import subprocess
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.datasets import DatasetSpec, LoadedDataset, default_dataset_registry
from benchmarks.metrics import MetricsAccumulator, MetricsSummary
from benchmarks.models import (
    ModelSpec,
    QuantizationProfile,
    build_model_adapter,
    default_quantization_registry,
    scoped_seed,
)
from benchmarks.reporting import metrics_to_row, telemetry_to_row, write_csv, write_json
from benchmarks.telemetry import (
    CorinthCanalArtifact,
    MyelinAcceleratorArtifact,
    TelemetrySnapshot,
    collect_telemetry_snapshot,
    merge_upstream_artifacts,
    telemetry_to_dict,
    write_telemetry_json,
)


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    run_name: str
    seed: int
    git_commit: str
    git_branch: str
    host: str
    os: str
    cpu: str
    cpu_count: int | None
    timestamp: str
    telemetry: TelemetrySnapshot


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    split: str
    sample_count: int
    quantization: QuantizationProfile
    metrics: MetricsSummary
    telemetry: TelemetrySnapshot | None = None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


#: Recorded for commit and branch when the git provenance cannot be read.
UNKNOWN_GIT_INFO = "unknown"

_GIT_TIMEOUT_SECONDS = 5


def _git(*args: str) -> str | None:
    """Run a read-only git command, or return None if it cannot be answered.

    Provenance is metadata about the run, not a precondition for it: a
    benchmark launched from an installed wheel, a source tarball or a container
    layer without a `.git` directory should still produce a report.
    """
    # Resolved to an absolute path rather than letting the OS search PATH, so a
    # `git` planted earlier in PATH cannot be what a benchmark run executes.
    git = shutil.which("git")
    if git is None:
        return None
    try:
        output = subprocess.check_output(  # nosec B603 - fixed argv, no shell
            [git, *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        # Not a repository, or a hung invocation.
        return None
    stripped = output.strip()
    return stripped or None


def get_git_info() -> tuple[str, str]:
    """Return (commit, branch), falling back to "unknown" for either."""
    commit = _git("rev-parse", "HEAD") or UNKNOWN_GIT_INFO
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or UNKNOWN_GIT_INFO
    return commit, branch


def _build_run_id(commit: str) -> str:
    """Build a run id unique across concurrent runs of the same commit.

    The commit prefix alone does not separate two runs, and the timestamp
    repeats when parallel jobs start in the same millisecond. Colliding ids
    resolve to the same JSON, CSV and telemetry paths, so the later run would
    silently overwrite the earlier one's reports. Both halves of the timestamp
    come from a single clock reading so they cannot straddle a second.
    """
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now))
    millis = int(now % 1 * 1000)
    return f"{stamp}.{millis:03d}Z-{commit[:7]}-{secrets.token_hex(4)}"


def build_metadata(run_name: str, seed: int) -> RunMetadata:
    commit, branch = get_git_info()
    telemetry = collect_telemetry_snapshot()
    return RunMetadata(
        run_id=_build_run_id(commit),
        run_name=run_name,
        seed=seed,
        git_commit=commit,
        git_branch=branch,
        host=platform.node(),
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        telemetry=telemetry,
    )


def load_datasets(config: dict[str, Any], base_path: Path) -> list[LoadedDataset]:
    registry = default_dataset_registry()
    datasets = []
    for raw in config["datasets"]:
        spec = DatasetSpec.from_dict(raw)
        if spec.path:
            path = Path(spec.path)
            if not path.is_absolute():
                spec = replace(spec, path=str((base_path / path).resolve()))
        loader = registry.loader_for(spec.source)
        datasets.append(loader.load(spec))
    return datasets


def _load_upstream_telemetry(
    config: dict[str, Any],
    base_path: Path,
    telemetry: TelemetrySnapshot,
) -> TelemetrySnapshot:
    """Load optional upstream telemetry artifacts referenced by the config."""
    corinth = None
    myelin = None

    telemetry_cfg = config.get("telemetry") or {}
    corinth_path = telemetry_cfg.get("corinth_canal_path")
    if corinth_path:
        corinth = CorinthCanalArtifact.from_file(base_path / corinth_path)

    myelin_path = telemetry_cfg.get("myelin_accelerator_path")
    if myelin_path:
        myelin = MyelinAcceleratorArtifact.from_file(base_path / myelin_path)

    return merge_upstream_artifacts(telemetry, corinth=corinth, myelin=myelin)


def run_benchmarks(
    config_path: Path,
    output_dir: Path,
    formats: list[str],
    seed_override: int | None = None,
) -> RunMetadata:
    config = load_config(config_path)
    run_name = config.get("run_name", "benchmark-run")
    seed = seed_override if seed_override is not None else config.get("seed", 0)
    metadata = build_metadata(run_name, seed)

    # Optionally enrich telemetry with upstream artifacts
    telemetry = _load_upstream_telemetry(
        config,
        config_path.parent,
        metadata.telemetry,
    )
    metadata = RunMetadata(
        run_id=metadata.run_id,
        run_name=metadata.run_name,
        seed=metadata.seed,
        git_commit=metadata.git_commit,
        git_branch=metadata.git_branch,
        host=metadata.host,
        os=metadata.os,
        cpu=metadata.cpu,
        cpu_count=metadata.cpu_count,
        timestamp=metadata.timestamp,
        telemetry=telemetry,
    )

    datasets = load_datasets(config, config_path.parent)
    model_spec = ModelSpec.from_dict(config["model"])
    quantization_names = config.get("quantization") or ["fp16"]

    registry = default_quantization_registry()
    results: list[DatasetResult] = []

    for quant_name in quantization_names:
        profile = registry.get(quant_name)
        adapter = build_model_adapter(model_spec, profile)

        for dataset in datasets:
            scoped = scoped_seed(seed, model_spec.name, quant_name, dataset.spec.name)
            rng = random.Random(scoped)
            accumulator = MetricsAccumulator()

            for record in dataset.records:
                prediction = adapter.predict(record, rng)
                accumulator.add(record, prediction)

            total_time = (
                accumulator.token_count / profile.speed_tps
                if profile.speed_tps
                else 0.0
            )
            metrics = accumulator.summary(total_time, profile.vram_gb)
            results.append(
                DatasetResult(
                    dataset=dataset.spec.name,
                    split=dataset.spec.split,
                    sample_count=len(dataset.records),
                    quantization=profile,
                    metrics=metrics,
                    telemetry=telemetry,
                )
            )

    write_reports(output_dir, formats, metadata, model_spec, results)
    return metadata


def write_reports(
    output_dir: Path,
    formats: list[str],
    metadata: RunMetadata,
    model_spec: ModelSpec,
    results: list[DatasetResult],
) -> None:
    rows = []
    for result in results:
        row = {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "seed": metadata.seed,
            "git_commit": metadata.git_commit,
            "git_branch": metadata.git_branch,
            "host": metadata.host,
            "os": metadata.os,
            "cpu": metadata.cpu,
            "cpu_count": metadata.cpu_count,
            "timestamp": metadata.timestamp,
            "model_name": model_spec.name,
            "model_backend": model_spec.backend,
            "model_revision": model_spec.revision,
            "dataset": result.dataset,
            "split": result.split,
            "sample_count": result.sample_count,
            "quantization": result.quantization.name,
            "precision": result.quantization.precision,
            "quantization_format": result.quantization.format,
            "quantization_bits": result.quantization.bits,
            **metrics_to_row(result.metrics),
            **telemetry_to_row(result.telemetry),
        }
        rows.append(row)

    payload = {
        "run": {
            "run_id": metadata.run_id,
            "run_name": metadata.run_name,
            "seed": metadata.seed,
            "git_commit": metadata.git_commit,
            "git_branch": metadata.git_branch,
            "host": metadata.host,
            "os": metadata.os,
            "cpu": metadata.cpu,
            "cpu_count": metadata.cpu_count,
            "timestamp": metadata.timestamp,
            "model": {
                "name": model_spec.name,
                "backend": model_spec.backend,
                "revision": model_spec.revision,
            },
            "telemetry": telemetry_to_dict(metadata.telemetry),
        },
        "results": rows,
    }

    run_id = metadata.run_id
    if "json" in formats:
        write_json(output_dir / "json" / f"{run_id}.json", payload)
    if "csv" in formats:
        write_csv(output_dir / "csv" / f"{run_id}.csv", rows)
    write_telemetry_json(
        output_dir / "telemetry" / f"{run_id}.telemetry.json",
        metadata.telemetry,
    )
