from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from combine_for_ai.environment import (
    AcceleratorBackend,
    EnvironmentSnapshot,
    fingerprint_from_snapshot,
)
from combine_for_ai.matrix_fingerprint import (
    MatrixCellResult,
    MatrixReportError,
    aggregate_matrix_report,
    compatibility_warnings,
    render_matrix_markdown,
    write_matrix_reports,
)


def _snapshot(**overrides: object) -> EnvironmentSnapshot:
    values: dict[str, object] = {
        "git_commit": "c" * 40,
        "git_dirty": False,
        "git_porcelain": None,
        "git_diff": None,
        "python_version": "3.14.0",
        "python_implementation": "CPython",
        "lock_kind": "uv.lock",
        "lock_bytes": b"version = 1\n",
        "os_system": "Linux",
        "os_release": "6.12.0-generic",
        "os_machine": "x86_64",
        "os_processor": None,
        "accelerator_backend": AcceleratorBackend.CPU,
        "accelerator_devices": (),
        "driver_version": None,
        "runtime_version": None,
        "home": "/home/alice",
        "username": "alice",
        "resolved_config": {"seed": 1, "quantization": ["fp16"]},
    }
    values.update(overrides)
    return EnvironmentSnapshot(**values)  # type: ignore[arg-type]


def _cell(
    cell_id: str,
    snapshot: EnvironmentSnapshot,
    *,
    model: str = "toy",
    quantization: str = "fp16",
    dataset: str = "lambada",
) -> MatrixCellResult:
    return MatrixCellResult(
        cell_id=cell_id,
        model=model,
        quantization=quantization,
        dataset=dataset,
        fingerprint=fingerprint_from_snapshot(snapshot),
        seed=1,
        metrics={"accuracy": 0.9},
    )


def test_cell_payload_embeds_versioned_fingerprint_and_digest() -> None:
    cell = _cell("toy/fp16/lambada", _snapshot())
    payload = cell.to_payload()
    assert payload["schema"] == "combine_for_ai.matrix_cell.v1"
    assert payload["fingerprint"]["schema"] == "combine_for_ai.environment_fingerprint.v1"
    assert payload["fingerprint"]["version"] == 1
    assert payload["fingerprint_digest"] == cell.fingerprint.digest()
    assert len(payload["fingerprint_digest"]) == 64


def test_matching_fingerprints_do_not_warn() -> None:
    snapshot = _snapshot()
    cells = [
        _cell("toy/fp16/lambada", snapshot),
        _cell("toy/saaq/lambada", snapshot, quantization="saaq"),
    ]
    report = aggregate_matrix_report(cells, matrix_id="compatible")
    assert report["compatible"] is True
    assert report["compatibility_warnings"] == []
    assert report["fingerprint_digests"] == [cells[0].fingerprint.digest()]
    markdown = render_matrix_markdown(report)
    assert "All cells share fingerprint digest" in markdown
    assert "WARNING" not in markdown


def test_drift_emits_concise_compatibility_warning() -> None:
    cpu_snapshot: EnvironmentSnapshot = _snapshot()
    cuda_snapshot: EnvironmentSnapshot = replace(
        cpu_snapshot,
        accelerator_backend=AcceleratorBackend.CUDA,
        accelerator_devices=("NVIDIA GeForce RTX 4090",),
        driver_version="545.23.08",
        runtime_version="12.4",
    )
    dirty_snapshot: EnvironmentSnapshot = replace(
        cpu_snapshot, git_commit="d" * 40, git_dirty=True
    )
    cells = [
        _cell("toy/fp16/lambada", cpu_snapshot),
        _cell("toy/saaq/lambada", cuda_snapshot, quantization="saaq"),
        _cell("toy/fp16/piqa", dirty_snapshot, dataset="piqa"),
    ]
    report = aggregate_matrix_report(cells, matrix_id="drift")
    assert report["compatible"] is False
    assert len(report["fingerprint_digests"]) == 3
    warnings = compatibility_warnings(cells)
    assert warnings[0].startswith("WARNING: environment fingerprints differ")
    assert "will not be pooled as comparable" in warnings[0]
    joined = "\n".join(warnings)
    assert "accelerator.backend" in joined
    assert "repository.commit" in joined
    markdown = render_matrix_markdown(report)
    assert "WARNING: environment fingerprints differ" in markdown
    assert "silently" not in markdown.lower() or "not be pooled" in markdown


def test_aggregate_references_digests_not_just_inline_fingerprints() -> None:
    cell = _cell("toy/fp16/lambada", _snapshot())
    report = aggregate_matrix_report([cell], matrix_id="single")
    assert report["fingerprint_digests"] == [cell.fingerprint.digest()]
    assert report["cells"][0]["fingerprint_digest"] == cell.fingerprint.digest()


def test_write_matrix_reports_rejects_unsafe_run_id(tmp_path: Path) -> None:
    report = aggregate_matrix_report(
        [_cell("toy/fp16/lambada", _snapshot())],
        matrix_id="demo",
    )
    with pytest.raises(MatrixReportError, match="run_id"):
        write_matrix_reports(report, tmp_path, run_id="../escaped")
    with pytest.raises(MatrixReportError, match="run_id"):
        write_matrix_reports(report, tmp_path, run_id="nested/id")
    assert list(tmp_path.rglob("*")) == []


def test_write_matrix_reports_rejects_unknown_formats(tmp_path: Path) -> None:
    report = aggregate_matrix_report(
        [_cell("toy/fp16/lambada", _snapshot())],
        matrix_id="demo",
    )
    with pytest.raises(MatrixReportError, match="unsupported report formats"):
        write_matrix_reports(report, tmp_path, run_id="demo-run", formats=["pdf"])
    with pytest.raises(MatrixReportError, match="no report formats"):
        write_matrix_reports(report, tmp_path, run_id="demo-run", formats=[])


def test_write_matrix_reports_json_and_markdown(tmp_path: Path) -> None:
    cpu_snapshot: EnvironmentSnapshot = _snapshot()
    other_snapshot: EnvironmentSnapshot = replace(cpu_snapshot, lock_bytes=b"version = 2\n")
    report = aggregate_matrix_report(
        [
            _cell("a/fp16/lambada", cpu_snapshot),
            _cell("b/fp16/lambada", other_snapshot, model="b"),
        ],
        matrix_id="demo",
    )
    written = write_matrix_reports(report, tmp_path, run_id="demo-run")
    json_payload = json.loads(written["json"].read_text(encoding="utf-8"))
    assert json_payload["compatible"] is False
    assert json_payload["compatibility_warnings"]
    markdown = written["markdown"].read_text(encoding="utf-8")
    assert markdown.startswith("# Matrix aggregate `demo`")
    assert "WARNING: environment fingerprints differ" in markdown
