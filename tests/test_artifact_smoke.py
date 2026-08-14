from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmarks.artifact_smoke import run_artifact_smoke, select_artifact_for_smoke
from combine_for_ai.manifest import load_manifest_from_string


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_select_artifact_prefers_existing_generated_awq() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        generated = tmp / "model-awq"
        generated.write_text("ok", encoding="utf-8")
        source = tmp / "source.gguf"
        source.write_text("ok", encoding="utf-8")

        manifest = load_manifest_from_string(
            json.dumps(
                {
                    "model_name": "demo",
                    "source_artifact": {"format": "gguf", "path": str(source)},
                    "generated_artifacts": [
                        {"format": "awq", "status": "success", "path": str(generated)}
                    ],
                }
            )
        )

        selection = select_artifact_for_smoke(manifest, tmp)
        assert selection.status == "success"
        assert selection.runtime_format == "generated_awq"
        assert selection.quantization_name == "awq"


def test_select_artifact_falls_back_to_source_hf() -> None:
    manifest = load_manifest_from_string(
        json.dumps(
            {
                "model_name": "hf-demo",
                "source_artifact": {
                    "format": "safetensors",
                    "hf_repo_id": "org/model",
                },
                "generated_artifacts": [
                    {"format": "awq", "status": "planned", "path": "missing-awq"}
                ],
            }
        )
    )

    selection = select_artifact_for_smoke(manifest, Path.cwd())
    assert selection.status == "success"
    assert selection.runtime_format == "safetensors_hf"
    assert selection.quantization_name == "fp16"


def test_run_artifact_smoke_success_creates_reports() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        artifact = tmp / "model.gguf"
        artifact.write_text("ok", encoding="utf-8")
        manifest_path = tmp / "manifest.json"
        _write_json(
            manifest_path,
            {
                "model_name": "gguf-demo",
                "source_artifact": {"format": "gguf", "path": str(artifact)},
            },
        )
        output_dir = tmp / "reports"

        payload = run_artifact_smoke(manifest_path, output_dir, ["json", "csv"])

        assert payload["result"]["status"] == "success"
        assert payload["result"]["source_format"] == "gguf"
        assert payload["result"]["runtime_format"] == "gguf"
        assert payload["result"]["perplexity"] is not None
        assert list((output_dir / "json").glob("*.artifact-smoke.json"))
        assert list((output_dir / "csv").glob("*.artifact-smoke.csv"))


def test_run_artifact_smoke_failure_creates_structured_failure_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest_path = tmp / "manifest.json"
        _write_json(
            manifest_path,
            {
                "model_name": "missing-gguf-demo",
                "source_artifact": {"format": "gguf", "path": "missing.gguf"},
            },
        )
        output_dir = tmp / "reports"

        payload = run_artifact_smoke(manifest_path, output_dir, ["json", "csv"])

        assert payload["result"]["status"] == "failed"
        assert payload["result"]["failure_reason"] == "GGUF source artifact path does not exist"
        assert list((output_dir / "json").glob("*.artifact-smoke.json"))
        assert list((output_dir / "csv").glob("*.artifact-smoke.csv"))
