from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import subprocess

from combine_for_ai.environment_probe import _apply_cuda_visibility, git_rev_parse_head
from combine_for_ai.environment import (
    REDACTED,
    REDACTED_USER,
    AcceleratorBackend,
    EnvironmentSnapshot,
    canonical_json_bytes,
    collect_environment_fingerprint,
    differing_material_fields,
    digest_resolved_config,
    fingerprint_from_snapshot,
    redact_text,
    sanitize_mapping,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _cpu_snapshot(**overrides: object) -> EnvironmentSnapshot:
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
        "driver_version": "545.23.08",
        "runtime_version": "12.4",
        "home": "/home/alice",
        "username": "alice",
        "resolved_config": {
            "model": {"backend": "mock", "name": "toy-model"},
            "quantization": ["fp16"],
            "seed": 1,
        },
    }
    values.update(overrides)
    return EnvironmentSnapshot(**values)  # type: ignore[arg-type]


def _cuda_snapshot(**overrides: object) -> EnvironmentSnapshot:
    return _cpu_snapshot(
        accelerator_backend=AcceleratorBackend.CUDA,
        accelerator_devices=("NVIDIA GeForce RTX 4090",),
        driver_version="545.23.08",
        runtime_version="12.4",
        **overrides,
    )


def test_equivalent_snapshots_are_byte_identical() -> None:
    left = fingerprint_from_snapshot(_cpu_snapshot())
    right = fingerprint_from_snapshot(_cpu_snapshot())
    assert left.canonical_json() == right.canonical_json()
    assert left.digest() == right.digest()
    assert left.canonical_json() == canonical_json_bytes(left.to_canonical_dict())


def test_canonical_json_has_deterministic_key_order() -> None:
    fingerprint = fingerprint_from_snapshot(
        _cpu_snapshot(
            resolved_config={"z": 1, "a": {"y": 2, "b": 3}, "m": [3, 1, 2]},
        )
    )
    text = fingerprint.canonical_json().decode("ascii")
    payload = fingerprint.to_canonical_dict()
    assert list(payload) == sorted(payload)
    assert list(payload["accelerator"]) == sorted(payload["accelerator"])
    assert list(payload["os"]) == sorted(payload["os"])
    assert list(payload["python"]) == sorted(payload["python"])
    assert list(payload["repository"]) == sorted(payload["repository"])
    assert list(payload["dependencies"]) == sorted(payload["dependencies"])
    assert text == canonical_json_bytes(payload).decode("ascii")
    assert text.index('"accelerator"') < text.index('"config_digest"')
    assert text.index('"devices":[]') < text.index('"driver_version":null')
    assert text.index('"implementation"') < text.index('"version"')


def test_resolved_config_key_insertion_order_does_not_change_digest() -> None:
    first = fingerprint_from_snapshot(
        _cpu_snapshot(resolved_config={"b": 1, "a": 2})
    )
    second = fingerprint_from_snapshot(
        _cpu_snapshot(resolved_config={"a": 2, "b": 1})
    )
    assert first.config_digest == second.config_digest
    assert first.digest() == second.digest()


def test_material_drift_changes_digest() -> None:
    baseline = fingerprint_from_snapshot(_cpu_snapshot())
    mutated = {
        "commit": fingerprint_from_snapshot(
            replace(_cpu_snapshot(), git_commit="d" * 40)
        ),
        "lock": fingerprint_from_snapshot(
            replace(_cpu_snapshot(), lock_bytes=b"version = 2\n")
        ),
        "backend": fingerprint_from_snapshot(_cuda_snapshot()),
        "config": fingerprint_from_snapshot(
            _cpu_snapshot(resolved_config={"model": {"name": "other"}})
        ),
    }
    assert mutated["commit"].digest() != baseline.digest()
    assert mutated["lock"].digest() != baseline.digest()
    assert mutated["backend"].digest() != baseline.digest()
    assert mutated["config"].digest() != baseline.digest()
    assert "repository.commit" in differing_material_fields(baseline, mutated["commit"])
    assert "dependencies.digest" in differing_material_fields(baseline, mutated["lock"])
    assert "accelerator.backend" in differing_material_fields(
        baseline, mutated["backend"]
    )
    assert "config_digest" in differing_material_fields(baseline, mutated["config"])


def test_cpu_only_drops_cuda_placeholders() -> None:
    fingerprint = fingerprint_from_snapshot(_cpu_snapshot())
    payload = fingerprint.to_canonical_dict()
    accelerator = payload["accelerator"]
    assert accelerator == {
        "backend": "cpu",
        "devices": [],
        "driver_version": None,
        "runtime_version": None,
    }
    blob = fingerprint.canonical_json().decode("ascii")
    assert "cuda" not in blob.lower()
    assert "nvidia" not in blob.lower()
    assert "n/a" not in blob.lower()
    assert "unknown" not in blob.lower()
    assert "placeholder" not in blob.lower()


def test_cuda_claimed_without_devices_is_cpu() -> None:
    fingerprint = fingerprint_from_snapshot(
        _cpu_snapshot(
            accelerator_backend=AcceleratorBackend.CUDA,
            accelerator_devices=(),
            driver_version="545.23.08",
            runtime_version="12.4",
        )
    )
    accelerator = fingerprint.to_canonical_dict()["accelerator"]
    assert accelerator["backend"] == "cpu"
    assert accelerator["devices"] == []
    assert accelerator["driver_version"] is None
    assert accelerator["runtime_version"] is None


def test_unavailable_gpu_fields_are_null_not_placeholders() -> None:
    fingerprint = fingerprint_from_snapshot(
        _cpu_snapshot(driver_version=None, runtime_version=None)
    )
    accelerator = fingerprint.to_canonical_dict()["accelerator"]
    assert accelerator["driver_version"] is None
    assert accelerator["runtime_version"] is None
    assert accelerator["devices"] == []


def test_redaction_of_secrets_usernames_and_home_paths() -> None:
    config = {
        "hf_token": "hf_abcdefghijklmnopqrstuvwxyz",
        "weights": "/home/alice/models/toy.safetensors",
        "shared": "/opt/alice/cache/weights.bin",
        "api_key": "sk-this-is-not-a-real-secret",
        "pid": 4321,
        "timestamp": "2026-09-15T00:00:00Z",
        "nested": {"password": "hunter2", "name": "ok"},
        "auth_header": "Bearer ghp_abcdefghijklmnopqrstuvwxyz1234",
    }
    sanitized = sanitize_mapping(
        config, home="/home/alice", username="alice"
    )
    assert sanitized["hf_token"] == REDACTED
    assert sanitized["api_key"] == REDACTED
    assert sanitized["nested"]["password"] == REDACTED
    assert sanitized["nested"]["name"] == "ok"
    assert sanitized["weights"] == "~/models/toy.safetensors"
    assert sanitized["shared"] == f"/opt/{REDACTED_USER}/cache/weights.bin"
    assert sanitized["auth_header"] == REDACTED
    assert "pid" not in sanitized
    assert "timestamp" not in sanitized

    fingerprint = fingerprint_from_snapshot(_cpu_snapshot(resolved_config=config))
    blob = fingerprint.canonical_json().decode("ascii")
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in blob
    assert "/home/alice" not in blob
    assert "hunter2" not in blob
    assert "ghp_" not in blob
    assert "sk-this-is-not-a-real-secret" not in blob


def test_redact_text_replaces_home_before_username() -> None:
    text = redact_text(
        "/home/alice/token-file", home="/home/alice", username="alice"
    )
    assert text == "~/token-file"
    assert "alice" not in text


def test_live_probe_succeeds_on_cpu_without_cuda_placeholders() -> None:
    fingerprint = collect_environment_fingerprint(
        {"model": {"backend": "mock", "name": "toy"}, "seed": 0},
        repo_root=Path(__file__).resolve().parents[1],
    )
    payload = fingerprint.to_canonical_dict()
    assert payload["schema"] == "combine_for_ai.environment_fingerprint.v1"
    assert payload["version"] == 1
    assert payload["python"]["version"]
    if payload["accelerator"]["backend"] == "cpu":
        blob = fingerprint.canonical_json().decode("ascii")
        assert '"backend":"cpu"' in blob
        assert payload["accelerator"]["devices"] == []
        assert payload["accelerator"]["driver_version"] is None
        assert payload["accelerator"]["runtime_version"] is None
        assert "n/a" not in blob.lower()


def test_sample_fixtures_match_frozen_snapshots() -> None:
    cpu = fingerprint_from_snapshot(_cpu_snapshot())
    cuda = fingerprint_from_snapshot(_cuda_snapshot())
    cpu_fixture = FIXTURES / "environment_fingerprint_cpu.sample.json"
    cuda_fixture = FIXTURES / "environment_fingerprint_cuda.sample.json"
    assert cpu_fixture.read_bytes().rstrip(b"\r\n") == cpu.canonical_json()
    assert cuda_fixture.read_bytes().rstrip(b"\r\n") == cuda.canonical_json()
    assert cpu.digest() != cuda.digest()


def test_nested_semantic_host_is_kept_in_config_digest() -> None:
    home = "/home/alice"
    username = "alice"
    one = digest_resolved_config(
        {"service": {"host": "one", "port": 9}},
        home=home,
        username=username,
    )
    two = digest_resolved_config(
        {"service": {"host": "two", "port": 9}},
        home=home,
        username=username,
    )
    assert one != two
    sanitized = sanitize_mapping(
        {"service": {"host": "one", "user": "svc"}},
        home=home,
        username=username,
    )
    assert sanitized["service"]["host"] == "one"
    assert sanitized["service"]["user"] == "svc"


def test_dirty_worktrees_with_different_porcelain_differ() -> None:
    clean = fingerprint_from_snapshot(_cpu_snapshot())
    first = fingerprint_from_snapshot(
        _cpu_snapshot(
            git_dirty=True,
            git_porcelain=" M src/combine_for_ai/environment.py",
            git_diff="diff --git a/src/combine_for_ai/environment.py b/src/combine_for_ai/environment.py\n+one",
        )
    )
    second = fingerprint_from_snapshot(
        _cpu_snapshot(
            git_dirty=True,
            git_porcelain=" M src/combine_for_ai/environment.py",
            git_diff="diff --git a/src/combine_for_ai/environment.py b/src/combine_for_ai/environment.py\n+two",
        )
    )
    assert first.digest() != clean.digest()
    assert first.digest() != second.digest()
    assert first.to_canonical_dict()["repository"]["worktree_digest"]
    assert first.to_canonical_dict()["repository"]["worktree_digest"] != (
        second.to_canonical_dict()["repository"]["worktree_digest"]
    )
    assert "repository.worktree_digest" in differing_material_fields(first, second)


def test_cpu_processor_is_material() -> None:
    intel = fingerprint_from_snapshot(_cpu_snapshot(os_processor="Intel Xeon"))
    amd = fingerprint_from_snapshot(_cpu_snapshot(os_processor="AMD EPYC"))
    assert intel.digest() != amd.digest()
    assert "os.processor" in differing_material_fields(intel, amd)


def test_cuda_visibility_filters_enumerated_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = ("GPU-0", "GPU-1", "GPU-2")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _apply_cuda_visibility(devices) == devices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert _apply_cuda_visibility(devices) == ("GPU-2", "GPU-0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _apply_cuda_visibility(devices) == ()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _apply_cuda_visibility(devices) == ()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1")
    assert _apply_cuda_visibility(devices) == ("GPU-1",)


def test_git_probe_invokes_literal_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="deadbeef\n", stderr="")

    monkeypatch.setattr(
        "combine_for_ai.environment_probe.subprocess.run",
        fake_run,
    )
    assert git_rev_parse_head(None) == "deadbeef"
    assert captured == [["git", "rev-parse", "HEAD"]]


def test_digest_resolved_config_ignores_volatile_process_fields() -> None:
    home = "/home/alice"
    username = "alice"
    left = digest_resolved_config(
        {"model": "toy", "pid": 1, "timestamp": "t0"},
        home=home,
        username=username,
    )
    right = digest_resolved_config(
        {"model": "toy", "pid": 2, "timestamp": "t1"},
        home=home,
        username=username,
    )
    assert left == right
