"""Deterministic, sanitized environment fingerprints for matrix cells.

Two equivalent sanitized environments must produce byte-identical canonical
JSON and the same digest. Collection is separated from fingerprinting so tests
can freeze the host: pass an ``EnvironmentSnapshot`` instead of probing.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, assert_never

try:
    from pynvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
        nvmlSystemGetDriverVersion,
    )
except ImportError:
    nvmlDeviceGetCount = None
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetName = None
    nvmlInit = None
    nvmlShutdown = None
    nvmlSystemGetDriverVersion = None


FINGERPRINT_SCHEMA = "combine_for_ai.environment_fingerprint.v1"
FINGERPRINT_VERSION = 1
DIGEST_ALGORITHM = "sha256"
REDACTED = "[REDACTED]"
REDACTED_USER = "[USER]"

_SECRET_KEY_RE = re.compile(
    r"(^|[_-])(token|secret|password|passwd|authorization|credential|"
    r"api[_-]?key|private[_-]?key|access[_-]?key|secret[_-]?key)s?$",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])("
    r"sk-[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|Bearer\s+\S+"
    r")(?![A-Za-z0-9])"
)
_TOP_LEVEL_VOLATILE_KEYS = frozenset(
    {
        "pid",
        "ppid",
        "cwd",
        "pwd",
        "timestamp",
        "hostname",
        "host",
        "run_id",
        "process_id",
        "username",
        "user",
    }
)
_LOCK_FILENAMES = ("uv.lock", "poetry.lock", "pdm.lock", "requirements.lock")
_GIT_TIMEOUT_S = 5.0
_ACCEL_TIMEOUT_S = 5.0


class AcceleratorBackend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


@dataclass(frozen=True)
class RepositoryIdentity:
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True)
class PythonIdentity:
    version: str
    implementation: str


@dataclass(frozen=True)
class DependencyLockIdentity:
    kind: str | None
    digest: str | None


@dataclass(frozen=True)
class OsIdentity:
    system: str
    release: str
    machine: str


@dataclass(frozen=True)
class AcceleratorIdentity:
    backend: AcceleratorBackend
    devices: tuple[str, ...]
    driver_version: str | None
    runtime_version: str | None


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Raw probe results. Tests construct this instead of touching the host."""

    git_commit: str | None
    git_dirty: bool | None
    python_version: str
    python_implementation: str
    lock_kind: str | None
    lock_bytes: bytes | None
    os_system: str
    os_release: str
    os_machine: str
    accelerator_backend: AcceleratorBackend
    accelerator_devices: tuple[str, ...]
    driver_version: str | None
    runtime_version: str | None
    home: str
    username: str
    resolved_config: Mapping[str, Any]


@dataclass(frozen=True)
class EnvironmentFingerprint:
    schema: str
    version: int
    repository: RepositoryIdentity
    python: PythonIdentity
    dependencies: DependencyLockIdentity
    os: OsIdentity
    accelerator: AcceleratorIdentity
    config_digest: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "accelerator": {
                "backend": self.accelerator.backend.value,
                "devices": list(self.accelerator.devices),
                "driver_version": self.accelerator.driver_version,
                "runtime_version": self.accelerator.runtime_version,
            },
            "config_digest": self.config_digest,
            "dependencies": {
                "digest": self.dependencies.digest,
                "kind": self.dependencies.kind,
            },
            "os": {
                "machine": self.os.machine,
                "release": self.os.release,
                "system": self.os.system,
            },
            "python": {
                "implementation": self.python.implementation,
                "version": self.python.version,
            },
            "repository": {
                "commit": self.repository.commit,
                "dirty": self.repository.dirty,
            },
            "schema": self.schema,
            "version": self.version,
        }

    def canonical_json(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Stable UTF-8 JSON: sorted keys, compact separators, no NaN/Infinity."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_from_snapshot(snapshot: EnvironmentSnapshot) -> EnvironmentFingerprint:
    """Build a sanitized, versioned fingerprint from a frozen snapshot."""
    home = snapshot.home or ""
    username = snapshot.username or ""
    devices = tuple(
        redact_text(name, home=home, username=username)
        for name in snapshot.accelerator_devices
    )
    backend = _cpu_safe_backend(snapshot.accelerator_backend, devices)
    driver, runtime = _cpu_safe_versions(backend, snapshot.driver_version, snapshot.runtime_version)
    lock_digest = sha256_hex(snapshot.lock_bytes) if snapshot.lock_bytes is not None else None
    config_digest = digest_resolved_config(
        snapshot.resolved_config,
        home=home,
        username=username,
    )
    return EnvironmentFingerprint(
        schema=FINGERPRINT_SCHEMA,
        version=FINGERPRINT_VERSION,
        repository=RepositoryIdentity(
            commit=_sanitize_commit(snapshot.git_commit),
            dirty=snapshot.git_dirty,
        ),
        python=PythonIdentity(
            version=redact_text(snapshot.python_version, home=home, username=username),
            implementation=redact_text(
                snapshot.python_implementation, home=home, username=username
            ),
        ),
        dependencies=DependencyLockIdentity(
            kind=snapshot.lock_kind,
            digest=lock_digest,
        ),
        os=OsIdentity(
            system=redact_text(snapshot.os_system, home=home, username=username),
            release=redact_text(snapshot.os_release, home=home, username=username),
            machine=redact_text(snapshot.os_machine, home=home, username=username),
        ),
        accelerator=AcceleratorIdentity(
            backend=backend,
            devices=devices,
            driver_version=_optional_redact(driver, home=home, username=username),
            runtime_version=_optional_redact(runtime, home=home, username=username),
        ),
        config_digest=config_digest,
    )


def collect_environment_fingerprint(
    resolved_config: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
    snapshot: EnvironmentSnapshot | None = None,
) -> EnvironmentFingerprint:
    """Probe the host (or use ``snapshot``) and return a sanitized fingerprint."""
    if snapshot is None:
        snapshot = probe_environment(resolved_config or {}, repo_root=repo_root)
    return fingerprint_from_snapshot(snapshot)


def probe_environment(
    resolved_config: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> EnvironmentSnapshot:
    root = repo_root if repo_root is not None else _discover_repo_root()
    commit, dirty = _probe_git(root)
    lock_kind, lock_bytes = _probe_lock(root)
    backend, devices, driver, runtime = _probe_accelerator()
    home = str(Path.home())
    username = _probe_username()
    return EnvironmentSnapshot(
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        lock_kind=lock_kind,
        lock_bytes=lock_bytes,
        os_system=platform.system(),
        os_release=platform.release(),
        os_machine=platform.machine(),
        accelerator_backend=backend,
        accelerator_devices=devices,
        driver_version=driver,
        runtime_version=runtime,
        home=home,
        username=username,
        resolved_config=resolved_config,
    )


def digest_resolved_config(
    resolved_config: Mapping[str, Any],
    *,
    home: str,
    username: str,
) -> str:
    sanitized = sanitize_mapping(resolved_config, home=home, username=username)
    return sha256_hex(canonical_json_bytes(sanitized))


def sanitize_mapping(
    value: Mapping[str, Any],
    *,
    home: str,
    username: str,
    drop_volatile: bool = True,
) -> dict[str, Any]:
    """Redact secrets/paths and drop volatile process-specific keys."""
    out: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if drop_volatile and key.lower() in _TOP_LEVEL_VOLATILE_KEYS:
            continue
        if _SECRET_KEY_RE.search(key):
            out[key] = REDACTED
            continue
        out[key] = _sanitize_value(raw_value, home=home, username=username)
    return out


def redact_text(value: str, *, home: str, username: str) -> str:
    text = value
    if home:
        text = text.replace(home, "~")
        try:
            resolved_home = str(Path(home).expanduser().resolve())
        except OSError:
            resolved_home = home
        if resolved_home and resolved_home != home:
            text = text.replace(resolved_home, "~")
    text = _SECRET_VALUE_RE.sub(REDACTED, text)
    if username and len(username) >= 3:
        escaped = re.escape(username)
        text = re.sub(rf"(?<=[/\\]){escaped}(?=[/\\]|$)", REDACTED_USER, text)
        if text == username:
            text = REDACTED_USER
    return text


def material_fingerprint_fields(fingerprint: EnvironmentFingerprint) -> dict[str, Any]:
    """Fields that make two cells incomparable when they differ."""
    payload = fingerprint.to_canonical_dict()
    accelerator = payload["accelerator"]
    dependencies = payload["dependencies"]
    os_info = payload["os"]
    python = payload["python"]
    repository = payload["repository"]
    return {
        "accelerator.backend": accelerator["backend"],
        "accelerator.devices": tuple(accelerator["devices"]),
        "accelerator.driver_version": accelerator["driver_version"],
        "accelerator.runtime_version": accelerator["runtime_version"],
        "config_digest": payload["config_digest"],
        "dependencies.digest": dependencies["digest"],
        "dependencies.kind": dependencies["kind"],
        "os.machine": os_info["machine"],
        "os.release": os_info["release"],
        "os.system": os_info["system"],
        "python.implementation": python["implementation"],
        "python.version": python["version"],
        "repository.commit": repository["commit"],
        "repository.dirty": repository["dirty"],
    }


def differing_material_fields(
    left: EnvironmentFingerprint,
    right: EnvironmentFingerprint,
) -> tuple[str, ...]:
    left_fields = material_fingerprint_fields(left)
    right_fields = material_fingerprint_fields(right)
    names = sorted(set(left_fields) | set(right_fields))
    return tuple(name for name in names if left_fields.get(name) != right_fields.get(name))


def _sanitize_value(value: Any, *, home: str, username: str) -> Any:
    if isinstance(value, Mapping):
        return sanitize_mapping(value, home=home, username=username)
    if isinstance(value, Path):
        return redact_text(str(value), home=home, username=username)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, home=home, username=username) for item in value]
    if isinstance(value, str):
        return redact_text(value, home=home, username=username)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value), home=home, username=username)


def _optional_redact(value: str | None, *, home: str, username: str) -> str | None:
    if value is None:
        return None
    return redact_text(value, home=home, username=username)


def _sanitize_commit(commit: str | None) -> str | None:
    if commit is None:
        return None
    text = commit.strip()
    return text or None


def _cpu_safe_backend(
    backend: AcceleratorBackend,
    devices: Sequence[str],
) -> AcceleratorBackend:
    """Refuse CUDA/ROCm labels when there is no visible device evidence."""
    if not devices:
        return AcceleratorBackend.CPU
    match backend:
        case AcceleratorBackend.CPU:
            return AcceleratorBackend.CPU
        case AcceleratorBackend.CUDA:
            return AcceleratorBackend.CUDA
        case AcceleratorBackend.ROCM:
            return AcceleratorBackend.ROCM
        case _:
            assert_never(backend)


def _cpu_safe_versions(
    backend: AcceleratorBackend,
    driver_version: str | None,
    runtime_version: str | None,
) -> tuple[str | None, str | None]:
    if backend is AcceleratorBackend.CPU:
        return None, None
    return driver_version, runtime_version


def _discover_repo_root() -> Path | None:
    output = _run_git(["rev-parse", "--show-toplevel"], cwd=None)
    if output is None:
        return Path.cwd()
    return Path(output)


def _probe_git(repo_root: Path | None) -> tuple[str | None, bool | None]:
    cwd = repo_root
    commit = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    porcelain = _run_git(["status", "--porcelain"], cwd=cwd)
    if porcelain is None:
        return commit, None
    return commit, bool(porcelain)


def _probe_lock(repo_root: Path | None) -> tuple[str | None, bytes | None]:
    base = repo_root if repo_root is not None else Path.cwd()
    for name in _LOCK_FILENAMES:
        path = base / name
        if path.is_file():
            return name, path.read_bytes()
    return None, None


def _probe_username() -> str:
    for key in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:
        return ""


def _probe_accelerator() -> tuple[
    AcceleratorBackend, tuple[str, ...], str | None, str | None
]:
    devices, driver = _nvml_devices_and_driver()
    if devices:
        return AcceleratorBackend.CUDA, devices, driver, _cuda_runtime_version()
    rocm_devices, rocm_driver = _rocm_devices_and_driver()
    if rocm_devices:
        return AcceleratorBackend.ROCM, rocm_devices, rocm_driver, None
    return AcceleratorBackend.CPU, (), None, None


def _nvml_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    if nvmlInit is None:
        return (), None
    try:
        nvmlInit()
    except Exception:
        return (), None
    try:
        count = int(nvmlDeviceGetCount())
        names: list[str] = []
        for index in range(count):
            handle = nvmlDeviceGetHandleByIndex(index)
            raw_name = nvmlDeviceGetName(handle)
            names.append(_decode_nvml(raw_name))
        driver = _decode_nvml(nvmlSystemGetDriverVersion())
        if not names:
            return (), None
        return tuple(names), driver or None
    except Exception:
        return (), None
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass


def _decode_nvml(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cuda_runtime_version() -> str | None:
    output = _run_command(["nvidia-smi"], timeout=_ACCEL_TIMEOUT_S)
    if output:
        marker = "CUDA Version:"
        if marker in output:
            tail = output.split(marker, maxsplit=1)[1].strip()
            parts = tail.split()
            if parts:
                return parts[0]
    output = _run_command(["nvcc", "--version"], timeout=_ACCEL_TIMEOUT_S)
    if output:
        marker = "release "
        if marker in output:
            tail = output.split(marker, maxsplit=1)[1]
            parts = tail.split(",", maxsplit=1)
            if parts:
                return parts[0].strip()
    return None


def _rocm_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    output = _run_command(["rocminfo"], timeout=_ACCEL_TIMEOUT_S)
    if not output:
        return (), None
    names: list[str] = []
    for line in output.splitlines():
        if "Marketing Name:" not in line:
            continue
        name = line.split(":", 1)[1].strip()
        if name and name.upper() != "AMD":
            names.append(name)
    if not names:
        return (), None
    return tuple(names), None


def _run_git(args: list[str], cwd: Path | None) -> str | None:
    return _run_command(["git", *args], cwd=cwd, timeout=_GIT_TIMEOUT_S)


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 5.0,
) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


__all__ = [
    "FINGERPRINT_SCHEMA",
    "FINGERPRINT_VERSION",
    "DIGEST_ALGORITHM",
    "REDACTED",
    "REDACTED_USER",
    "AcceleratorBackend",
    "AcceleratorIdentity",
    "DependencyLockIdentity",
    "EnvironmentFingerprint",
    "EnvironmentSnapshot",
    "OsIdentity",
    "PythonIdentity",
    "RepositoryIdentity",
    "canonical_json_bytes",
    "collect_environment_fingerprint",
    "differing_material_fields",
    "digest_resolved_config",
    "fingerprint_from_snapshot",
    "material_fingerprint_fields",
    "probe_environment",
    "redact_text",
    "sanitize_mapping",
    "sha256_hex",
]
