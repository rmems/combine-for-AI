"""Host probes for environment fingerprints.

Each subprocess invocation uses a resolved absolute argv so scanners can see
there is no shell interpolation and no relative executable. Failures degrade
to ``None`` rather than raising.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

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

_GIT_TIMEOUT_S = 5.0
_ACCEL_TIMEOUT_S = 5.0


def discover_repo_root() -> Path:
    output = git_show_toplevel()
    if output is None:
        return Path.cwd()
    return Path(output)


def probe_git(repo_root: Path | None) -> tuple[str | None, bool | None, str | None]:
    commit = git_rev_parse_head(repo_root)
    porcelain = git_status_porcelain(repo_root)
    if porcelain is None:
        return commit, None, None
    return commit, bool(porcelain), porcelain or None


def probe_accelerator() -> tuple[str, tuple[str, ...], str | None, str | None]:
    devices, driver = nvml_devices_and_driver()
    if not devices:
        devices, smi_driver = nvidia_smi_devices_and_driver()
        driver = driver or smi_driver
    devices = _apply_cuda_visibility(devices)
    if devices:
        return "cuda", devices, driver, cuda_runtime_version()
    rocm_devices, rocm_driver = rocm_devices_and_driver()
    if rocm_devices:
        return "rocm", rocm_devices, rocm_driver, None
    return "cpu", (), None, None


def git_rev_parse_head(cwd: Path | None) -> str | None:
    return _run_probe("git", "rev-parse", "HEAD", cwd=cwd, timeout=_GIT_TIMEOUT_S)


def git_show_toplevel() -> str | None:
    return _run_probe("git", "rev-parse", "--show-toplevel", timeout=_GIT_TIMEOUT_S)


def git_status_porcelain(cwd: Path | None) -> str | None:
    return _run_probe("git", "status", "--porcelain", cwd=cwd, timeout=_GIT_TIMEOUT_S)


def cuda_runtime_version() -> str | None:
    nvcc = _run_probe("nvcc", "--version", timeout=_ACCEL_TIMEOUT_S)
    release = _text_after("release ", nvcc)
    if release:
        token = release.split(",", maxsplit=1)
        if token:
            return token[0].strip()
    smi = _run_probe("nvidia-smi", timeout=_ACCEL_TIMEOUT_S)
    tail = _text_after("CUDA Version:", smi)
    if tail:
        token = tail.split()
        if token:
            return token[0]
    return None


def nvidia_smi_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    names_text = _run_probe(
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
        timeout=_ACCEL_TIMEOUT_S,
    )
    if not names_text:
        return (), None
    names = tuple(line.strip() for line in names_text.splitlines() if line.strip())
    if not names:
        return (), None
    driver = None
    smi = _run_probe("nvidia-smi", timeout=_ACCEL_TIMEOUT_S)
    tail = _text_after("Driver Version:", smi)
    if tail:
        token = tail.split()
        if token:
            driver = token[0]
    return names, driver


def rocm_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    output = _run_probe("rocminfo", timeout=_ACCEL_TIMEOUT_S)
    if not output:
        return (), None
    names: list[str] = []
    for line in output.splitlines():
        name = _text_after("Marketing Name:", line)
        if name and name.upper() != "AMD":
            names.append(name)
    if not names:
        return (), None
    return tuple(names), None


def nvml_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    if nvmlInit is None:
        return (), None
    if not _nvml_start():
        return (), None
    try:
        return _nvml_read_devices()
    finally:
        _nvml_stop()


def _nvml_start() -> bool:
    try:
        nvmlInit()
    except Exception as exc:
        _ignore_probe_error(exc)
        return False
    return True


def _nvml_stop() -> None:
    if nvmlShutdown is None:
        return
    try:
        nvmlShutdown()
    except Exception as exc:
        _ignore_probe_error(exc)


def _nvml_read_devices() -> tuple[tuple[str, ...], str | None]:
    if not _nvml_readers_ready():
        return (), None
    try:
        return _nvml_collect()
    except Exception as exc:
        _ignore_probe_error(exc)
        return (), None


def _nvml_readers_ready() -> bool:
    readers = (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlSystemGetDriverVersion,
    )
    return not any(reader is None for reader in readers)


def _nvml_collect() -> tuple[tuple[str, ...], str | None]:
    count = int(nvmlDeviceGetCount())
    names = tuple(name for name in (_nvml_name(index) for index in range(count)) if name)
    driver = _decode_nvml(nvmlSystemGetDriverVersion()) or None
    if not names:
        return (), None
    return names, driver


def _nvml_name(index: int) -> str:
    handle = nvmlDeviceGetHandleByIndex(index)
    return _decode_nvml(nvmlDeviceGetName(handle))


def _decode_nvml(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _apply_cuda_visibility(devices: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return devices
    stripped = raw.strip()
    if stripped == "":
        return devices
    if stripped == "-1":
        return ()
    selected: list[str] = []
    for part in stripped.split(","):
        token = part.strip()
        if not token.isdigit():
            continue
        index = int(token)
        if 0 <= index < len(devices):
            selected.append(devices[index])
    return tuple(selected)


def _resolved_executable(name: str) -> str | None:
    found = shutil.which(name)
    if not found:
        return None
    path = Path(found)
    if not path.is_absolute():
        return None
    return str(path)


def _run_probe(
    name: str,
    *args: str,
    cwd: Path | None = None,
    timeout: float,
) -> str | None:
    executable = _resolved_executable(name)
    if executable is None:
        return None
    try:
        completed = subprocess.run(  # nosec B603
            [executable, *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _text_after(marker: str, text: str | None) -> str | None:
    if not text or marker not in text:
        return None
    _, found, tail = text.partition(marker)
    if not found:
        return None
    stripped = tail.strip()
    return stripped or None


def _ok_stdout(completed: subprocess.CompletedProcess[str]) -> str | None:
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


def _ignore_probe_error(exc: BaseException) -> None:
    _ = (type(exc).__name__, str(exc))
