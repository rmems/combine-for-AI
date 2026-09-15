"""Host probes for environment fingerprints.

Each subprocess invocation uses a literal argv so scanners can see there is
no shell interpolation and no caller-controlled executable. Failures degrade
to ``None`` rather than raising.
"""

from __future__ import annotations

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


def probe_git(repo_root: Path | None) -> tuple[str | None, bool | None]:
    commit = git_rev_parse_head(repo_root)
    porcelain = git_status_porcelain(repo_root)
    if porcelain is None:
        return commit, None
    return commit, bool(porcelain)


def probe_accelerator() -> tuple[str, tuple[str, ...], str | None, str | None]:
    devices, driver = nvml_devices_and_driver()
    if devices:
        return "cuda", devices, driver, cuda_runtime_version()
    rocm_devices, rocm_driver = rocm_devices_and_driver()
    if rocm_devices:
        return "rocm", rocm_devices, rocm_driver, None
    return "cpu", (), None, None


def git_rev_parse_head(cwd: Path | None) -> str | None:
    try:
        completed = subprocess.run(  # nosec B603
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def git_show_toplevel() -> str | None:
    try:
        completed = subprocess.run(  # nosec B603
            ["git", "rev-parse", "--show-toplevel"],
            cwd=None,
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def git_status_porcelain(cwd: Path | None) -> str | None:
    try:
        completed = subprocess.run(  # nosec B603
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def cuda_runtime_version() -> str | None:
    smi = _nvidia_smi()
    tail = _text_after("CUDA Version:", smi)
    if tail:
        token = tail.split()
        if token:
            return token[0]
    nvcc = _nvcc_version()
    release = _text_after("release ", nvcc)
    if release:
        token = release.split(",", maxsplit=1)
        if token:
            return token[0].strip()
    return None


def _nvidia_smi() -> str | None:
    try:
        completed = subprocess.run(  # nosec B603
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_ACCEL_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _nvcc_version() -> str | None:
    try:
        completed = subprocess.run(  # nosec B603
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_ACCEL_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def rocm_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    output = _rocminfo()
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


def _rocminfo() -> str | None:
    try:
        completed = subprocess.run(  # nosec B603
            ["rocminfo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_ACCEL_TIMEOUT_S,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


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
    readers = (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlSystemGetDriverVersion,
    )
    if any(reader is None for reader in readers):
        return (), None
    try:
        count = int(nvmlDeviceGetCount())
        names = tuple(_nvml_name(index) for index in range(count))
        names = tuple(name for name in names if name)
        driver = _decode_nvml(nvmlSystemGetDriverVersion()) or None
        if not names:
            return (), None
        return names, driver
    except Exception as exc:
        _ignore_probe_error(exc)
        return (), None


def _nvml_name(index: int) -> str:
    handle = nvmlDeviceGetHandleByIndex(index)
    return _decode_nvml(nvmlDeviceGetName(handle))


def _decode_nvml(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
