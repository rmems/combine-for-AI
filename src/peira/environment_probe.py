"""Host probes for environment fingerprints.

Each subprocess invocation uses a literal argv (no shell, no dynamic executable).
Failures degrade to ``None`` rather than raising.
"""

from __future__ import annotations

import os
import re
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
_ROCM_DEVICE_TYPE_FIELD = "Device Type:"


def discover_repo_root() -> Path:
    output = git_show_toplevel()
    if output is None:
        return Path.cwd()
    return Path(output)


def probe_git(
    repo_root: Path | None,
) -> tuple[str | None, bool | None, str | None, str | None]:
    commit = git_rev_parse_head(repo_root)
    porcelain = git_status_porcelain(repo_root)
    if porcelain is None:
        return commit, None, None, None
    if porcelain == "":
        return commit, False, None, None
    return commit, True, porcelain, git_diff_head(repo_root)


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
    return _run_git_rev_parse_head(cwd, _GIT_TIMEOUT_S)


def git_show_toplevel() -> str | None:
    return _run_git_show_toplevel(_GIT_TIMEOUT_S)


def git_status_porcelain(cwd: Path | None) -> str | None:
    return _run_git_status_porcelain(cwd, _GIT_TIMEOUT_S)


def git_diff_head(cwd: Path | None) -> str | None:
    return _run_git_diff_head(cwd, _GIT_TIMEOUT_S)


def cuda_runtime_version() -> str | None:
    nvcc = _run_nvcc_version(_ACCEL_TIMEOUT_S)
    release = _text_after("release ", nvcc)
    if release:
        token = release.split(",", maxsplit=1)
        if token:
            return token[0].strip()
    smi = _run_nvidia_smi(_ACCEL_TIMEOUT_S)
    tail = _text_after("CUDA Version:", smi)
    if tail:
        token = tail.split()
        if token:
            return token[0]
    return None


def nvidia_smi_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    names_text = _run_nvidia_smi_gpu_names(_ACCEL_TIMEOUT_S)
    if not names_text:
        return (), None
    names = tuple(line.strip() for line in names_text.splitlines() if line.strip())
    if not names:
        return (), None
    driver = None
    smi = _run_nvidia_smi(_ACCEL_TIMEOUT_S)
    tail = _text_after("Driver Version:", smi)
    if tail:
        token = tail.split()
        if token:
            driver = token[0]
    return names, driver


def rocm_devices_and_driver() -> tuple[tuple[str, ...], str | None]:
    output = _run_rocminfo(_ACCEL_TIMEOUT_S)
    if not output:
        return (), None
    names: list[str] = []
    for block in re.split(r"^Agent \d+\s*$", output, flags=re.MULTILINE):
        if _ROCM_DEVICE_TYPE_FIELD not in block or not _rocm_block_is_gpu(block):
            continue
        names.extend(_rocm_marketing_names(block))
    if not names:
        return (), None
    return tuple(names), None


def _rocm_block_is_gpu(block: str) -> bool:
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith(_ROCM_DEVICE_TYPE_FIELD):
            continue
        device_type = (_text_after(_ROCM_DEVICE_TYPE_FIELD, stripped) or "").upper()
        return device_type == "GPU"
    return False


def _rocm_marketing_names(block: str) -> list[str]:
    names: list[str] = []
    for line in block.splitlines():
        name = _text_after("Marketing Name:", line)
        if name and name.upper() != "AMD":
            names.append(name)
    return names


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


def _visible_cuda_device(
    devices: tuple[str, ...],
    token: str,
    *,
    uuid_to_name: dict[str, str],
) -> str | None:
    if _cuda_visibility_token_invalid(token):
        return None
    if token.isdigit():
        return _cuda_device_by_index(devices, int(token))
    lowered = token.lower()
    mapped = _cuda_device_by_uuid(lowered, uuid_to_name)
    if mapped is not None:
        return mapped
    return _cuda_device_by_name(devices, lowered)


def _cuda_visibility_token_invalid(token: str) -> bool:
    if not token:
        return True
    if token.lstrip("+-").isdigit():
        return int(token) < 0
    return False


def _cuda_device_by_index(devices: tuple[str, ...], index: int) -> str | None:
    if 0 <= index < len(devices):
        return devices[index]
    return None


def _cuda_device_by_uuid(lowered: str, uuid_to_name: dict[str, str]) -> str | None:
    if not lowered.startswith(("gpu-", "mig-")):
        return None
    return uuid_to_name.get(lowered)


def _cuda_device_by_name(devices: tuple[str, ...], lowered: str) -> str | None:
    for device in devices:
        name = device.lower()
        if name == lowered or lowered in name:
            return device
    return None


def _apply_cuda_visibility(devices: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return devices
    stripped = raw.strip()
    if not stripped:
        return ()
    if stripped.lstrip("+-").isdigit() and int(stripped) < 0:
        return ()
    uuid_to_name = _nvidia_smi_uuid_to_name(_ACCEL_TIMEOUT_S)
    selected: list[str] = []
    for token in stripped.split(","):
        name = _visible_cuda_device(
            devices, token.strip(), uuid_to_name=uuid_to_name
        )
        if name is None:
            break
        selected.append(name)
    return tuple(selected)


def _nvidia_smi_uuid_to_name(timeout: float) -> dict[str, str]:
    text = _run_nvidia_smi_gpu_uuid_and_name(timeout)
    if not text:
        return {}
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split(",", maxsplit=1)
        if len(parts) != 2:
            continue
        uuid, name = parts[0].strip().lower(), parts[1].strip()
        if uuid and name:
            mapping[uuid] = name
    return mapping


def _run_git_rev_parse_head(cwd: Path | None, timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_git_show_toplevel(timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_git_status_porcelain(cwd: Path | None, timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed, preserve_empty=True)


def _run_git_diff_head(cwd: Path | None, timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["git", "diff", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_nvcc_version(timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_nvidia_smi(timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_nvidia_smi_gpu_names(timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_nvidia_smi_gpu_uuid_and_name(timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            [
                "nvidia-smi",
                "--query-gpu=uuid,name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _ok_stdout(completed)


def _run_rocminfo(timeout: float) -> str | None:
    try:
        completed = subprocess.run(  # nosec B607
            ["rocminfo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # nosec B603
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


def _ok_stdout(
    completed: subprocess.CompletedProcess[str],
    *,
    preserve_empty: bool = False,
) -> str | None:
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    if not text and preserve_empty:
        return ""
    return text or None


def _ignore_probe_error(exc: BaseException) -> None:
    _ = (type(exc).__name__, str(exc))
