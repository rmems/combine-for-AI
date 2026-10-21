"""Mock-only performance presets.

These numbers are invented fixture values so offline tests and the mock path
have deterministic throughput and memory. They are never measurements. Only
code that holds a ``MockModelAdapter`` may call :func:`mock_performance`;
measured or imported results must supply real values or an unavailable status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmarks.models import QuantizationProfile
from combine_for_ai.evidence import ExecutionKind


@dataclass(frozen=True)
class MockPerformance:
    speed_tps: float
    vram_gb: float


MOCK_PERFORMANCE_PRESETS: dict[str, MockPerformance] = {
    "fp16": MockPerformance(1200.0, 14.0),
    "awq": MockPerformance(1700.0, 10.5),
    "gptq": MockPerformance(1500.0, 11.0),
    "gguf": MockPerformance(1400.0, 11.5),
    "ternary": MockPerformance(900.0, 6.0),
    "saaq": MockPerformance(950.0, 5.5),
}


def _resolve(profile: QuantizationProfile) -> tuple[float | None, float | None]:
    """Explicit profile overrides win over the preset table."""
    preset = MOCK_PERFORMANCE_PRESETS.get(profile.name)
    speed = profile.speed_tps if profile.speed_tps is not None else getattr(preset, "speed_tps", None)
    vram = profile.vram_gb if profile.vram_gb is not None else getattr(preset, "vram_gb", None)
    return speed, vram


def _require_positive(label: str, name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(f"no mock performance preset for quantization '{name}'")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"mock {label} for '{name}' must be finite and > 0, got {value}")
    return value


def mock_performance(
    profile: QuantizationProfile, execution_kind: ExecutionKind = ExecutionKind.MOCK
) -> MockPerformance:
    """Fixture speed/VRAM for ``profile``; refuses any non-mock execution."""
    if execution_kind is not ExecutionKind.MOCK:
        raise ValueError(
            f"mock performance presets are fixture-only, not valid for {execution_kind.value}"
        )
    speed, vram = _resolve(profile)
    return MockPerformance(
        _require_positive("speed_tps", profile.name, speed),
        _require_positive("vram_gb", profile.name, vram),
    )
