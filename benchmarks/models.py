from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, Protocol

from benchmarks.datasets import DatasetRecord


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    name: str
    revision: str | None = None

    @staticmethod
    def from_dict(raw: dict) -> "ModelSpec":
        return ModelSpec(
            backend=raw.get("backend", "mock"),
            name=raw["name"],
            revision=raw.get("revision"),
        )


@dataclass(frozen=True)
class QuantizationProfile:
    name: str
    precision: str
    format: str
    bits: int | None
    supported: bool
    speed_tps: float
    vram_gb: float
    notes: str


class QuantizationRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, QuantizationProfile] = {}

    def register(self, profile: QuantizationProfile) -> None:
        self._profiles[profile.name] = profile

    def get(self, name: str) -> QuantizationProfile:
        if name not in self._profiles:
            raise ValueError(f"unknown quantization '{name}'")
        return self._profiles[name]

    def supported_names(self) -> Iterable[str]:
        return sorted(self._profiles.keys())


def default_quantization_registry() -> QuantizationRegistry:
    registry = QuantizationRegistry()
    registry.register(
        QuantizationProfile(
            name="fp16",
            precision="fp16",
            format="baseline",
            bits=16,
            supported=True,
            speed_tps=1200.0,
            vram_gb=14.0,
            notes="Baseline FP16 evaluation path.",
        )
    )
    registry.register(
        QuantizationProfile(
            name="awq",
            precision="int4",
            format="awq",
            bits=4,
            supported=True,
            speed_tps=1700.0,
            vram_gb=10.5,
            notes="Activation-aware weight quantization.",
        )
    )
    registry.register(
        QuantizationProfile(
            name="gptq",
            precision="int4",
            format="gptq",
            bits=4,
            supported=True,
            speed_tps=1500.0,
            vram_gb=11.0,
            notes="GPTQ static quantization.",
        )
    )
    registry.register(
        QuantizationProfile(
            name="gguf",
            precision="int4",
            format="gguf",
            bits=4,
            supported=True,
            speed_tps=1400.0,
            vram_gb=11.5,
            notes="GGUF quantized weights.",
        )
    )
    registry.register(
        QuantizationProfile(
            name="ternary",
            precision="int2",
            format="ternary",
            bits=2,
            supported=True,
            speed_tps=900.0,
            vram_gb=6.0,
            notes="Ternary {-1,0,+1} weights (GOZ1 ternary payload without full SAAQ metadata).",
        )
    )
    registry.register(
        QuantizationProfile(
            name="saaq",
            precision="int2",
            format="goz1",
            bits=2,
            supported=True,
            speed_tps=950.0,
            vram_gb=5.5,
            notes=(
                "SAAQ (Spiking Adaptive Activity Quantization) path for GOZ1 packs: "
                "ternary_snn experts + preserve routers/norms; metrics often imported "
                "from grok-ozempic experiments rather than full in-process inference."
            ),
        )
    )
    return registry


@dataclass(frozen=True)
class Prediction:
    output: str | int
    logprob: float
    tokens: int


class ModelAdapter(Protocol):
    def predict(self, record: DatasetRecord, rng: random.Random) -> Prediction:
        ...

    @property
    def profile(self) -> QuantizationProfile:
        ...

    @property
    def spec(self) -> ModelSpec:
        ...


class MockModelAdapter:
    def __init__(self, spec: ModelSpec, profile: QuantizationProfile) -> None:
        self._spec = spec
        self._profile = profile

    @property
    def profile(self) -> QuantizationProfile:
        return self._profile

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def predict(self, record: DatasetRecord, rng: random.Random) -> Prediction:
        token_budget = len(record.prompt.split())
        if record.choices:
            choice = rng.randrange(len(record.choices))
            output = choice
            logprob = math.log(1.0 / max(len(record.choices), 1))
            token_budget += len(record.choices[choice].split())
        else:
            correct_rate = self._correct_rate()
            if record.reference and rng.random() < correct_rate:
                output = record.reference
            else:
                output = self._mutate_reference(record.reference or "unknown", rng)
            logprob = rng.uniform(-5.0, -0.5)
            token_budget += len(str(output).split())

        return Prediction(output=output, logprob=logprob, tokens=max(token_budget, 1))

    def _correct_rate(self) -> float:
        rates = {
            "fp16": 0.86,
            "awq": 0.84,
            "gptq": 0.83,
            "gguf": 0.82,
            "ternary": 0.78,
            "saaq": 0.79,
        }
        return rates.get(self._profile.name, 0.8)

    @staticmethod
    def _mutate_reference(reference: str, rng: random.Random) -> str:
        words = reference.split()
        if not words:
            return reference
        index = rng.randrange(len(words))
        words[index] = words[index][::-1]
        return " ".join(words)


def build_model_adapter(spec: ModelSpec, profile: QuantizationProfile) -> ModelAdapter:
    if not profile.supported:
        raise NotImplementedError(
            f"quantization '{profile.name}' is registered as a hook but has no backend yet"
        )

    if spec.backend == "mock":
        return MockModelAdapter(spec, profile)

    raise ValueError(f"unknown model backend '{spec.backend}'")


def scoped_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return int(digest, 16) ^ base_seed
