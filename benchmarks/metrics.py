from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace

from benchmarks.datasets import DatasetRecord
from benchmarks.models import Prediction


@dataclass(frozen=True)
class SaaqMetricOverlay:
    """SAAQ scalars mapped from corinth-canal telemetry into a metrics summary."""

    firing_rate: float | None = None
    membrane_pressure: float | None = None
    saaq_delta_q: float | None = None
    saaq_delta_q_last: float | None = None
    saaq_delta_q_legacy: float | None = None
    saaq_delta_q_v15: float | None = None
    saaq_rule: str | None = None
    spike_density: float | None = None


@dataclass(frozen=True)
class MetricsSummary:
    accuracy: float
    perplexity: float
    throughput: float
    latency_ms: float
    vram_gb: float
    routing_entropy: float
    spike_density: float | None = None
    # MoE / SNN fields (nullable — import from grok-ozempic experiments when present)
    route_top1_agreement: float | None = None
    route_top2_agreement: float | None = None
    block_output_cosine: float | None = None
    resid_in_drift: float | None = None
    block_index: int | None = None
    expert_load_js: float | None = None
    scale_source: str | None = None
    goz1_version: int | None = None
    sparsity: float | None = None
    firing_rate: float | None = None
    membrane_pressure: float | None = None
    saaq_delta_q: float | None = None
    saaq_delta_q_last: float | None = None
    saaq_delta_q_legacy: float | None = None
    saaq_delta_q_v15: float | None = None
    saaq_rule: str | None = None


class MetricsAccumulator:
    def __init__(self) -> None:
        self._total = 0
        self._correct = 0
        self._logprob_sum = 0.0
        self._token_count = 0
        self._choice_counts: Counter[str] = Counter()
        self._saaq: SaaqMetricOverlay | None = None

    def apply_saaq(self, overlay: SaaqMetricOverlay) -> None:
        """Attach SAAQ telemetry so ``summary()`` emits it beside traditional metrics."""

        self._saaq = overlay

    def add(self, record: DatasetRecord, prediction: Prediction) -> None:
        self._total += 1
        self._logprob_sum += prediction.logprob
        self._token_count += prediction.tokens

        if record.is_multiple_choice:
            key = f"choice_{prediction.output}"
            self._choice_counts[key] += 1
            if prediction.output == record.answer_index:
                self._correct += 1
        else:
            if prediction.output == record.reference:
                self._correct += 1
                self._choice_counts["correct"] += 1
            else:
                self._choice_counts["incorrect"] += 1

    def summary(self, total_time_s: float, vram_gb: float) -> MetricsSummary:
        accuracy = self._correct / self._total if self._total else 0.0
        avg_logprob = self._logprob_sum / self._total if self._total else 0.0
        perplexity = math.exp(-avg_logprob) if self._total else 0.0
        throughput = self._token_count / total_time_s if total_time_s else 0.0
        latency_ms = (total_time_s / self._total * 1000.0) if self._total else 0.0

        routing_entropy = entropy_from_counts(self._choice_counts)
        summary = MetricsSummary(
            accuracy=accuracy,
            perplexity=perplexity,
            throughput=throughput,
            latency_ms=latency_ms,
            vram_gb=vram_gb,
            routing_entropy=routing_entropy,
            spike_density=None,
            route_top1_agreement=None,
            route_top2_agreement=None,
            block_output_cosine=None,
            resid_in_drift=None,
            block_index=None,
            expert_load_js=None,
            scale_source=None,
            goz1_version=None,
            sparsity=None,
        )
        if self._saaq is None:
            return summary
        return replace(
            summary,
            spike_density=_coalesce(self._saaq.spike_density, summary.spike_density),
            firing_rate=self._saaq.firing_rate,
            membrane_pressure=self._saaq.membrane_pressure,
            saaq_delta_q=self._saaq.saaq_delta_q,
            saaq_delta_q_last=self._saaq.saaq_delta_q_last,
            saaq_delta_q_legacy=self._saaq.saaq_delta_q_legacy,
            saaq_delta_q_v15=self._saaq.saaq_delta_q_v15,
            saaq_rule=self._saaq.saaq_rule,
        )

    @property
    def token_count(self) -> int:
        return self._token_count


def _coalesce(new: float | None, old: float | None) -> float | None:
    return new if new is not None else old


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count == 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob)
    return entropy
