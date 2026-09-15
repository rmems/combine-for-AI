from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.cases import DatasetCase, DatasetCaseError, TaskKind

_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_TRAILING_PUNCT = ".,;:!?\"'`)]}"


class ScorerError(DatasetCaseError):
    """Scoring failed for a specific example."""


@dataclass(frozen=True)
class ScoreResult:
    dataset: str
    example_id: str
    metric: str
    expected: Any
    observed: Any
    value: float
    passed: bool
    tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "metric": self.metric,
            "expected": self.expected,
            "observed": self.observed,
            "value": self.value,
            "passed": self.passed,
            "tolerance": self.tolerance,
        }


def normalize_cloze(text: str) -> str:
    """Case-fold, collapse whitespace, and strip edge punctuation."""
    folded = unicodedata.normalize("NFC", text).strip().lower()
    folded = _WS.sub(" ", folded)
    folded = folded.strip(_TRAILING_PUNCT + " ")
    return _WS.sub(" ", folded).strip()


def extract_math_answer(text: str) -> str | None:
    """GSM8K-style: prefer the token after the last ``####``, else the last number."""
    if text is None:
        return None
    candidate = text
    if "####" in text:
        candidate = text.rsplit("####", 1)[-1]
    matches = _NUMBER.findall(candidate)
    if matches:
        return matches[-1]
    stripped = candidate.strip()
    return stripped or None


def canonicalize_math(token: str | None) -> str | None:
    if token is None:
        return None
    stripped = token.strip().replace(",", "")
    if not stripped:
        return None
    try:
        value = Decimal(stripped)
    except InvalidOperation:
        return stripped.lower()
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _require_finite(value: float, *, dataset: str, example_id: str, expected: Any, observed: Any) -> float:
    if not math.isfinite(value):
        raise ScorerError(
            "non-finite metric value",
            dataset=dataset,
            example_id=example_id,
            expected=expected,
            observed=observed,
        )
    return value


def _require_task(case: DatasetCase, metric: TaskKind) -> None:
    if case.task is not metric:
        raise ScorerError(
            "task mismatch",
            dataset=case.dataset,
            example_id=case.example_id,
            expected=metric.value,
            observed=case.task.value,
        )


def _choice_prediction(case: DatasetCase, prediction: str | int) -> int:
    if case.choices is None:
        raise ScorerError(
            "malformed choices",
            dataset=case.dataset,
            example_id=case.example_id,
            expected="non-empty choices",
            observed=None,
        )
    if isinstance(prediction, bool):
        raise ScorerError(
            "invalid classification prediction",
            dataset=case.dataset,
            example_id=case.example_id,
            expected=f"index in 0..{len(case.choices) - 1} or a choice string",
            observed=prediction,
        )
    if isinstance(prediction, int):
        if not 0 <= prediction < len(case.choices):
            raise ScorerError(
                "classification prediction out of range",
                dataset=case.dataset,
                example_id=case.example_id,
                expected=f"index in 0..{len(case.choices) - 1}",
                observed=prediction,
            )
        return prediction
    if prediction in case.choices:
        return case.choices.index(prediction)
    normalized = normalize_cloze(prediction)
    for index, choice in enumerate(case.choices):
        if normalize_cloze(choice) == normalized:
            return index
    raise ScorerError(
        "classification prediction does not match any choice",
        dataset=case.dataset,
        example_id=case.example_id,
        expected=list(case.choices),
        observed=prediction,
    )


def score_classification(case: DatasetCase, prediction: str | int) -> ScoreResult:
    _require_task(case, TaskKind.CLASSIFICATION)
    observed_index = _choice_prediction(case, prediction)
    expected_index = case.answer_index
    value = 1.0 if observed_index == expected_index else 0.0
    return ScoreResult(
        dataset=case.dataset,
        example_id=case.example_id,
        metric=TaskKind.CLASSIFICATION.value,
        expected={"index": expected_index, "label": case.target},
        observed={"index": observed_index, "label": case.choices[observed_index] if case.choices else None},
        value=value,
        passed=value == 1.0,
        tolerance=0.0,
    )


def score_cloze(case: DatasetCase, prediction: str) -> ScoreResult:
    _require_task(case, TaskKind.CLOZE)
    expected = normalize_cloze(case.target)
    observed = normalize_cloze(str(prediction))
    value = 1.0 if expected == observed else 0.0
    return ScoreResult(
        dataset=case.dataset,
        example_id=case.example_id,
        metric=TaskKind.CLOZE.value,
        expected=expected,
        observed=observed,
        value=value,
        passed=value == 1.0,
        tolerance=0.0,
    )


def score_exact_match_math(case: DatasetCase, prediction: str) -> ScoreResult:
    _require_task(case, TaskKind.EXACT_MATCH_MATH)
    expected = canonicalize_math(extract_math_answer(case.target) or case.target)
    observed = canonicalize_math(extract_math_answer(str(prediction)))
    if expected is None:
        raise ScorerError(
            "missing target",
            dataset=case.dataset,
            example_id=case.example_id,
            expected="numeric answer",
            observed=case.expected,
        )
    value = 1.0 if observed is not None and observed == expected else 0.0
    return ScoreResult(
        dataset=case.dataset,
        example_id=case.example_id,
        metric=TaskKind.EXACT_MATCH_MATH.value,
        expected=expected,
        observed=observed,
        value=value,
        passed=value == 1.0,
        tolerance=0.0,
    )


def _as_logprobs(logprobs: Sequence[float], *, dataset: str, example_id: str) -> list[float]:
    if not logprobs:
        raise ScorerError(
            "missing logprobs",
            dataset=dataset,
            example_id=example_id,
            expected="non-empty finite logprobs",
            observed=list(logprobs),
        )
    values: list[float] = []
    for item in logprobs:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ScorerError(
                "non-finite metric value",
                dataset=dataset,
                example_id=example_id,
                expected="finite logprobs",
                observed=item,
            ) from exc
        if not math.isfinite(number):
            raise ScorerError(
                "non-finite metric value",
                dataset=dataset,
                example_id=example_id,
                expected="finite logprobs",
                observed=number,
            )
        values.append(number)
    return values


def score_perplexity(
    case: DatasetCase,
    logprobs: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> ScoreResult:
    """Token-mean NLL perplexity: ``exp(-mean(logprobs))`` with natural-log logprobs."""
    _require_task(case, TaskKind.PERPLEXITY)
    values = _as_logprobs(logprobs, dataset=case.dataset, example_id=case.example_id)
    mean_logprob = sum(values) / len(values)
    perplexity = _require_finite(
        math.exp(-mean_logprob),
        dataset=case.dataset,
        example_id=case.example_id,
        expected="finite perplexity",
        observed=None,
    )
    return ScoreResult(
        dataset=case.dataset,
        example_id=case.example_id,
        metric=TaskKind.PERPLEXITY.value,
        expected=case.target,
        observed={"logprob_count": len(values), "mean_logprob": mean_logprob},
        value=perplexity,
        passed=True,
        tolerance=tolerance,
    )


def score_perplexity_corpus(
    cases: Sequence[DatasetCase],
    logprobs_by_case: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-12,
) -> ScoreResult:
    if len(cases) != len(logprobs_by_case):
        raise ScorerError(
            "logprob groups must align with cases",
            dataset=cases[0].dataset if cases else None,
            example_id=None,
            expected=len(cases),
            observed=len(logprobs_by_case),
        )
    if not cases:
        raise ScorerError(
            "missing logprobs",
            dataset=None,
            example_id=None,
            expected="at least one case",
            observed=0,
        )
    token_logprobs: list[float] = []
    for case, group in zip(cases, logprobs_by_case, strict=True):
        token_logprobs.extend(
            _as_logprobs(group, dataset=case.dataset, example_id=case.example_id)
        )
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    perplexity = _require_finite(
        math.exp(-mean_logprob),
        dataset=cases[0].dataset,
        example_id="corpus",
        expected="finite perplexity",
        observed=None,
    )
    return ScoreResult(
        dataset=cases[0].dataset,
        example_id="corpus",
        metric=TaskKind.PERPLEXITY.value,
        expected={"token_count": len(token_logprobs)},
        observed={"logprob_count": len(token_logprobs), "mean_logprob": mean_logprob},
        value=perplexity,
        passed=True,
        tolerance=tolerance,
    )


def score_case(
    case: DatasetCase,
    *,
    prediction: str | int | None = None,
    logprobs: Sequence[float] | None = None,
) -> ScoreResult:
    if case.task is TaskKind.CLASSIFICATION:
        if prediction is None:
            raise ScorerError(
                "missing prediction",
                dataset=case.dataset,
                example_id=case.example_id,
                expected="choice index or label",
                observed=None,
            )
        return score_classification(case, prediction)
    if case.task is TaskKind.CLOZE:
        if prediction is None:
            raise ScorerError(
                "missing prediction",
                dataset=case.dataset,
                example_id=case.example_id,
                expected="cloze string",
                observed=None,
            )
        return score_cloze(case, str(prediction))
    if case.task is TaskKind.EXACT_MATCH_MATH:
        if prediction is None:
            raise ScorerError(
                "missing prediction",
                dataset=case.dataset,
                example_id=case.example_id,
                expected="math answer string",
                observed=None,
            )
        return score_exact_match_math(case, str(prediction))
    if logprobs is None:
        raise ScorerError(
            "missing logprobs",
            dataset=case.dataset,
            example_id=case.example_id,
            expected="finite logprobs",
            observed=None,
        )
    return score_perplexity(case, logprobs)


def mean_score(results: Sequence[ScoreResult]) -> float:
    if not results:
        return 0.0
    total = 0.0
    for result in results:
        _require_finite(
            result.value,
            dataset=result.dataset,
            example_id=result.example_id,
            expected="finite metric",
            observed=result.value,
        )
        total += result.value
    value = total / len(results)
    return _require_finite(
        value,
        dataset=results[0].dataset,
        example_id="aggregate",
        expected="finite mean",
        observed=value,
    )


def assert_score(
    result: ScoreResult,
    expected_value: float,
    *,
    tolerance: float | None = None,
) -> None:
    allowed = result.tolerance if tolerance is None else tolerance
    if allowed is None:
        allowed = 0.0
    _require_finite(
        result.value,
        dataset=result.dataset,
        example_id=result.example_id,
        expected=expected_value,
        observed=result.value,
    )
    if abs(result.value - expected_value) > allowed:
        raise ScorerError(
            "score mismatch",
            dataset=result.dataset,
            example_id=result.example_id,
            expected=expected_value,
            observed=result.value,
        )


def results_from_fixture(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replay a golden fixture of inputs and return scorer outputs."""
    outputs: list[dict[str, Any]] = []
    for entry in entries:
        case = DatasetCase.model_validate(entry["case"])
        result = score_case(
            case,
            prediction=entry.get("prediction"),
            logprobs=entry.get("logprobs"),
        )
        payload = result.to_dict()
        payload["input"] = {
            "prediction": entry.get("prediction"),
            "logprobs": entry.get("logprobs"),
        }
        outputs.append(payload)
    return outputs
