from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping, Never, Sequence

from benchmarks.cases import DatasetCase, DatasetCaseError, TaskKind, parse_case

_WS = re.compile(r"\s+")
_NUMBER = re.compile(
    r"-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_EDGE_PUNCT = ".,;:!?\"'`()[]{}<>«»“”‘’"


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
    folded = folded.strip(_EDGE_PUNCT + " ")
    return _WS.sub(" ", folded).strip()


def extract_math_answer(text: str | None) -> str | None:
    """GSM8K-style: prefer the token after the last ``####``, else the last number."""
    if text is None:
        return None
    candidate = text.rsplit("####", 1)[-1] if "####" in text else text
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
    if not value.is_finite():
        return None
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _require_finite(
    value: float,
    *,
    dataset: str | None,
    example_id: str | None,
    expected: Any,
    observed: Any,
) -> float:
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


def _index_from_choice_string(case: DatasetCase, prediction: str) -> int:
    choices = case.choices or []
    if prediction in choices:
        return choices.index(prediction)
    normalized = normalize_cloze(prediction)
    for index, choice in enumerate(choices):
        if normalize_cloze(choice) == normalized:
            return index
    raise ScorerError(
        "classification prediction does not match any choice",
        dataset=case.dataset,
        example_id=case.example_id,
        expected=list(choices),
        observed=prediction,
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
    return _index_from_choice_string(case, prediction)


def score_classification(case: DatasetCase, prediction: str | int) -> ScoreResult:
    _require_task(case, TaskKind.CLASSIFICATION)
    observed_index = _choice_prediction(case, prediction)
    expected_index = case.answer_index
    value = 1.0 if observed_index == expected_index else 0.0
    observed_label = case.choices[observed_index] if case.choices else None
    return ScoreResult(
        dataset=case.dataset,
        example_id=case.example_id,
        metric=TaskKind.CLASSIFICATION.value,
        expected={"index": expected_index, "label": case.target},
        observed={"index": observed_index, "label": observed_label},
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
        if number > 0:
            raise ScorerError(
                "positive token log-probability",
                dataset=dataset,
                example_id=example_id,
                expected="non-positive logprobs",
                observed=number,
            )
        values.append(number)
    return values


def _perplexity_from_mean(
    mean_logprob: float,
    *,
    dataset: str,
    example_id: str,
) -> float:
    try:
        value = math.exp(-mean_logprob)
    except OverflowError as exc:
        raise ScorerError(
            "non-finite metric value",
            dataset=dataset,
            example_id=example_id,
            expected="finite perplexity",
            observed=mean_logprob,
        ) from exc
    return _require_finite(
        value,
        dataset=dataset,
        example_id=example_id,
        expected="finite perplexity",
        observed=mean_logprob,
    )


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
    perplexity = _perplexity_from_mean(
        mean_logprob,
        dataset=case.dataset,
        example_id=case.example_id,
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


def _corpus_token_logprobs(
    cases: Sequence[DatasetCase],
    logprobs_by_case: Sequence[Sequence[float]],
) -> tuple[str, list[float]]:
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
        _require_task(case, TaskKind.PERPLEXITY)
        token_logprobs.extend(
            _as_logprobs(group, dataset=case.dataset, example_id=case.example_id)
        )
    if not token_logprobs:
        raise ScorerError(
            "missing logprobs",
            dataset=cases[0].dataset,
            example_id="corpus",
            expected="at least one logprob value",
            observed=0,
        )
    datasets = {case.dataset for case in cases}
    if len(datasets) != 1:
        raise ScorerError(
            "corpus cases must share one dataset",
            dataset=cases[0].dataset,
            example_id="corpus",
            expected="single dataset",
            observed=sorted(datasets),
        )
    return cases[0].dataset, token_logprobs


def score_perplexity_corpus(
    cases: Sequence[DatasetCase],
    logprobs_by_case: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-12,
) -> ScoreResult:
    dataset, token_logprobs = _corpus_token_logprobs(cases, logprobs_by_case)
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    perplexity = _perplexity_from_mean(
        mean_logprob,
        dataset=dataset,
        example_id="corpus",
    )
    return ScoreResult(
        dataset=dataset,
        example_id="corpus",
        metric=TaskKind.PERPLEXITY.value,
        expected={"token_count": len(token_logprobs)},
        observed={"logprob_count": len(token_logprobs), "mean_logprob": mean_logprob},
        value=perplexity,
        passed=True,
        tolerance=tolerance,
    )


def _unhandled_task(task: Never) -> Never:
    raise ScorerError(
        "task mismatch",
        dataset=None,
        example_id=None,
        expected="known task",
        observed=task,
    )


def _score_with_prediction(
    case: DatasetCase,
    prediction: str | int | None,
    expected: str,
    scorer: Callable[[DatasetCase, str | int], ScoreResult],
) -> ScoreResult:
    if prediction is None:
        raise ScorerError(
            "missing prediction",
            dataset=case.dataset,
            example_id=case.example_id,
            expected=expected,
            observed=None,
        )
    return scorer(case, prediction)


def score_case(
    case: DatasetCase,
    *,
    prediction: str | int | None = None,
    logprobs: Sequence[float] | None = None,
) -> ScoreResult:
    task = case.task
    if task is TaskKind.CLASSIFICATION:
        return _score_with_prediction(
            case, prediction, "choice index or label", score_classification
        )
    if task is TaskKind.CLOZE:
        return _score_with_prediction(
            case, prediction, "cloze string", lambda c, p: score_cloze(c, str(p))
        )
    if task is TaskKind.EXACT_MATCH_MATH:
        return _score_with_prediction(
            case,
            prediction,
            "math answer string",
            lambda c, p: score_exact_match_math(c, str(p)),
        )
    if task is TaskKind.PERPLEXITY:
        if logprobs is None:
            raise ScorerError(
                "missing logprobs",
                dataset=case.dataset,
                example_id=case.example_id,
                expected="finite logprobs",
                observed=None,
            )
        return score_perplexity(case, logprobs)
    return _unhandled_task(task)


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
    _require_finite(
        float(expected_value),
        dataset=result.dataset,
        example_id=result.example_id,
        expected="finite expected score",
        observed=expected_value,
    )
    if allowed < 0 or not math.isfinite(allowed):
        raise ScorerError(
            "invalid tolerance",
            dataset=result.dataset,
            example_id=result.example_id,
            expected="non-negative finite tolerance",
            observed=allowed,
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
        case = parse_case(entry["case"])
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
