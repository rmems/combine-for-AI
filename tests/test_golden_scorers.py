from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.cases import DatasetCase, DatasetCaseError, parse_case, validate_case_batch
from benchmarks.scorers import (
    ScorerError,
    assert_score,
    canonicalize_math,
    extract_math_answer,
    mean_score,
    normalize_cloze,
    score_case,
    score_perplexity_corpus,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_scorers.json"


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _parse_logprob(value: float | str) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "nan":
            return float("nan")
        if lowered in {"inf", "infinity", "+inf", "+infinity"}:
            return float("inf")
        if lowered in {"-inf", "-infinity"}:
            return float("-inf")
        return float(value)
    return float(value)


@pytest.mark.parametrize("family", ["classification", "cloze", "perplexity", "exact_match_math"])
def test_golden_metric_family_scores(family: str) -> None:
    golden = _load_golden()
    group = golden["metric_families"][family]
    tolerance = float(group["tolerance"])
    results = []
    for item in group["items"]:
        case = DatasetCase.model_validate(item["case"])
        logprobs = item.get("logprobs")
        result = score_case(
            case,
            prediction=item.get("prediction"),
            logprobs=None if logprobs is None else [_parse_logprob(value) for value in logprobs],
        )
        assert_score(result, float(item["expected_value"]), tolerance=tolerance)
        assert result.dataset == case.dataset
        assert result.example_id == case.example_id
        assert result.metric == family
        if "expected_observed_index" in item:
            assert result.observed["index"] == item["expected_observed_index"]
        if "expected_normalized" in item:
            assert result.expected == item["expected_normalized"]
            assert result.observed == normalize_cloze(item["prediction"])
        if "expected_canonical" in item:
            assert result.expected == item["expected_canonical"]
        results.append(result)
    if family != "perplexity":
        expected_mean = sum(float(item["expected_value"]) for item in group["items"]) / len(
            group["items"]
        )
        assert mean_score(results) == pytest.approx(expected_mean)


def test_golden_perplexity_corpus_mean() -> None:
    group = _load_golden()["metric_families"]["perplexity"]
    cases = [DatasetCase.model_validate(item["case"]) for item in group["items"]]
    logprobs = [item["logprobs"] for item in group["items"]]
    result = score_perplexity_corpus(cases, logprobs, tolerance=float(group["tolerance"]))
    token_count = sum(len(group) for group in logprobs)
    total = sum(sum(values) for values in logprobs)
    expected = math.exp(-(total / token_count))
    assert_score(result, expected, tolerance=1e-12)
    assert result.example_id == "corpus"


@pytest.mark.parametrize("item_index", range(4))
def test_invalid_cases_fail_with_dataset_and_id(item_index: int) -> None:
    item = _load_golden()["invalid_cases"][item_index]
    with pytest.raises(DatasetCaseError) as exc_info:
        parse_case(item["payload"])
    message = str(exc_info.value)
    for snippet in item["error_substrings"]:
        assert snippet in message
    assert exc_info.value.dataset == item["payload"]["dataset"]
    assert exc_info.value.example_id == item["payload"]["example_id"]


def test_duplicate_ids_fail_clearly() -> None:
    spec = _load_golden()["duplicate_ids"]
    cases = [parse_case(payload) for payload in spec["payloads"]]
    with pytest.raises(DatasetCaseError) as exc_info:
        validate_case_batch(cases)
    message = str(exc_info.value)
    for snippet in spec["error_substrings"]:
        assert snippet in message
    assert exc_info.value.example_id == "lambada:dup:0000"
    assert exc_info.value.expected == "unique example_id"
    assert exc_info.value.observed == "lambada:dup:0000"


@pytest.mark.parametrize("item_index", range(2))
def test_non_finite_metrics_fail_clearly(item_index: int) -> None:
    item = _load_golden()["invalid_scores"][item_index]
    case = DatasetCase.model_validate(item["case"])
    logprobs = [_parse_logprob(value) for value in item["logprobs"]]
    with pytest.raises(ScorerError) as exc_info:
        score_case(case, logprobs=logprobs)
    message = str(exc_info.value)
    for snippet in item["error_substrings"]:
        assert snippet in message
    assert exc_info.value.dataset == case.dataset
    assert exc_info.value.example_id == case.example_id
    assert exc_info.value.expected == "finite logprobs"


def test_score_mismatch_names_dataset_id_expected_and_observed() -> None:
    case = DatasetCase.model_validate(
        _load_golden()["metric_families"]["cloze"]["items"][0]["case"]
    )
    result = score_case(case, prediction="cat")
    with pytest.raises(ScorerError, match="score mismatch") as exc_info:
        assert_score(result, 1.0, tolerance=0.0)
    assert exc_info.value.dataset == "lambada"
    assert exc_info.value.example_id == "lambada:validation:0000"
    assert exc_info.value.expected == 1.0
    assert exc_info.value.observed == 0.0


def test_classification_out_of_range_prediction() -> None:
    case = DatasetCase.model_validate(
        _load_golden()["metric_families"]["classification"]["items"][0]["case"]
    )
    with pytest.raises(ScorerError, match="out of range") as exc_info:
        score_case(case, prediction=9)
    assert exc_info.value.dataset == "hellaswag"
    assert exc_info.value.example_id == "hellaswag:validation:0000"
    assert exc_info.value.observed == 9


def test_normalize_cloze_edge_cases() -> None:
    assert normalize_cloze("  Dog. ") == "dog"
    assert normalize_cloze("AIR") == "air"
    assert normalize_cloze("the   store") == "the store"
    assert normalize_cloze("(dog)") == "dog"


def test_math_extraction_edge_cases() -> None:
    assert canonicalize_math(extract_math_answer("#### 72")) == "72"
    assert canonicalize_math(extract_math_answer("The total is 1,000.")) == "1000"
    assert canonicalize_math(extract_math_answer("72.0")) == "72"
    assert canonicalize_math(extract_math_answer(".5")) == "0.5"
    assert canonicalize_math(extract_math_answer("1e3")) == "1000"
    assert extract_math_answer("no numbers") == "no numbers"


def test_corpus_rejects_non_perplexity_cases() -> None:
    case = DatasetCase.model_validate(
        _load_golden()["metric_families"]["cloze"]["items"][0]["case"]
    )
    with pytest.raises(ScorerError, match="task mismatch"):
        score_perplexity_corpus([case], [[-0.5]])


def test_assert_score_rejects_nan_expected() -> None:
    case = DatasetCase.model_validate(
        _load_golden()["metric_families"]["cloze"]["items"][0]["case"]
    )
    result = score_case(case, prediction="dog")
    with pytest.raises(ScorerError, match="non-finite"):
        assert_score(result, float("nan"))


def test_golden_output_artifact_matches_fixture() -> None:
    artifact_path = Path(__file__).resolve().parent / "fixtures" / "golden_scorer_outputs.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    golden = _load_golden()
    by_name = {row["name"]: row for row in artifact["rows"]}
    assert artifact["network_access"] is False
    for family, group in golden["metric_families"].items():
        for item in group["items"]:
            row = by_name[item["name"]]
            assert row["metric"] == family
            assert row["passed"] is True
            assert row["observed_value"] == pytest.approx(
                float(item["expected_value"]),
                abs=float(group["tolerance"] or 0.0) or 1e-15,
            )


def test_golden_fixture_is_self_contained() -> None:
    golden = _load_golden()
    assert golden["schema"] == "combine_for_ai.golden_scorers.v1"
    families = set(golden["metric_families"])
    assert families == {
        "classification",
        "cloze",
        "perplexity",
        "exact_match_math",
    }
    rendered = []
    for family, group in golden["metric_families"].items():
        for item in group["items"]:
            case = DatasetCase.model_validate(item["case"])
            result = score_case(
                case,
                prediction=item.get("prediction"),
                logprobs=item.get("logprobs"),
            )
            rendered.append(
                {
                    "name": item["name"],
                    "metric": family,
                    "dataset": result.dataset,
                    "example_id": result.example_id,
                    "prediction": item.get("prediction"),
                    "logprobs": item.get("logprobs"),
                    "expected_value": item["expected_value"],
                    "observed_value": result.value,
                    "expected": result.expected,
                    "observed": result.observed,
                    "tolerance": group["tolerance"],
                    "passed": True,
                }
            )
    assert len(rendered) == 14
