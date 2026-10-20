from __future__ import annotations

import json
from pathlib import Path

import pytest

from combine_for_ai.live_baselines import (
    LiveBaselineError,
    annotate_row_hygiene,
    byte_identical_gold_src,
    contains_gold_test_patch,
    filter_usable_runs,
    flagged_baseline_ids,
    git_blob_sha1,
    ingest_result_rows,
    ingest_run_directory,
    is_leaderboard_aggregate_payload,
    load_baseline_registry,
    rank_runs_on_shared_rows,
    shared_instance_ids,
    TaskResultRow,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "swe_bench_live"
REGISTRY = Path(__file__).resolve().parents[1] / "data" / "swe_bench_live" / "baselines.json"


def test_registry_marks_known_bad_submissions_unusable() -> None:
    registry = load_baseline_registry(REGISTRY)
    blocked = flagged_baseline_ids(registry)
    assert "tianxicode-swe-bench-live-pr-53" in blocked
    assert "tianxicode-swe-bench-live-pr-59" in blocked
    assert "slingshot-swe-bench-live" in blocked
    assert "agav-swe-bench-live" in blocked
    assert "ami-swe-bench-live" in blocked
    for entry in registry.entries:
        if entry.id in blocked:
            assert entry.usable_as_baseline is False
            assert entry.reason.strip()


def test_leaderboard_per_language_totals_are_rejected() -> None:
    aggregate = {"rust": {"resolved": 83, "total": 171}}
    assert is_leaderboard_aggregate_payload(aggregate) is True
    with pytest.raises(LiveBaselineError, match="leaderboard aggregate"):
        ingest_result_rows(aggregate, source="leaderboard.json")


def test_ingest_run_directory_collects_instance_ids() -> None:
    run = ingest_run_directory(FIXTURES / "run_a", run_id="run_a")
    assert run.instance_ids == ("alpha__demo-1", "alpha__demo-2")


def test_ranking_uses_shared_instance_intersection_only() -> None:
    run_a = ingest_run_directory(FIXTURES / "run_a", run_id="run_a")
    run_b = ingest_run_directory(FIXTURES / "run_b", run_id="run_b")
    assert shared_instance_ids(run_a, run_b) == frozenset({"alpha__demo-1"})
    ranked = rank_runs_on_shared_rows([run_a, run_b])
    assert len(ranked) == 2
    assert ranked[0].evaluated_count == 1
    assert ranked[0].resolved_count == 1
    assert ranked[0].resolve_rate == pytest.approx(1.0)


def test_flagged_baselines_excluded_from_ranking_pool() -> None:
    registry = load_baseline_registry(REGISTRY)
    run_a = ingest_run_directory(FIXTURES / "run_a", run_id="candidate")
    run_b = ingest_run_directory(FIXTURES / "run_b", run_id="reference")
    usable = filter_usable_runs(
        [run_a, run_b],
        registry,
        baseline_id_for_run={"reference": "agav-swe-bench-live"},
    )
    assert [run.run_id for run in usable] == ["candidate"]


def test_byte_identical_gold_src_ignores_test_files() -> None:
    base_files = {"src/lib.py": "old\n", "tests/test_lib.py": "check\n"}
    gold_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/tests/test_lib.py b/tests/test_lib.py\n"
        "--- a/tests/test_lib.py\n"
        "+++ b/tests/test_lib.py\n"
        "@@ -1 +1 @@\n"
        "-check\n"
        "+assert\n"
    )
    model_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    test_patch = (
        "diff --git a/tests/test_lib.py b/tests/test_lib.py\n"
        "--- a/tests/test_lib.py\n"
        "+++ a/tests/test_lib.py\n"
        "@@ -1 +1 @@\n"
        "-check\n"
        "+assert\n"
    )
    assert byte_identical_gold_src(
        base_files=base_files,
        gold_patch=gold_patch,
        model_patch=model_patch,
        test_patch=test_patch,
    )


def test_byte_identical_gold_src_false_when_src_differs() -> None:
    base_files = {"src/lib.py": "old\n"}
    gold_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    model_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+other\n"
    )
    assert not byte_identical_gold_src(
        base_files=base_files,
        gold_patch=gold_patch,
        model_patch=model_patch,
    )


def test_contains_gold_test_patch_detects_embedded_test_diff() -> None:
    test_patch = (
        "diff --git a/tests/test_lib.py b/tests/test_lib.py\n"
        "--- a/tests/test_lib.py\n"
        "+++ b/tests/test_lib.py\n"
        "@@ -1 +1 @@\n"
        "-check\n"
        "+assert\n"
    )
    model_patch = "prefix\n" + test_patch + "\nsuffix"
    assert contains_gold_test_patch(model_patch=model_patch, test_patch=test_patch)


def test_annotate_row_hygiene_sets_flags() -> None:
    row = TaskResultRow(
        instance_id="alpha__demo-1",
        resolved=True,
        model_patch=(
            "diff --git a/src/lib.py b/src/lib.py\n"
            "--- a/src/lib.py\n"
            "+++ b/src/lib.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
    )
    gold_patch = row.model_patch or ""
    annotated = annotate_row_hygiene(
        row,
        gold_patch=gold_patch,
        test_patch=None,
        base_files={"src/lib.py": "old\n"},
    )
    assert annotated.byte_identical_gold_src is True
    assert annotated.contains_gold_test_patch is False


def test_git_blob_sha1_matches_git() -> None:
    payload = b"hello\n"
    assert git_blob_sha1(payload) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_per_task_row_json_roundtrip_fixture() -> None:
    path = FIXTURES / "run_a" / "alpha__demo-1" / "result.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    rows = ingest_result_rows(row, source=str(path))
    assert rows[0].instance_id == "alpha__demo-1"
