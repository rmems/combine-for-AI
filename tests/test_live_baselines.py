from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from combine_for_ai.live_baselines import (
    LiveBaselineError,
    _apply_unified_hunks,
    annotate_row_hygiene,
    byte_identical_gold_src,
    contains_gold_test_patch,
    filter_usable_runs,
    flagged_baseline_ids,
    git_blob_sha1,
    ingest_result_rows,
    ingest_run_directory,
    ingest_submission_directory,
    is_leaderboard_aggregate_payload,
    is_submission_results_payload,
    is_test_path,
    load_base_files,
    load_baseline_registry,
    load_pinned_instance_ids,
    rank_runs_on_shared_rows,
    rows_from_submission_bundle,
    shared_instance_ids,
    RunSnapshot,
    TaskResultRow,
)
from combine_for_ai.rmems_eval_v0_1 import rank_runs as rank_rmems_eval_v0_1

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "swe_bench_live"
SUBMISSIONS = FIXTURES / "submissions"
REGISTRY = Path(__file__).resolve().parents[1] / "data" / "swe_bench_live" / "baselines.json"


def test_registry_pr59_documents_87_of_171_and_rmems_trajectory_note() -> None:
    registry = load_baseline_registry(REGISTRY)
    entry = next(e for e in registry.entries if e.id == "tianxicode-swe-bench-live-pr-59")
    assert "87/171" in entry.reason
    assert "83/171" in entry.reason
    assert "rmems trajectory" in entry.reason.lower()


def test_registry_marks_known_bad_submissions_unusable() -> None:
    registry = load_baseline_registry(REGISTRY)
    blocked = flagged_baseline_ids(registry)
    assert "tianxicode-swe-bench-live-pr-53" in blocked
    assert "tianxicode-swe-bench-live-pr-57" in blocked
    assert "tianxicode-swe-bench-live-pr-58" in blocked
    assert "tianxicode-swe-bench-live-pr-59" in blocked
    assert "slingshot-swe-bench-live" in blocked
    assert "agav-swe-bench-live" in blocked
    assert "ami-swe-bench-live" in blocked
    for entry in registry.entries:
        if entry.id in blocked:
            assert entry.usable_as_baseline is False
            assert entry.reason.strip()


def test_empty_ingest_payload_is_rejected() -> None:
    with pytest.raises(LiveBaselineError, match="empty per-task"):
        ingest_result_rows({}, source="empty.json")


def test_leaderboard_per_language_totals_are_rejected() -> None:
    aggregate = {"rust": {"resolved": 83, "total": 171}}
    assert is_leaderboard_aggregate_payload(aggregate) is True
    with pytest.raises(LiveBaselineError, match="leaderboard aggregate"):
        ingest_result_rows(aggregate, source="leaderboard.json")


def test_submission_results_json_is_not_leaderboard_aggregate() -> None:
    path = SUBMISSIONS / "tianxicode_rust_pr59" / "deepseek-flash" / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert is_submission_results_payload(payload) is True
    assert is_leaderboard_aggregate_payload(payload) is False


def test_ingest_tianxicode_rust_submission_fixture() -> None:
    run = ingest_submission_directory(
        SUBMISSIONS / "tianxicode_rust_pr59" / "deepseek-flash",
        run_id="tianxicode-rust-pr59",
    )
    assert run.instance_ids == (
        "linebender__vello-1100",
        "apache__datafusion-17233",
    )
    by_id = {row.instance_id: row for row in run.rows}
    assert by_id["linebender__vello-1100"].resolved is True
    assert by_id["apache__datafusion-17233"].resolved is False
    assert by_id["linebender__vello-1100"].model_patch
    assert "diff --git" in (by_id["linebender__vello-1100"].model_patch or "")


def test_ingest_agav_lite_submission_fixture() -> None:
    run = ingest_run_directory(SUBMISSIONS / "agav_lite" / "gpt-5.5", run_id="agav-lite")
    assert len(run.rows) == 2
    assert all(row.model_patch for row in run.rows)


def test_resolved_ids_count_as_resolved() -> None:
    results = {
        "submitted_ids": ["repo__a-1", "repo__b-2"],
        "resolved_ids": ["repo__a-1"],
        "unresolved_ids": ["repo__b-2"],
    }
    preds = {
        "repo__a-1": {"model_patch": "diff --git a/a.py b/a.py\n"},
        "repo__b-2": {"model_patch": "diff --git a/b.py b/b.py\n"},
    }
    rows = rows_from_submission_bundle(results, preds, source="ami-resolved.json")
    by_id = {row.instance_id: row for row in rows}
    assert by_id["repo__a-1"].resolved is True
    assert by_id["repo__b-2"].resolved is False
    assert is_submission_results_payload(results) is True


def test_missing_resolved_and_success_ids_fails_loudly() -> None:
    results = {"submitted_ids": ["repo__a-1"]}
    preds = {"repo__a-1": {"model_patch": ""}}
    with pytest.raises(LiveBaselineError, match="silent zero-resolved"):
        rows_from_submission_bundle(results, preds, source="unknown.json")


def test_unknown_results_format_fails_loudly() -> None:
    with pytest.raises(LiveBaselineError, match="unknown submission results format"):
        rows_from_submission_bundle({"note": "counts only"}, {}, source="counts.json")


def test_bare_submission_results_json_requires_preds_companion() -> None:
    path = SUBMISSIONS / "agav_lite" / "gpt-5.5" / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(LiveBaselineError, match="preds.json"):
        ingest_result_rows(payload, source=str(path))


def test_is_test_path_detects_rust_tests_dir_without_test_patch() -> None:
    assert is_test_path("crates/foo/tests/integration_test.rs") is True
    assert is_test_path("src/lib.rs") is False
    assert is_test_path("pkg/widget_test.go") is True
    assert is_test_path("frontend/component.test.tsx") is True


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


def _snapshot_from_resolved(
    run_id: str,
    resolved_ids: Sequence[str],
    reported_unresolved_ids: Sequence[str] = (),
) -> RunSnapshot:
    rows = [TaskResultRow(instance_id=iid, resolved=True) for iid in resolved_ids]
    rows.extend(
        TaskResultRow(instance_id=iid, resolved=False) for iid in reported_unresolved_ids
    )
    return RunSnapshot(
        run_id=run_id,
        instance_ids=tuple(row.instance_id for row in rows),
        rows=tuple(rows),
    )


def test_pinned_ranking_equal_resolves_despite_missing_rows() -> None:
    pinned = load_pinned_instance_ids()
    assert len(pinned) == 239
    resolved = pinned[:30]
    omitted = pinned[30:]
    base = _snapshot_from_resolved("BASE", resolved)
    synthetic = _snapshot_from_resolved(
        "SYNTHETIC_ONLY", resolved, reported_unresolved_ids=omitted
    )
    ranked = rank_rmems_eval_v0_1([base, synthetic])
    by_id = {item.run_id: item for item in ranked}
    assert by_id["BASE"].resolved_count == by_id["SYNTHETIC_ONLY"].resolved_count == 30
    assert by_id["BASE"].evaluated_count == 239
    assert by_id["SYNTHETIC_ONLY"].evaluated_count == 239
    assert by_id["BASE"].missing_count == 209
    assert by_id["SYNTHETIC_ONLY"].missing_count == 0


def test_pinned_denominator_ignores_omitted_skips_in_run_report() -> None:
    pinned = load_pinned_instance_ids()
    skipped = pinned[:9]
    resolved = pinned[9:39]
    reported = _snapshot_from_resolved("sparse-report", resolved)
    assert set(skipped).isdisjoint(reported.instance_ids)
    ranked = rank_rmems_eval_v0_1([reported])
    assert ranked[0].evaluated_count == 239
    assert ranked[0].resolved_count == 30
    assert ranked[0].missing_count == 209


def test_v0_1_default_path_always_uses_239_pin() -> None:
    pinned = load_pinned_instance_ids()
    resolved = pinned[:30]
    run = _snapshot_from_resolved("v0.1", resolved)
    ranked = rank_rmems_eval_v0_1([run])
    assert ranked[0].evaluated_count == 239
    assert ranked[0].resolved_count == 30
    assert ranked[0].missing_count == 209


def test_v0_1_out_of_pin_row_raises_named_ids() -> None:
    pinned = load_pinned_instance_ids()
    run = _snapshot_from_resolved(
        "stray",
        [pinned[0], "linebender__vello-1100"],
    )
    with pytest.raises(LiveBaselineError, match="linebender__vello-1100"):
        rank_rmems_eval_v0_1([run])


def test_harness_row_without_resolved_fails_loudly(tmp_path: Path) -> None:
    cell = tmp_path / "x__y-1"
    cell.mkdir()
    (cell / "result.json").write_text(
        json.dumps({"instance_id": "x__y-1", "status": "success"}),
        encoding="utf-8",
    )
    with pytest.raises(LiveBaselineError, match="missing boolean resolved"):
        ingest_run_directory(tmp_path, run_id="harness")


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


def test_byte_identical_gold_src_fails_closed_when_base_path_missing() -> None:
    gold_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert not byte_identical_gold_src(
        base_files={},
        gold_patch=gold_patch,
        model_patch=gold_patch,
    )


def test_byte_identical_gold_src_uses_loaded_base_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("old\n", encoding="utf-8")
    gold_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    loaded = load_base_files(tmp_path, ["src/lib.py"])
    assert loaded == {"src/lib.py": "old\n"}
    assert byte_identical_gold_src(
        base_files=loaded,
        gold_patch=gold_patch,
        model_patch=gold_patch,
    )


def test_byte_identical_gold_src_handles_new_file_hunks() -> None:
    gold_patch = (
        "diff --git a/src/created.py b/src/created.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/created.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+hello\n"
        "+world\n"
    )
    assert byte_identical_gold_src(
        base_files={},
        gold_patch=gold_patch,
        model_patch=gold_patch,
    )


def test_byte_identical_gold_src_keeps_hunk_lines_starting_with_dashes() -> None:
    base_files = {"src/a.sql": "select 1;\n-- old\nselect 2;\nend\n"}
    patch = (
        "diff --git a/src/a.sql b/src/a.sql\n"
        "--- a/src/a.sql\n"
        "+++ b/src/a.sql\n"
        "@@ -1,4 +1,3 @@\n"
        " select 1;\n"
        "--- old\n"
        " select 2;\n"
        " end\n"
    )
    assert byte_identical_gold_src(
        base_files=base_files, gold_patch=patch, model_patch=patch
    )


def test_unified_hunk_insert_after_line_uses_zero_old_count() -> None:
    out = _apply_unified_hunks("a\nb\nc\n", "@@ -2,0 +3,1 @@\n+X\n")
    assert out == "a\nb\nX\nc\n"


def test_byte_identical_gold_src_respects_hunk_line_offset() -> None:
    base_lines = [f"l{i}\n" for i in range(1, 21)]
    base = "".join(base_lines)
    gold_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -16,3 +16,3 @@\n"
        " l15\n"
        "-l16\n"
        "+FIX\n"
        " l17\n"
    )
    model_patch = (
        "diff --git a/src/lib.py b/src/lib.py\n"
        "--- a/src/lib.py\n"
        "+++ b/src/lib.py\n"
        "@@ -3,3 +3,3 @@\n"
        " l2\n"
        "-l3\n"
        "+FIX\n"
        " l4\n"
    )
    assert not byte_identical_gold_src(
        base_files={"src/lib.py": base},
        gold_patch=gold_patch,
        model_patch=model_patch,
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
