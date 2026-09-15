"""Resume journal, stale-cell recovery, and crash-point tests (RM-1319)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.journal import (
    CellState,
    CommitPoint,
    JournalError,
    encode_record,
    recover_journal_file,
    replay_journal,
)
from benchmarks.matrix import (
    DuplicateCellIdError,
    IncompatibleMatrixError,
    MatrixDefinition,
    MatrixError,
    RetryPolicy,
    sha256_hex,
)
from benchmarks.matrix_runner import (
    CrashPlan,
    MatrixHooks,
    SimulatedInterrupt,
    load_matrix_config,
    run_matrix,
)
from scripts.run_matrix import main as run_matrix_main


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps({"prompt": "hello", "reference": "world"}) + "\n",
        encoding="utf-8",
    )


def write_matrix_config(
    tmp_path: Path,
    *,
    name: str = "fixture-resume",
    revision: str = "rev1",
    models: list[dict] | None = None,
    quantization: list[str] | None = None,
    extra_datasets: int = 0,
) -> Path:
    dataset_path = tmp_path / "data.jsonl"
    _write_dataset(dataset_path)
    datasets = [
        {
            "name": "smoke",
            "source": "jsonl",
            "path": str(dataset_path),
            "split": "validation",
            "max_samples": 1,
        }
    ]
    for index in range(extra_datasets):
        extra = tmp_path / f"data-{index}.jsonl"
        _write_dataset(extra)
        datasets.append(
            {
                "name": f"smoke-{index}",
                "source": "jsonl",
                "path": str(extra),
                "split": "validation",
                "max_samples": 1,
            }
        )
    config = {
        "matrix_name": name,
        "config_revision": revision,
        "seed": 7,
        "retry": {"retry_failed": False, "max_attempts": 1},
        "models": models
        or [{"backend": "mock", "name": "toy-alpha", "revision": "r1"}],
        "quantization": quantization or ["fp16", "awq"],
        "datasets": datasets,
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_cell_id_is_stable_and_sensitive_to_identity(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path, revision="rev1")
    definition = load_matrix_config(config_path)
    again = load_matrix_config(config_path)
    assert [cell.cell_id() for cell in definition.cells] == [
        cell.cell_id() for cell in again.cells
    ]
    assert len({cell.cell_id() for cell in definition.cells}) == 2
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["config_revision"] = "rev2"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    changed = load_matrix_config(config_path)
    assert {cell.cell_id() for cell in definition.cells}.isdisjoint(
        {cell.cell_id() for cell in changed.cells}
    )


def test_duplicate_cell_ids_are_rejected_before_execution() -> None:
    raw = {
        "matrix_name": "dupes",
        "config_revision": "rev1",
        "seed": 1,
        "models": [
            {"backend": "mock", "name": "toy", "revision": "r1"},
            {"backend": "mock", "name": "toy", "revision": "r1"},
        ],
        "quantization": ["fp16"],
        "datasets": [
            {"name": "smoke", "source": "jsonl", "path": "unused.jsonl", "split": "validation"}
        ],
    }
    with pytest.raises(DuplicateCellIdError, match="duplicate cell id"):
        MatrixDefinition.from_dict(raw)


def test_incompatible_matrix_definition_is_rejected(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path, revision="rev1")
    output = tmp_path / "out"
    run_matrix(config_path, output)
    write_matrix_config(tmp_path, revision="rev2")
    with pytest.raises(IncompatibleMatrixError, match="does not match the journal"):
        run_matrix(config_path, output)


def test_corrupt_tail_keeps_earlier_records(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    first = {"kind": "header", "schema": "combine.matrix.journal.v1"}
    second = {"kind": "transition", "cell_id": "abc", "state": "succeeded", "attempt": 1}
    path.write_bytes(encode_record(first) + encode_record(second) + b'{"v":1,"crc32":"deadbeef","body":')
    replay = recover_journal_file(path)
    assert replay.truncated_tail is True
    assert len(replay.records) == 2
    assert replay.records[1]["state"] == "succeeded"
    assert path.read_bytes() == encode_record(first) + encode_record(second)
    # A second replay must not drop the recovered prefix.
    again = replay_journal(path)
    assert again.truncated_tail is False
    assert len(again.records) == 2


def test_crc_mismatch_at_tail_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    body = {"kind": "header", "schema": "combine.matrix.journal.v1"}
    bad = (
        json.dumps(
            {"body": {"kind": "transition", "state": "running"}, "crc32": "00000000", "v": 1},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    path.write_bytes(encode_record(body) + bad.encode("utf-8"))
    replay = recover_journal_file(path)
    assert replay.truncated_tail is True
    assert [record["kind"] for record in replay.records] == ["header"]


def test_kill_and_restart_does_not_lose_or_duplicate_results(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path)
    definition = load_matrix_config(config_path)
    first_id, second_id = [cell.cell_id() for cell in definition.cells]
    output = tmp_path / "out"

    with pytest.raises(SimulatedInterrupt):
        run_matrix(
            config_path,
            output,
            hooks=MatrixHooks(
                crash=CrashPlan(
                    point=CommitPoint.JOURNAL_WRITE,
                    cell_id=second_id,
                    state=CellState.RUNNING,
                )
            ),
        )

    before = replay_journal(output / definition.name / "journal.jsonl")
    completed = [
        record
        for record in before.records
        if record.get("kind") == "transition" and record.get("state") == "succeeded"
    ]
    assert [record["cell_id"] for record in completed] == [first_id]
    first_artifact = json.loads(
        (output / definition.name / "cells" / first_id / "result.json").read_text()
    )

    resumed = run_matrix(config_path, output)
    cell_ids = [cell["cell_id"] for cell in resumed.aggregate["cells"]]
    assert cell_ids == [first_id, second_id]
    assert cell_ids == list(dict.fromkeys(cell_ids))
    by_id = {cell["cell_id"]: cell for cell in resumed.aggregate["cells"]}
    assert by_id[first_id]["disposition"] == "skipped"
    assert by_id[second_id]["disposition"] == "succeeded"
    assert json.loads(
        (output / definition.name / "cells" / first_id / "result.json").read_text()
    ) == first_artifact
    succeeded_or_skipped = [
        record
        for record in replay_journal(resumed.journal_path).records
        if record.get("kind") == "transition" and record.get("state") in {"succeeded", "skipped"}
    ]
    # First cell: one succeeded + one skipped. Second cell: one succeeded.
    assert [record["cell_id"] for record in succeeded_or_skipped] == [
        first_id,
        first_id,
        second_id,
    ]


def test_success_is_skipped_only_after_checksum_and_fingerprint_match(
    tmp_path: Path,
) -> None:
    config_path = write_matrix_config(tmp_path, quantization=["fp16"])
    output = tmp_path / "out"
    first = run_matrix(config_path, output)
    cell_id = first.definition.cells[0].cell_id()
    artifact = output / first.definition.name / "cells" / cell_id / "result.json"
    original = artifact.read_bytes()

    second = run_matrix(config_path, output)
    assert second.statuses[cell_id].state is CellState.SKIPPED
    assert second.aggregate["cells"][0]["disposition"] == "skipped"

    artifact.write_bytes(original + b"\n")
    stale = run_matrix(config_path, output)
    assert stale.statuses[cell_id].state is CellState.SUCCEEDED
    assert stale.statuses[cell_id].retry_count >= 1
    assert stale.statuses[cell_id].artifact_checksum == sha256_hex(artifact.read_bytes())


def test_fingerprint_mismatch_requeues_successful_cell(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path, quantization=["fp16"])
    output = tmp_path / "out"
    result = run_matrix(config_path, output)
    journal_path = result.journal_path
    replay = replay_journal(journal_path)
    rewritten: list[dict] = []
    for record in replay.records:
        if record.get("kind") == "transition" and record.get("state") == "succeeded":
            record = dict(record)
            record["fingerprint"] = "0" * 64
        rewritten.append(record)
    journal_path.write_bytes(b"".join(encode_record(record) for record in rewritten))

    resumed = run_matrix(config_path, output)
    cell_id = result.definition.cells[0].cell_id()
    assert resumed.statuses[cell_id].state is CellState.SUCCEEDED
    assert resumed.statuses[cell_id].fingerprint == result.definition.cells[0].fingerprint()


@pytest.mark.parametrize(
    "point,state",
    [
        (CommitPoint.JOURNAL_WRITE, CellState.RUNNING),
        (CommitPoint.RESULT_COMMIT, None),
        (CommitPoint.JOURNAL_WRITE, CellState.SUCCEEDED),
        (CommitPoint.AGGREGATE_UPDATE, None),
    ],
)
def test_interrupt_at_documented_commit_points(
    tmp_path: Path,
    point: CommitPoint,
    state: CellState | None,
) -> None:
    config_path = write_matrix_config(tmp_path, quantization=["fp16"])
    definition = load_matrix_config(config_path)
    cell_id = definition.cells[0].cell_id()
    output = tmp_path / "out"
    crash_cell = None if point is CommitPoint.AGGREGATE_UPDATE else cell_id
    with pytest.raises(SimulatedInterrupt) as info:
        run_matrix(
            config_path,
            output,
            hooks=MatrixHooks(
                crash=CrashPlan(point=point, cell_id=crash_cell, state=state)
            ),
        )
    assert info.value.point is point

    resumed = run_matrix(config_path, output)
    assert len(resumed.aggregate["cells"]) == 1
    cell = resumed.aggregate["cells"][0]
    assert cell["cell_id"] == cell_id
    assert cell["disposition"] in {"succeeded", "skipped"}
    artifact = output / definition.name / "cells" / cell_id / "result.json"
    assert artifact.is_file()
    if point is CommitPoint.AGGREGATE_UPDATE:
        assert cell["disposition"] == "skipped"
        assert cell["retry_count"] == 0
    else:
        # Interrupted before a durable succeeded record: requeued, then completed.
        assert cell["attempt"] >= 1


def test_failed_cells_follow_retry_policy(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path, quantization=["fp16"])
    definition = load_matrix_config(config_path)
    cell_id = definition.cells[0].cell_id()
    output = tmp_path / "out"

    first = run_matrix(
        config_path,
        output,
        retry=RetryPolicy(retry_failed=False, max_attempts=1),
        hooks=MatrixHooks(fail_on_attempt={cell_id: 1}),
    )
    assert first.statuses[cell_id].state is CellState.FAILED
    assert first.aggregate["cells"][0]["retry_count"] == 0
    assert first.aggregate["cells"][0]["disposition"] == "failed"

    kept = run_matrix(
        config_path,
        output,
        retry=RetryPolicy(retry_failed=False, max_attempts=1),
    )
    assert kept.statuses[cell_id].state is CellState.FAILED

    retried = run_matrix(
        config_path,
        output,
        retry=RetryPolicy(retry_failed=True, max_attempts=2),
    )
    assert retried.statuses[cell_id].state is CellState.SUCCEEDED
    assert retried.aggregate["cells"][0]["retry_count"] == 1
    assert retried.aggregate["cells"][0]["disposition"] == "succeeded"


def test_sample_matrix_resume_round_trip(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    config_path = repo / "configs" / "matrix.sample.json"
    output = tmp_path / "reports"
    first = run_matrix(config_path, output)
    assert len(first.definition.cells) == 4
    assert all(status.state is CellState.SUCCEEDED for status in first.statuses.values())
    second = run_matrix(config_path, output)
    assert all(status.state is CellState.SKIPPED for status in second.statuses.values())
    assert second.aggregate["counts"]["skipped"] == 4
    ids = [cell["cell_id"] for cell in second.aggregate["cells"]]
    assert len(ids) == len(set(ids))
    for cell in second.aggregate["cells"]:
        assert cell["retry_count"] == 0
        assert cell["disposition"] == "skipped"


def test_aggregate_exposes_retry_count_and_disposition(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path)
    definition = load_matrix_config(config_path)
    second_id = definition.cells[1].cell_id()
    output = tmp_path / "out"
    with pytest.raises(SimulatedInterrupt):
        run_matrix(
            config_path,
            output,
            hooks=MatrixHooks(
                crash=CrashPlan(
                    point=CommitPoint.RESULT_COMMIT,
                    cell_id=second_id,
                )
            ),
        )
    resumed = run_matrix(config_path, output)
    by_id = {cell["cell_id"]: cell for cell in resumed.aggregate["cells"]}
    assert by_id[definition.cells[0].cell_id()]["disposition"] == "skipped"
    assert by_id[second_id]["disposition"] == "succeeded"
    assert by_id[second_id]["retry_count"] == 1
    assert "retry_count" in resumed.aggregate["cells"][0]
    assert "disposition" in resumed.aggregate["cells"][0]


def test_cli_rejects_zero_max_attempts(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path, quantization=["fp16"])
    with pytest.raises(MatrixError, match="max_attempts"):
        run_matrix_main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--max-attempts",
                "0",
            ]
        )


def test_malformed_transition_is_journal_error(tmp_path: Path) -> None:
    config_path = write_matrix_config(tmp_path, quantization=["fp16"])
    output = tmp_path / "out"
    result = run_matrix(config_path, output)
    result.journal_path.write_bytes(
        result.journal_path.read_bytes()
        + encode_record({"kind": "transition", "state": "running", "attempt": 1})
    )
    with pytest.raises(JournalError, match="malformed"):
        run_matrix(config_path, output)
