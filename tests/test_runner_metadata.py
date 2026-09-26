"""Run metadata must not be a precondition for running a benchmark.

Git provenance describes a run; it does not enable one. A benchmark launched
from an installed wheel, a source tarball, or a container layer without a
`.git` directory has no provenance to record and must still produce a report.
"""

from __future__ import annotations

import string
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from benchmarks.runner import (
    UNKNOWN_GIT_INFO,
    _build_run_id,
    build_metadata,
    get_git_info,
)


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["git"], 0, stdout=stdout, stderr="")


def test_git_info_outside_a_repository_falls_back(tmp_path: Path, monkeypatch) -> None:
    """The previous version raised CalledProcessError and killed the run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    commit, branch = get_git_info()
    assert commit == UNKNOWN_GIT_INFO
    assert branch == UNKNOWN_GIT_INFO


def test_build_metadata_outside_a_repository_still_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    metadata = build_metadata("run-without-git", seed=7)
    assert metadata.git_commit == UNKNOWN_GIT_INFO
    assert metadata.git_branch == UNKNOWN_GIT_INFO
    assert metadata.run_name == "run-without-git"
    assert metadata.seed == 7
    assert metadata.run_id


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 5),
        FileNotFoundError("git"),
    ],
)
def test_git_info_survives_every_subprocess_failure(failure: Exception) -> None:
    """git missing from PATH, or hung, must degrade the same way."""
    with mock.patch("benchmarks.runner.subprocess.run", side_effect=failure):
        assert get_git_info() == (UNKNOWN_GIT_INFO, UNKNOWN_GIT_INFO)


def test_git_commands_are_given_a_timeout() -> None:
    """A hung git invocation must not hang the benchmark indefinitely."""
    with mock.patch(
        "benchmarks.runner.subprocess.run", return_value=_completed("abc123\n")
    ) as run:
        get_git_info()

    assert run.call_args_list, "git was never invoked"
    for call in run.call_args_list:
        assert call.kwargs.get("timeout"), f"no timeout passed: {call}"
        assert call.kwargs.get("check") is True
        assert call.kwargs.get("capture_output") is True
        assert call.kwargs.get("text") is True


def test_blank_git_output_is_treated_as_unknown() -> None:
    with mock.patch(
        "benchmarks.runner.subprocess.run", return_value=_completed("  \n")
    ):
        assert get_git_info() == (UNKNOWN_GIT_INFO, UNKNOWN_GIT_INFO)


def test_git_missing_from_path_is_treated_as_unknown() -> None:
    with mock.patch(
        "benchmarks.runner.subprocess.run", side_effect=FileNotFoundError("git")
    ):
        assert get_git_info() == (UNKNOWN_GIT_INFO, UNKNOWN_GIT_INFO)


def test_git_is_invoked_with_literal_argv() -> None:
    """Semgrep only accepts a fully static argv list at each call site."""
    with mock.patch(
        "benchmarks.runner.subprocess.run", return_value=_completed("abc123\n")
    ) as run:
        get_git_info()

    assert [call.args[0] for call in run.call_args_list] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    ]


def test_run_ids_do_not_collide_within_one_millisecond() -> None:
    """Parallel runs of the same commit must not overwrite each other's reports."""
    with mock.patch("benchmarks.runner.time.time", return_value=1_767_225_600.123):
        ids = {_build_run_id("abcdef1234567890") for _ in range(64)}
    assert len(ids) == 64, "run ids collide when the clock does not advance"


def test_run_id_still_carries_the_commit_prefix() -> None:
    run_id = _build_run_id("abcdef1234567890")
    assert "abcdef1" in run_id, "the commit prefix identifies which build ran"


def test_run_id_ends_with_a_full_random_hex_suffix() -> None:
    """It is the suffix that separates concurrent runs, so check all of it.

    `endswith(tuple(HEXDIGITS))` would only constrain the final character.
    """
    suffix = _build_run_id("abcdef1234567890").rsplit("-", 1)[-1]
    assert len(suffix) == 8, f"expected 8 hex characters, got {suffix!r}"
    assert set(suffix) <= set(string.hexdigits.lower()), f"not hex: {suffix!r}"


def test_run_id_is_a_safe_single_path_component() -> None:
    """The run id is interpolated straight into report paths."""
    for commit in ("abcdef1234567890", UNKNOWN_GIT_INFO):
        run_id = _build_run_id(commit)
        assert "/" not in run_id
        assert "\\" not in run_id
        assert ".." not in run_id
