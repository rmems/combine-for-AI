"""Run metadata must not be a precondition for running a benchmark.

Git provenance describes a run; it does not enable one. A benchmark launched
from an installed wheel, a source tarball, or a container layer without a
`.git` directory has no provenance to record and must still produce a report.
"""

from __future__ import annotations

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
    with mock.patch("benchmarks.runner.subprocess.check_output", side_effect=failure):
        assert get_git_info() == (UNKNOWN_GIT_INFO, UNKNOWN_GIT_INFO)


def test_git_commands_are_given_a_timeout() -> None:
    """A hung git invocation must not hang the benchmark indefinitely."""
    with mock.patch(
        "benchmarks.runner.subprocess.check_output", return_value="abc123\n"
    ) as check_output:
        get_git_info()

    assert check_output.call_args_list, "git was never invoked"
    for call in check_output.call_args_list:
        assert call.kwargs.get("timeout"), f"no timeout passed: {call}"


def test_blank_git_output_is_treated_as_unknown() -> None:
    with mock.patch("benchmarks.runner.subprocess.check_output", return_value="  \n"):
        assert get_git_info() == (UNKNOWN_GIT_INFO, UNKNOWN_GIT_INFO)


def test_git_missing_from_path_is_treated_as_unknown() -> None:
    with mock.patch("benchmarks.runner.shutil.which", return_value=None):
        assert get_git_info() == (UNKNOWN_GIT_INFO, UNKNOWN_GIT_INFO)


def test_git_is_invoked_by_absolute_path() -> None:
    """A `git` planted earlier in PATH must not be what a run executes."""
    with mock.patch(
        "benchmarks.runner.shutil.which", return_value="/usr/bin/git"
    ), mock.patch(
        "benchmarks.runner.subprocess.check_output", return_value="abc123\n"
    ) as check_output:
        get_git_info()

    assert check_output.call_args_list, "git was never invoked"
    for call in check_output.call_args_list:
        argv = call.args[0]
        assert argv[0] == "/usr/bin/git", f"not an absolute path: {argv[0]!r}"


def test_run_ids_do_not_collide_within_one_millisecond() -> None:
    """Parallel runs of the same commit must not overwrite each other's reports."""
    with mock.patch("benchmarks.runner.time.time", return_value=1_767_225_600.123):
        ids = {_build_run_id("abcdef1234567890") for _ in range(64)}
    assert len(ids) == 64, "run ids collide when the clock does not advance"


def test_run_id_still_carries_the_commit_prefix() -> None:
    run_id = _build_run_id("abcdef1234567890")
    assert "abcdef1" in run_id, "the commit prefix identifies which build ran"
    assert run_id.endswith(tuple("0123456789abcdef"))


def test_run_id_is_a_safe_single_path_component() -> None:
    """The run id is interpolated straight into report paths."""
    for commit in ("abcdef1234567890", UNKNOWN_GIT_INFO):
        run_id = _build_run_id(commit)
        assert "/" not in run_id
        assert "\\" not in run_id
        assert ".." not in run_id
