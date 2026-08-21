from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from scripts import colab_t4_regression

TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_runner_cli_requires_branch_and_full_commit_sha() -> None:
    with pytest.raises(SystemExit):
        colab_t4_regression.parse_args(["--expected-commit", TEST_COMMIT])
    with pytest.raises(SystemExit):
        colab_t4_regression.parse_args(["--branch", "topic"])
    with pytest.raises(SystemExit):
        colab_t4_regression.parse_args(
            ["--branch", "topic", "--expected-commit", TEST_COMMIT[:7]]
        )

    args = colab_t4_regression.parse_args(
        ["--branch", "topic", "--expected-commit", TEST_COMMIT.upper()]
    )
    assert args.repo_url == (
        "https://github.com/anime-song/instrument-agnostic-amt.git"
    )
    assert args.branch == "topic"
    assert args.expected_commit == TEST_COMMIT


def test_runner_verifies_the_exact_checked_out_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_commits = iter((TEST_COMMIT, "f" * 40))
    monkeypatch.setattr(
        colab_t4_regression,
        "_capture",
        lambda *_args, **_kwargs: next(actual_commits),
    )

    assert (
        colab_t4_regression._verify_expected_commit(
            Path("/tmp/repository"), TEST_COMMIT.upper()
        )
        == TEST_COMMIT
    )
    with pytest.raises(RuntimeError, match="does not match expected commit"):
        colab_t4_regression._verify_expected_commit(
            Path("/tmp/repository"), TEST_COMMIT
        )


def test_runner_stops_before_install_when_commit_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        colab_t4_regression,
        "parse_args",
        lambda: argparse.Namespace(
            repo_url="https://example.invalid/repository.git",
            branch="topic",
            expected_commit=TEST_COMMIT,
        ),
    )
    monkeypatch.setattr(
        colab_t4_regression.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        colab_t4_regression,
        "_run",
        lambda command, **_kwargs: commands.append(list(command)),
    )
    monkeypatch.setattr(
        colab_t4_regression,
        "_verify_expected_commit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("does not match expected commit")
        ),
    )

    with pytest.raises(RuntimeError, match="does not match expected commit"):
        colab_t4_regression.main()

    assert commands == [
        ["nvidia-smi"],
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "topic",
            "--",
            "https://example.invalid/repository.git",
            str(tmp_path / "repository"),
        ],
    ]


def test_runner_uses_locked_environment_and_runs_compile_regressions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, object]] = []
    worktree = tmp_path / "repository"
    monkeypatch.setattr(
        colab_t4_regression,
        "parse_args",
        lambda: argparse.Namespace(
            repo_url="https://example.invalid/repository.git",
            branch="topic",
            expected_commit=TEST_COMMIT,
        ),
    )
    monkeypatch.setattr(
        colab_t4_regression.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path),
    )

    def fake_run(
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        events.append(
            (
                "run",
                (
                    list(command),
                    cwd,
                    None if env is None else env.get("RUN_ACCELERATOR_COMPILE_TEST"),
                ),
            )
        )

    monkeypatch.setattr(colab_t4_regression, "_run", fake_run)

    def verify(path: Path, expected_commit: str) -> str:
        events.append(("verify", (path, expected_commit)))
        return TEST_COMMIT

    monkeypatch.setattr(colab_t4_regression, "_verify_expected_commit", verify)

    colab_t4_regression.main()

    assert events[2] == ("verify", (worktree, TEST_COMMIT))
    commands = [payload for kind, payload in events if kind == "run"]
    assert commands[2] == (
        [sys.executable, "-m", "pip", "install", "--quiet", "uv==0.8.17"],
        None,
        None,
    )
    assert commands[3] == (
        ["uv", "sync", "--locked", "--all-extras"],
        worktree,
        None,
    )
    assert commands[4][0][:4] == ["uv", "run", "python", "-c"]
    assert "Expected Tesla T4" in commands[4][0][4]
    assert commands[5] == (["uv", "run", "pytest", "-q"], worktree, None)
    assert commands[6] == (
        ["uv", "run", "pytest", "-q", "tests/test_cuda_inference.py"],
        worktree,
        "1",
    )
