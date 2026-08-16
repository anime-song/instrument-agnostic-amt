import argparse
from pathlib import Path

import pytest

from scripts import colab_t4_regression


EXPECTED_COMMIT = "0c254ecd7a393a4cc7499c28eced2d261a6c479d"


def test_parse_args_requires_expected_commit() -> None:
    with pytest.raises(SystemExit):
        colab_t4_regression.parse_args(["--branch", "topic"])


def test_parse_args_uses_upstream_repository_by_default() -> None:
    args = colab_t4_regression.parse_args(
        [
            "--branch",
            "topic",
            "--expected-commit",
            EXPECTED_COMMIT,
        ]
    )

    assert args.repo_url == (
        "https://github.com/anime-song/instrument-agnostic-amt.git"
    )
    assert args.expected_commit == EXPECTED_COMMIT


def test_parse_args_rejects_non_full_commit_sha() -> None:
    with pytest.raises(SystemExit):
        colab_t4_regression.parse_args(
            [
                "--branch",
                "topic",
                "--expected-commit",
                "0c254ec",
            ]
        )


def test_verify_expected_commit_accepts_exact_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colab_t4_regression,
        "_capture",
        lambda *args, **kwargs: EXPECTED_COMMIT,
    )

    actual = colab_t4_regression._verify_expected_commit(
        Path("/tmp/repository"), EXPECTED_COMMIT.upper()
    )

    assert actual == EXPECTED_COMMIT


def test_verify_expected_commit_rejects_different_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_commit = "f" * 40
    monkeypatch.setattr(
        colab_t4_regression,
        "_capture",
        lambda *args, **kwargs: actual_commit,
    )

    with pytest.raises(RuntimeError, match="does not match expected commit"):
        colab_t4_regression._verify_expected_commit(
            Path("/tmp/repository"), EXPECTED_COMMIT
        )


def test_main_stops_before_dependency_install_when_commit_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        colab_t4_regression,
        "parse_args",
        lambda: argparse.Namespace(
            repo_url="https://example.invalid/repository.git",
            branch="topic",
            expected_commit=EXPECTED_COMMIT,
        ),
    )
    monkeypatch.setattr(
        colab_t4_regression.tempfile,
        "mkdtemp",
        lambda **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        colab_t4_regression,
        "_run",
        lambda command, **kwargs: commands.append(list(command)),
    )

    def reject_commit(*args: object, **kwargs: object) -> str:
        raise RuntimeError("does not match expected commit")

    monkeypatch.setattr(
        colab_t4_regression, "_verify_expected_commit", reject_commit
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
