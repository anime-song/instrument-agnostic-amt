#!/usr/bin/env python3
"""Colab T4上で、指定ブランチの固定コミットに対するCUDA回帰を再現する。"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_REPOSITORY_URL = (
    "https://github.com/anime-song/instrument-agnostic-amt.git"
)
FULL_COMMIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
    )


def _full_commit_sha(value: str) -> str:
    if FULL_COMMIT_SHA_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "expected commit must be a full 40-character hexadecimal SHA"
        )
    return value.lower()


def _capture(command: Sequence[str], *, cwd: Path | None = None) -> str:
    print(f"+ {shlex.join(command)}", flush=True)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _verify_expected_commit(worktree: Path, expected_commit: str) -> str:
    actual_commit = _capture(["git", "rev-parse", "HEAD"], cwd=worktree)
    print(f"Checked out commit: {actual_commit}", flush=True)
    if actual_commit.lower() != expected_commit.lower():
        raise RuntimeError(
            f"Checked out HEAD {actual_commit} does not match expected commit "
            f"{expected_commit}"
        )
    return actual_commit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the inference regression suite on a Colab Tesla T4."
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPOSITORY_URL)
    parser.add_argument(
        "--branch",
        required=True,
        help="Remote branch to test (required to avoid testing main by mistake).",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        type=_full_commit_sha,
        help="Full 40-character commit SHA that the cloned HEAD must match.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    worktree = Path(tempfile.mkdtemp(prefix="amt-colab-t4-")) / "repository"

    _run(["nvidia-smi"])
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            str(args.branch),
            "--",
            str(args.repo_url),
            str(worktree),
        ]
    )
    _verify_expected_commit(worktree, args.expected_commit)
    # READMEの再現手順と同じuvで、lockfileの解決結果を固定する。
    _run([sys.executable, "-m", "pip", "install", "--quiet", "uv==0.8.17"])
    _run(["uv", "sync", "--locked", "--all-extras"], cwd=worktree)
    _run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "import torch; "
                "print('torch', torch.__version__); "
                "print('cuda', torch.version.cuda); "
                "device_name = torch.cuda.get_device_name(0); "
                "print('device', device_name); "
                "assert 'T4' in device_name.upper(), "
                "f'Expected Tesla T4, got {device_name}'; "
                "print('bf16', torch.cuda.is_bf16_supported())"
            ),
        ],
        cwd=worktree,
    )
    _run(["uv", "run", "pytest", "-q"], cwd=worktree)

    # 通常はopt-inの実機compile回帰も、このT4検証では必ず実行する。
    compile_environment = dict(os.environ)
    compile_environment["RUN_ACCELERATOR_COMPILE_TEST"] = "1"
    _run(
        ["uv", "run", "pytest", "-q", "tests/test_cuda_inference.py"],
        cwd=worktree,
        env=compile_environment,
    )


if __name__ == "__main__":
    main()
