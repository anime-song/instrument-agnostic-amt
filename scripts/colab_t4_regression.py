#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_REPOSITORY_URL = (
    "https://github.com/ntamotsu/fork-instrument-agnostic-amt.git"
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the inference regression suite on a Colab Tesla T4."
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPOSITORY_URL)
    parser.add_argument(
        "--branch",
        required=True,
        help="Remote branch to test (required to avoid testing main by mistake).",
    )
    return parser.parse_args()


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
            str(args.repo_url),
            str(worktree),
        ]
    )
    _run(["git", "rev-parse", "HEAD"], cwd=worktree)
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

    compile_environment = dict(os.environ)
    compile_environment["RUN_ACCELERATOR_COMPILE_TEST"] = "1"
    _run(
        ["uv", "run", "pytest", "-q", "tests/test_cuda_inference.py"],
        cwd=worktree,
        env=compile_environment,
    )


if __name__ == "__main__":
    main()
