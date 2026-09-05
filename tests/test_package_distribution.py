from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temporary_root = tmp_path_factory.mktemp("package-distribution")
    output_directory = temporary_root / "dist"
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to verify the project distribution")
    result = subprocess.run(
        [
            uv,
            "build",
            "--offline",
            "--no-python-downloads",
            "--python",
            sys.executable,
            "--wheel",
            "--out-dir",
            str(output_directory),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        result.stdout
        + result.stderr
        + "\nHint: if `uv sync` has completed, hatchling should be available "
        "in uv's cache."
    )
    wheels = tuple(output_directory.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_only_tracked_package_files(
    built_wheel: Path,
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required to verify the wheel contents")
    result = subprocess.run(
        [git, "ls-files", "-z", "--", "instrument_agnostic_amt"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("a Git checkout is required to verify the wheel contents")
    tracked_package_files = set(result.stdout.split("\0")) - {""}

    with zipfile.ZipFile(built_wheel) as wheel:
        members = set(wheel.namelist())

    package_prefix = "instrument_agnostic_amt/"
    wheel_version = built_wheel.name.removeprefix(
        "instrument_agnostic_amt-"
    ).split("-", maxsplit=1)[0]
    distribution_prefix = f"instrument_agnostic_amt-{wheel_version}.dist-info/"
    unexpected_members = {
        member
        for member in members
        if not member.startswith((package_prefix, distribution_prefix))
    }
    assert not unexpected_members
    wheel_package_files = {
        member for member in members if member.startswith(package_prefix)
    }
    assert wheel_package_files == tracked_package_files
