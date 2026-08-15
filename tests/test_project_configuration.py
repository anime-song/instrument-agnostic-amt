from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_configuration() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def test_project_pins_supported_pytorch_versions() -> None:
    project = _load_configuration()["project"]

    assert project["requires-python"] == ">=3.10,<3.15"
    assert "torch==2.13.0" in project["dependencies"]
    assert "torchaudio==2.11.0" in project["dependencies"]


def test_project_groups_dependencies_by_workflow() -> None:
    configuration = _load_configuration()

    assert set(configuration["project"]["dependencies"]) == {
        "einops",
        "dlchordx",
        "mido",
        "numpy",
        "pretty-midi",
        "pyyaml",
        "scipy",
        "soundfile",
        "torch==2.13.0",
        "torchaudio==2.11.0",
        "tqdm",
    }
    assert configuration["project"]["optional-dependencies"] == {
        "evaluation": ["mir-eval"],
        "stem": ["chord-romanizer", "librosa", "stem-splitter"],
        "training": [
            "audiomentations",
            "pedalboard",
            "tensorboard",
            "torch-optimizer",
            "wandb",
        ],
    }
    assert configuration["dependency-groups"] == {"dev": ["pytest"]}


def test_uv_uses_cuda_index_only_on_linux() -> None:
    configuration = _load_configuration()

    assert configuration["tool"]["uv"]["package"] is False
    assert configuration["tool"]["uv"]["sources"] == {
        "torch": [
            {"index": "pytorch-cu130", "marker": "sys_platform == 'linux'"}
        ],
        "torchaudio": [
            {"index": "pytorch-cu130", "marker": "sys_platform == 'linux'"}
        ],
    }
    assert configuration["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cu130",
            "url": "https://download.pytorch.org/whl/cu130",
            "explicit": True,
        }
    ]


def test_uv_files_are_the_dependency_source_of_truth() -> None:
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8") == "3.12\n"
    assert not (PROJECT_ROOT / "requirements.txt").exists()


def test_pytest_imports_project_modules_from_uv_environment() -> None:
    configuration = _load_configuration()

    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]
