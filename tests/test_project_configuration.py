from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10向け
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_configuration() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def _load_lock() -> dict[str, object]:
    with (PROJECT_ROOT / "uv.lock").open("rb") as file:
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
    assert configuration["dependency-groups"] == {
        "dev": ["pytest", "tomli; python_version < '3.11'"]
    }


def test_uv_uses_cuda_index_on_supported_desktop_platforms() -> None:
    configuration = _load_configuration()
    cuda_platform_marker = "sys_platform == 'linux' or sys_platform == 'win32'"

    assert configuration["tool"]["uv"]["package"] is False
    assert configuration["tool"]["uv"]["sources"] == {
        "torch": [
            {"index": "pytorch-cu130", "marker": cuda_platform_marker}
        ],
        "torchaudio": [
            {"index": "pytorch-cu130", "marker": cuda_platform_marker}
        ],
    }
    assert configuration["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cu130",
            "url": "https://download.pytorch.org/whl/cu130",
            "explicit": True,
        }
    ]


def test_lock_contains_windows_cuda_wheels_for_supported_python_versions() -> None:
    packages = _load_lock()["package"]
    expected_versions = {
        "torch": "2.13.0+cu130",
        "torchaudio": "2.11.0+cu130",
    }

    for package_name, package_version in expected_versions.items():
        package = next(
            item
            for item in packages
            if item["name"] == package_name and item["version"] == package_version
        )
        wheel_urls = [wheel["url"] for wheel in package["wheels"]]
        for python_tag in ("cp310", "cp311", "cp312", "cp313", "cp314"):
            assert any(
                f"-{python_tag}-{python_tag}-win_amd64.whl" in url
                for url in wheel_urls
            )


def test_uv_files_and_colab_setup_are_the_dependency_source_of_truth() -> None:
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8") == "3.12\n"
    assert not (PROJECT_ROOT / "requirements.txt").exists()

    notebook = json.loads(
        (PROJECT_ROOT / "Colab_Inference.ipynb").read_text(encoding="utf-8")
    )
    assert all(isinstance(cell["source"], list) for cell in notebook["cells"])
    setup_header = next(
        cell
        for cell in notebook["cells"]
        if cell["metadata"].get("id") == "setup-header"
    )
    setup_header_source = "".join(setup_header["source"])
    assert "restart" in setup_header_source
    assert "run the same cell once more" in setup_header_source

    setup_cell = next(
        cell for cell in notebook["cells"] if cell["metadata"].get("id") == "setup-code"
    )
    setup_source = "".join(setup_cell["source"])
    assert '"--format", "pylock.toml"' in setup_source
    assert 'PYLOCK_PATH = Path("/content/pylock.' in setup_source
    assert '"--output-file", str(PYLOCK_PATH)' in setup_source
    assert '"uv", "pip", "install", "--system"' in setup_source
    assert "--require-hashes" in setup_source
    assert setup_source.count('"--preview-features", "pylock"') == 2
    assert '"-r", str(PYLOCK_PATH)' in setup_source
    assert "--extra-index-url" not in setup_source
    assert "--index-strategy" not in setup_source

    assert "subprocess.run" in setup_source
    assert "check=True" in setup_source
    assert ".iaamt-colab-setup.json" in setup_source
    assert "do_shutdown(restart=True)" in setup_source
    assert "uuid.uuid4().hex" in setup_source
    assert '"install_kernel_token": IAAMT_KERNEL_TOKEN' in setup_source
    assert 'importlib.metadata.version("numpy")' in setup_source
    assert "IAAMT_SETUP_READY = True" in setup_source
    assert "import scipy.signal" in setup_source

    helper_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["metadata"].get("id") == "stem-sep-helpers"
    )
    helper_source = "".join(helper_cell["source"])
    assert "IAAMT_SETUP_READY" in helper_source
    assert "Run the setup cell again" in helper_source


def test_pytest_imports_project_modules_from_uv_environment() -> None:
    configuration = _load_configuration()

    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]
