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


def _load_colab_notebook() -> dict[str, object]:
    return json.loads(
        (PROJECT_ROOT / "Colab_Inference.ipynb").read_text(encoding="utf-8")
    )


def _get_colab_cell_source(cell_id: str) -> str:
    notebook = _load_colab_notebook()
    cell = next(
        cell
        for cell in notebook["cells"]
        if cell["metadata"].get("id") == cell_id
    )
    return "".join(cell["source"])


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


def test_uv_files_are_the_dependency_source_of_truth() -> None:
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8") == "3.12\n"
    assert not (PROJECT_ROOT / "requirements.txt").exists()

    for readme_name in ("README.md", "README_ja.md"):
        readme = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
        assert "uv sync --locked" in readme
        assert "pip install -r requirements.txt" not in readme


def test_colab_setup_installs_locked_dependency_sources_with_hashes() -> None:
    notebook = _load_colab_notebook()
    setup_source = _get_colab_cell_source("setup-code")

    assert all(isinstance(cell["source"], list) for cell in notebook["cells"])
    assert '"--format", "pylock.toml"' in setup_source
    assert 'PYLOCK_PATH = Path("/content/pylock.' in setup_source
    assert '"--output-file", str(PYLOCK_PATH)' in setup_source
    assert '"uv", "pip", "install", "--system"' in setup_source
    assert "--require-hashes" in setup_source
    assert setup_source.count('"--preview-features", "pylock"') == 2
    assert '"-r", str(PYLOCK_PATH)' in setup_source
    assert "--extra-index-url" not in setup_source
    assert "--index-strategy" not in setup_source


def test_colab_setup_restarts_only_the_installing_kernel() -> None:
    setup_source = _get_colab_cell_source("setup-code")

    assert ".iaamt-colab-setup.json" in setup_source
    assert "uuid.uuid4().hex" in setup_source
    assert '"install_kernel_token": IAAMT_KERNEL_TOKEN' in setup_source
    assert 'setup_state.get("lock_sha256") != lock_sha256' in setup_source
    assert 'setup_state.get("install_kernel_token") == IAAMT_KERNEL_TOKEN' in setup_source
    assert "do_shutdown(restart=True)" in setup_source


def test_colab_continues_from_audio_upload_after_restart() -> None:
    setup_header_source = _get_colab_cell_source("setup-header")
    upload_source = _get_colab_cell_source("upload-code")

    assert "restart" in setup_header_source
    assert "continue directly to step 2" in setup_header_source
    assert "Run all" in setup_header_source
    assert ".iaamt-colab-setup.json" in upload_source
    assert 'setup_state.get("lock_sha256") != lock_sha256' in upload_source
    assert 'importlib.metadata.version("numpy")' in upload_source
    assert "import scipy.signal" in upload_source
    assert "IAAMT_SETUP_READY = True" in upload_source


def test_colab_prominently_warns_that_first_run_all_stops() -> None:
    description_source = _get_colab_cell_source("description")

    assert "⚠️ **IMPORTANT" in description_source
    assert "first `Run all` stops once by design" in description_source
    assert "expected, not a failure" in description_source
    assert "continue from **2. Prepare Audio**" in description_source


def test_colab_setup_reports_commands_and_heartbeat_progress() -> None:
    setup_source = _get_colab_cell_source("setup-code")

    assert "def run_step" in setup_source
    assert "shlex.join(command)" in setup_source
    assert "subprocess.Popen" in setup_source
    assert "subprocess.TimeoutExpired" in setup_source
    assert "time.monotonic()" in setup_source
    assert "still running" in setup_source
    assert "raise subprocess.CalledProcessError" in setup_source
    assert setup_source.count("run_step(") >= 6


def test_colab_upload_restores_the_recorded_clone_path_with_lock_fallback() -> None:
    setup_source = _get_colab_cell_source("setup-code")
    upload_source = _get_colab_cell_source("upload-code")

    assert 'PROJECT_DIR = Path("/content") / Path(REPOSITORY_URL).stem' in setup_source
    assert '"project_dir": str(PROJECT_DIR)' in setup_source
    assert 'setup_state["project_dir"] = str(PROJECT_DIR)' in setup_source
    assert 'if "project_dir" in setup_state' in upload_source
    assert 'Path("/content").glob("*/uv.lock")' in upload_source
    assert '== setup_state.get("lock_sha256")' in upload_source
    assert 'setup_state["project_dir"] = str(PROJECT_DIR)' in upload_source


def test_colab_upload_exposes_an_absolute_audio_path() -> None:
    upload_source = _get_colab_cell_source("upload-code")

    assert "uploaded_name = next(iter(uploaded))" in upload_source
    assert "audio_path = str(Path(uploaded_name).resolve())" in upload_source


def test_colab_helpers_require_the_post_restart_upload_bootstrap() -> None:
    helper_source = _get_colab_cell_source("stem-sep-helpers")

    assert 'globals().get("IAAMT_SETUP_READY", False)' in helper_source
    assert "Run the audio upload cell after Colab reconnects" in helper_source


def test_pytest_imports_project_modules_from_uv_environment() -> None:
    configuration = _load_configuration()

    assert configuration["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]
