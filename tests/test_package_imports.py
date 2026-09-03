from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "instrument_agnostic_amt"
PACKAGE_DIRECTORIES = {PACKAGE_ROOT.name, "tests"}


def _repository_module_names() -> set[str]:
    module_names = {path.stem for path in REPOSITORY_ROOT.glob("*.py")}
    module_names.update(
        path.name
        for path in REPOSITORY_ROOT.iterdir()
        if path.is_dir()
        and path.name.isidentifier()
        and path.name not in PACKAGE_DIRECTORIES
        and any(path.rglob("*.py"))
    )
    return module_names


def test_package_does_not_import_repository_modules() -> None:
    repository_module_names = _repository_module_names()
    violations: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            ):
                imported_names = [node.module]
            else:
                continue

            for imported_name in imported_names:
                root_name = imported_name.partition(".")[0]
                if root_name in repository_module_names:
                    relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
                    violations.append(f"{relative_path}:{node.lineno}: {root_name}")

    assert not violations, (
        "package modules must not import repository-only modules:\n"
        + "\n".join(violations)
    )
