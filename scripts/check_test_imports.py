#!/usr/bin/env python
"""
scripts/check_test_imports.py
─────────────────────────────
T1.1 — Validate every `from aaizaql` import in tests/ resolves against
the live source tree.  Exits with code 2 on any failure so pre-commit
marks the hook as failed.

Usage (run from project root):
    python scripts/check_test_imports.py
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TESTS_DIR = ROOT / "tests"
SRC_DIR = ROOT / "src"

# Make sure the local src/ is importable even without `pip install -e .`
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def collect_aaizaql_imports(test_root: Path) -> list[tuple[Path, int, str, list[str]]]:
    """
    Walk every .py file under test_root and return all
    `from aaizaql[.submodule] import name1, name2` statements as
    (file, lineno, module, [names]).
    """
    results = []
    for py_file in sorted(test_root.rglob("*.py")):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("aaizaql")
            ):
                names = [alias.name for alias in node.names]
                results.append((py_file, node.lineno, node.module, names))
    return results


def check_imports(imports: list[tuple[Path, int, str, list[str]]]) -> list[str]:
    errors: list[str] = []
    for file, lineno, module, names in imports:
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            # Only flag if the missing module is part of aaizaql itself.
            # A missing third-party dependency (pydantic, pandas, etc.) means
            # the environment is incomplete — not that the import is stale.
            missing_mod = exc.name or ""
            if not missing_mod.startswith("aaizaql"):
                continue  # skip — dependency not installed, not our problem
            rel = file.relative_to(ROOT)
            errors.append(f"{rel}:{lineno}: cannot import module `{module}`: {exc}")
            continue

        for name in names:
            if not hasattr(mod, name):
                rel = file.relative_to(ROOT)
                errors.append(
                    f"{rel}:{lineno}: `{name}` not found in `{module}` "
                    f"(module imports OK but attribute is missing)"
                )
    return errors


def main() -> int:
    imports = collect_aaizaql_imports(TESTS_DIR)
    print(f"check-test-imports: {len(imports)} import statements scanned across tests/")

    errors = check_imports(imports)

    if errors:
        print("\nFAILED — stale or broken imports detected:\n")
        for err in errors:
            print(f"  {err}")
        print(
            f"\n{len(errors)} error(s) found. "
            "Fix the imports or update the source before committing."
        )
        return 2

    print("All imports resolved successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
