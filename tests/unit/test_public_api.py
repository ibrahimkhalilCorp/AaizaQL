"""
tests/unit/test_public_api.py
──────────────────────────────
T1.1 — Guard the public API surface.

Every name listed in aaizaql.__all__ must be importable directly from
the top-level `aaizaql` package. If a name is removed from source but
left in __all__, or removed from __all__ but still expected by callers,
this test will catch it before it reaches PyPI.

No API keys, no database, no network required.
"""

from __future__ import annotations

import importlib

import aaizaql

# ── Symbols every aaizaql user can rely on ────────────────────────────────────

# Core engine + result
EXPECTED_CORE = [
    "QueryEngine",
    "QueryResult",
    "make_settings",
]

# Exceptions that callers are expected to catch
EXPECTED_EXCEPTIONS = [
    "AAIZAQLError",
    "SecurityException",
    "SQLGenerationError",
    "UnsupportedQueryError",
    "MaxRetriesExceeded",
    "DatabaseError",
    "ConnectorNotFound",
    "LLMError",
    "LLMTimeoutError",
]

# Package metadata
EXPECTED_METADATA = [
    "__version__",
]

ALL_EXPECTED = EXPECTED_CORE + EXPECTED_EXCEPTIONS + EXPECTED_METADATA


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPublicAPIImportable:
    """Every name in ALL_EXPECTED must be importable from the top-level package."""

    def test_all_expected_names_exist_on_package(self) -> None:
        missing = [name for name in ALL_EXPECTED if not hasattr(aaizaql, name)]
        assert not missing, (
            "The following public names are missing from `aaizaql`:\n"  # fix: was f"..."
            + "\n".join(f"  - {n}" for n in missing)
            + "\nRestore them in src/aaizaql/__init__.py or update this list."
        )

    def test_all_expected_names_in_dunder_all(self) -> None:
        """Every expected name should also appear in __all__ for IDE/type-checker support."""
        assert hasattr(aaizaql, "__all__"), "`aaizaql` package is missing __all__"
        not_in_all = [name for name in ALL_EXPECTED if name not in aaizaql.__all__]
        assert not not_in_all, (
            "The following public names are not listed in `aaizaql.__all__`:\n"  # fix: was f"..."
            + "\n".join(f"  - {n}" for n in not_in_all)
            + "\nAdd them to __all__ in src/aaizaql/__init__.py."
        )

    def test_no_unexpected_private_leakage(self) -> None:
        """Names starting with _ must not appear in __all__."""
        private_in_all = [n for n in aaizaql.__all__ if n.startswith("_") and n != "__version__"]
        assert (
            not private_in_all
        ), "Private names found in `aaizaql.__all__` (remove or rename):\n" + "\n".join(
            f"  - {n}" for n in private_in_all
        )  # fix: was f"..."


class TestPublicAPIDirectImport:
    """Each name can be imported directly with `from aaizaql import <name>`."""

    def test_query_engine_importable(self) -> None:
        from aaizaql import QueryEngine  # noqa: F401

    def test_query_result_importable(self) -> None:
        from aaizaql import QueryResult  # noqa: F401

    def test_make_settings_importable(self) -> None:
        from aaizaql import make_settings  # noqa: F401

    def test_version_importable(self) -> None:
        from aaizaql import __version__

        assert isinstance(__version__, str)
        assert __version__.count(".") >= 1, f"__version__ looks wrong: {__version__!r}"

    def test_all_exceptions_importable(self) -> None:
        mod = importlib.import_module("aaizaql")
        for exc_name in EXPECTED_EXCEPTIONS:
            cls = getattr(mod, exc_name, None)
            assert cls is not None, f"`aaizaql.{exc_name}` is not importable"
            assert isinstance(cls, type), f"`aaizaql.{exc_name}` is not a class"
            assert issubclass(cls, Exception), f"`aaizaql.{exc_name}` is not an Exception subclass"


class TestExceptionHierarchy:
    """Verify the exception hierarchy is intact — callers rely on catching base classes."""

    def test_all_exceptions_inherit_from_root(self) -> None:
        from aaizaql import AAIZAQLError

        for exc_name in EXPECTED_EXCEPTIONS:
            if exc_name == "AAIZAQLError":
                continue
            cls = getattr(aaizaql, exc_name)
            assert issubclass(cls, AAIZAQLError), (
                f"`aaizaql.{exc_name}` does not inherit from `AAIZAQLError` — "
                "callers using `except AAIZAQLError` would miss it"
            )

    def test_llm_timeout_is_llm_error(self) -> None:
        """LLMTimeoutError must be a subclass of LLMError for fine-grained catching."""
        from aaizaql import LLMError, LLMTimeoutError

        assert issubclass(LLMTimeoutError, LLMError), (
            "LLMTimeoutError must subclass LLMError — "
            "callers catching LLMError expect timeouts to be included"
        )

    def test_security_exception_is_catchable_as_root(self) -> None:
        from aaizaql import AAIZAQLError, SecurityException

        assert issubclass(SecurityException, AAIZAQLError)
