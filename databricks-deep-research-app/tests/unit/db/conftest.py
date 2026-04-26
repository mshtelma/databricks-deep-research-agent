"""Shared fixtures for db/ unit tests."""

import pytest

# Canonical dotted path for the import-time asyncpg-version probe.
# Kept here so callers never repeat the magic string.
_SUPPORTS_NAME_FUNC_PATH = "deep_research.db.asyncpg_config._SUPPORTS_NAME_FUNC"


@pytest.fixture
def force_name_func_supported(monkeypatch):
    """Pin the import-time asyncpg probe to True.

    Use when a test needs to exercise the "asyncpg supports
    prepared_statement_name_func" branch regardless of which asyncpg wheel
    the developer (or CI) happens to have installed. Without this, asyncpg
    wheels that lack PR #846 (v0.29 and some 0.30/0.31 builds) skip the
    branch and the assertions that touch prepared_statement_name_func
    KeyError.

    Relies on asyncpg_config.py reading _SUPPORTS_NAME_FUNC as a module
    global inside the function body. If that ever changes (e.g., the flag
    gets captured in a closure or default arg), this fixture becomes a
    no-op and the "supported" assertions will KeyError again — exactly
    the signal we want.
    """
    monkeypatch.setattr(_SUPPORTS_NAME_FUNC_PATH, True)


@pytest.fixture
def force_name_func_unsupported(monkeypatch):
    """Pin the import-time asyncpg probe to False.

    Use to prove the version-fallback path omits the kwarg so calling
    asyncpg.connect(...) / create_async_engine(...) on older wheels does
    not raise TypeError.
    """
    monkeypatch.setattr(_SUPPORTS_NAME_FUNC_PATH, False)
