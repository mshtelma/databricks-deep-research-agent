"""Pin the TS DeployHereErrorKind union to the Python Literal (Section S).

This test reads the TypeScript source file and asserts that the set of error
kind string literals matches the Python ``DeployHereErrorKind`` Literal
*exactly* — neither language may have kinds the other does not.

Expected failure: this test FAILS until the frontend executor lands the
matching TS update adding ``"app_name_collision"``, ``"framework_tag_unreachable"``,
and ``"reachability_timeout"`` to ``frontend/src/types/deployment.ts``.
"""
from __future__ import annotations

import pathlib
import re
import typing

import pytest

from deep_research.schemas.deployment import DeployHereErrorKind

_TS_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "types"
    / "deployment.ts"
)


def _parse_ts_union() -> set[str]:
    text = _TS_FILE.read_text()
    match = re.search(
        r"export type DeployHereErrorKind\s*=\s*((?:\s*\|?\s*'[^']+')+)",
        text,
    )
    assert match is not None, "DeployHereErrorKind union not found in deployment.ts"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def test_deploy_here_error_kind_parity() -> None:
    """The TS union must list exactly the same kinds as the Python Literal."""
    python_kinds = set(typing.get_args(DeployHereErrorKind))
    ts_kinds = _parse_ts_union()
    assert python_kinds == ts_kinds, (
        f"TS/Python DeployHereErrorKind drift:\n"
        f"  only-in-python: {python_kinds - ts_kinds}\n"
        f"  only-in-ts:     {ts_kinds - python_kinds}"
    )
