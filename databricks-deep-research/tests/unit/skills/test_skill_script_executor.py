"""Tests for the hardened out-of-process skill-script executor (A2).

Covers the three boundary layers independently: the AST policy
(``validate_script_source``), the in-process restricted exec (``run_script``), and
the OS sandbox (``ProcessSandbox`` — env scrub, timeout SIGKILL with no orphan,
fail-fast AST rejection, runtime errors), plus the ``RunSkillScriptTool`` gating
and a parity check that keeps the self-contained policy in lockstep with
``compute.py``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from databricks_deep_research.tools.builtins import compute as _compute
from databricks_deep_research.tools.builtins._skill_script_runner import (
    ALLOWED_MODULES,
    SkillScriptPolicyError,
    _guarded_import,
    run_script,
    validate_script_source,
)
from databricks_deep_research.tools.builtins.run_skill_script import RunSkillScriptTool
from databricks_deep_research.tools.builtins.skill_script_executor import (
    ProcessSandbox,
    SandboxResult,
    sanitize_subprocess_env,
)
from databricks_deep_research.tools.protocol import ToolContext

# ---------------------------------------------------------------------------
# AST policy
# ---------------------------------------------------------------------------


def test_policy_allows_safe_code_and_allowlisted_imports() -> None:
    validate_script_source("import math\nresult = math.sqrt(16)")
    validate_script_source("from statistics import mean\nresult = mean([1, 2, 3])")
    validate_script_source("result = sum(x for x in range(10))")


@pytest.mark.parametrize(
    "snippet",
    [
        "import os",
        "import socket",
        "import subprocess",
        "import sys",
        "from os import system",
        "from os.path import join",
        "import os.path",
        "import importlib",
        "from . import sibling",
    ],
)
def test_policy_blocks_disallowed_imports(snippet: str) -> None:
    with pytest.raises(SkillScriptPolicyError):
        validate_script_source(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "__import__('os')",
        "eval('1+1')",
        "exec('x=1')",
        "compile('1', '<s>', 'eval')",
        "open('/etc/passwd')",
        "getattr(object, 'x')",
        "globals()",
        "().__class__.__bases__",
        "x = (1).__class__",
        "breakpoint()",
    ],
)
def test_policy_blocks_reflective_and_eval_primitives(snippet: str) -> None:
    with pytest.raises(SkillScriptPolicyError):
        validate_script_source(snippet)


def test_policy_rejects_empty_and_oversize() -> None:
    with pytest.raises(SkillScriptPolicyError):
        validate_script_source("   ")
    with pytest.raises(SkillScriptPolicyError):
        validate_script_source("x = 1\n" * 10_001)


def test_policy_matches_compute_allowlist() -> None:
    """The self-contained policy must mirror compute's vetted module allowlist."""
    assert frozenset(_compute._ALLOWED_IMPORT_MODULES) == ALLOWED_MODULES


def test_policy_dunder_blocklist_is_superset_of_compute() -> None:
    from databricks_deep_research.tools.builtins._skill_script_runner import (
        _BLOCKED_DUNDER_ATTRS,
    )

    assert _compute._BLOCKED_DUNDER_ATTRS <= _BLOCKED_DUNDER_ATTRS


# ---------------------------------------------------------------------------
# Restricted in-process exec (run_script)
# ---------------------------------------------------------------------------


def test_run_script_returns_result_and_args() -> None:
    out = run_script("result = sum(nums)", {"nums": [1, 2, 3]})
    assert out["ok"] is True
    assert out["result"] == 6


def test_run_script_captures_stdout() -> None:
    out = run_script("print('hello')\nresult = 1", {})
    assert out["ok"] is True
    assert "hello" in out["stdout"]
    assert out["result"] == 1


def test_run_script_reports_runtime_error() -> None:
    out = run_script("result = 1 / 0", {})
    assert out["ok"] is False
    assert out["error_type"] == "ZeroDivisionError"


def test_run_script_rejects_policy_violation() -> None:
    out = run_script("import os", {})
    assert out["ok"] is False
    assert out["error_type"] == "SkillScriptPolicyError"


def test_run_script_coerces_unserialisable_result() -> None:
    out = run_script("result = object()", {})
    assert out["ok"] is True
    assert isinstance(out["result"], str)
    assert "note" in out


def test_guarded_import_blocks_non_allowlisted() -> None:
    assert _guarded_import("math") is not None
    with pytest.raises(ImportError):
        _guarded_import("os")
    with pytest.raises(ImportError):
        _guarded_import("math", level=1)


# ---------------------------------------------------------------------------
# Sandbox escape regressions (security review CRIT-1 + MED-1)
# ---------------------------------------------------------------------------


def test_guarded_import_returns_facade_without_live_submodules() -> None:
    # CRIT-1: the imported handle must NOT expose re-exported modules
    # (statistics.sys / calendar.sys / fractions.sys) — the R6 RCE vector.
    stats = _guarded_import("statistics")
    assert not hasattr(stats, "sys")
    assert callable(stats.mean)  # legitimate compute attribute still works


def test_run_script_blocks_statistics_sys_modules_rce() -> None:
    # The canonical escape: statistics.sys.modules['os'].popen('id').
    out = run_script("import statistics\nresult = statistics.sys.modules", {})
    assert out["ok"] is False


def test_run_script_still_supports_real_compute() -> None:
    out = run_script("import statistics\nresult = statistics.mean([2, 4, 6])", {})
    assert out["ok"] is True
    assert out["result"] == 4


@pytest.mark.parametrize(
    "snippet",
    [
        '"{0.__class__}".format([])',
        'fmt = "{0.__class__.__bases__}"',
        'name = "__subclasses__"',
        'x = "().__class__"',
    ],
)
def test_policy_blocks_dunder_in_string_literals(snippet: str) -> None:
    # MED-1: a blocked dunder smuggled inside a string literal (format-string
    # traversal) is rejected by the AST policy.
    with pytest.raises(SkillScriptPolicyError):
        validate_script_source(snippet)


# ---------------------------------------------------------------------------
# Environment scrub
# ---------------------------------------------------------------------------


def test_sanitize_env_drops_secrets() -> None:
    polluted = {
        "DATABRICKS_TOKEN": "secret",
        "DATABRICKS_HOST": "https://x",
        "PGPASSWORD": "p",
        "PGHOST": "h",
        "AWS_SECRET_ACCESS_KEY": "k",
        "OPENAI_API_KEY": "o",
        "MY_SESSION_ID": "s",
        "PATH": "/usr/bin",
        "LANG": "en_US.UTF-8",
    }
    env = sanitize_subprocess_env(polluted)
    for leaked in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_HOST",
        "PGPASSWORD",
        "PGHOST",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
        "MY_SESSION_ID",
    ):
        assert leaked not in env, leaked
    assert env["PATH"] == "/usr/bin"
    assert env["LANG"] == "en_US.UTF-8"


def test_sanitize_env_always_has_path() -> None:
    env = sanitize_subprocess_env({})
    assert env.get("PATH")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


# ---------------------------------------------------------------------------
# ProcessSandbox (out-of-process)
# ---------------------------------------------------------------------------


async def test_sandbox_happy_path() -> None:
    sandbox = ProcessSandbox()
    out = await sandbox.run("result = 2 + 2")
    assert out.ok is True
    assert out.result == 4


async def test_sandbox_passes_args() -> None:
    sandbox = ProcessSandbox()
    out = await sandbox.run("result = a * b", {"a": 6, "b": 7})
    assert out.ok is True
    assert out.result == 42


async def test_sandbox_rejects_policy_without_spawn() -> None:
    sandbox = ProcessSandbox()
    out = await sandbox.run("import os\nos.system('id')")
    assert out.ok is False
    assert out.error_type == "SkillScriptPolicyError"


async def test_sandbox_reports_runtime_error() -> None:
    sandbox = ProcessSandbox()
    out = await sandbox.run("result = 1 / 0")
    assert out.ok is False
    assert out.error_type == "ZeroDivisionError"


async def test_sandbox_kills_on_timeout_no_orphan() -> None:
    """A busy loop must be SIGKILL-ed at the wall-clock deadline (no orphan)."""
    sandbox = ProcessSandbox(cpu_seconds=1, wall_timeout_seconds=1.0)
    out = await sandbox.run("while True:\n    pass")
    assert out.ok is False
    assert out.timed_out is True
    # The run returns only after the process is reaped, so nothing is orphaned.
    assert out.duration_seconds < 5.0


async def test_sandbox_cannot_read_secret_env() -> None:
    """Even if a script could import os, the subprocess env carries no secret."""
    os.environ["SKILL_SANDBOX_LEAK_TOKEN"] = "topsecret"
    try:
        # The script cannot import os (AST blocks it) — proving env is unreachable
        # from script code regardless of what the parent process holds.
        sandbox = ProcessSandbox()
        out = await sandbox.run("import os\nresult = os.environ.get('SKILL_SANDBOX_LEAK_TOKEN')")
        assert out.ok is False
        assert out.error_type == "SkillScriptPolicyError"
        # And the scrub function drops it from any forwarded env.
        assert "SKILL_SANDBOX_LEAK_TOKEN" not in sanitize_subprocess_env()
    finally:
        del os.environ["SKILL_SANDBOX_LEAK_TOKEN"]


# ---------------------------------------------------------------------------
# RunSkillScriptTool
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSkill:
    def __init__(self, name: str, scripts: dict[str, str]) -> None:
        self.name = name
        self.description = "test skill"
        self.scripts = scripts
        self.security_verdict = None


class _FakeStore:
    def __init__(self, skills: dict[str, _FakeSkill]) -> None:
        self._skills = skills

    async def list_skills(self) -> list[Any]:
        return [_FakeMeta(n) for n in self._skills]

    async def get_skill(self, name: str) -> Any:
        return self._skills.get(name)

    async def put_skill(self, skill: Any, *, scan: Any) -> None:  # pragma: no cover
        raise NotImplementedError


class _FakeScan:
    def __init__(self, *, safe: bool, reason: str = "") -> None:
        self.safe = safe
        self.reason = reason


class _FakeScanner:
    def __init__(self, *, safe: bool) -> None:
        self._safe = safe

    async def scan(self, skill: Any) -> Any:
        return _FakeScan(safe=self._safe)


def _store_with_script() -> _FakeStore:
    return _FakeStore(
        {"analysis": _FakeSkill("analysis", {"sum": "result = sum(values)"})}
    )


async def test_tool_disabled_refuses() -> None:
    tool = RunSkillScriptTool(_store_with_script(), enabled=False)
    res = await tool.execute(
        {"skill": "analysis", "script": "sum", "arguments": {"values": [1]}},
        ToolContext(),
    )
    assert res.success is False
    assert res.error == "skill_scripts_disabled"


async def test_tool_runs_script() -> None:
    tool = RunSkillScriptTool(_store_with_script(), enabled=True)
    args = tool.validate_arguments(
        {"skill": "analysis", "script": "sum", "arguments": {"values": [10, 20]}}
    )
    res = await tool.execute(args, ToolContext())
    assert res.success is True
    assert "30" in res.content


async def test_tool_unknown_skill_and_script() -> None:
    tool = RunSkillScriptTool(_store_with_script(), enabled=True)
    miss_skill = await tool.execute(
        {"skill": "nope", "script": "sum", "arguments": {}}, ToolContext()
    )
    assert miss_skill.success is False
    assert miss_skill.error == "skill_not_found"

    miss_script = await tool.execute(
        {"skill": "analysis", "script": "nope", "arguments": {}}, ToolContext()
    )
    assert miss_script.success is False
    assert miss_script.error == "script_not_found"


async def test_tool_fails_closed_on_unsafe_scan() -> None:
    tool = RunSkillScriptTool(
        _store_with_script(), enabled=True, scanner=_FakeScanner(safe=False)
    )
    res = await tool.execute(
        {"skill": "analysis", "script": "sum", "arguments": {"values": [1]}},
        ToolContext(),
    )
    assert res.success is False
    assert res.error == "skill_unsafe"


def test_tool_validate_arguments_requires_names() -> None:
    tool = RunSkillScriptTool(_store_with_script(), enabled=True)
    with pytest.raises(ValueError):
        tool.validate_arguments({"skill": "", "script": "sum"})
    with pytest.raises(ValueError):
        tool.validate_arguments({"skill": "analysis", "script": ""})
    with pytest.raises(ValueError):
        tool.validate_arguments(
            {"skill": "a", "script": "b", "arguments": "notdict"}
        )


def test_sandbox_result_model_defaults() -> None:
    r = SandboxResult(ok=True, result=5)
    assert r.stdout == ""
    assert r.timed_out is False
