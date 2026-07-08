"""SandboxSession: persistent per-run REPL semantics + hardening behaviors."""

from __future__ import annotations

import asyncio

import pytest

from databricks_deep_research.tools.code_executor import (
    SandboxSession,
    SandboxSessionHolder,
)


@pytest.fixture
async def session() -> SandboxSession:
    s = SandboxSession(wall_timeout_seconds=10.0)
    yield s
    await s.close()


async def test_namespace_persists_across_calls(session: SandboxSession) -> None:
    first = await session.run("x = 41\nresult = x", {})
    assert first.ok and first.result == 41
    second = await session.run("result = x + 1", {})
    assert second.ok and second.result == 42


async def test_shadow_and_described_track_bindings(session: SandboxSession) -> None:
    res = await session.run(
        "import datetime\nnums = [1, 2, 3]\nstamp = datetime.date(2026, 1, 1)\nresult = None",
        {},
    )
    assert res.ok
    assert session.shadow().get("nums") == [1, 2, 3]
    # date objects are not JSON-able -> described, not shadowed
    assert "stamp" in session.described()
    assert {"nums", "stamp"} <= session.known_names()


async def test_inject_and_read_back(session: SandboxSession) -> None:
    assert await session.inject("seed", 7) is True
    res = await session.run("result = seed * 6", {})
    assert res.ok and res.result == 42
    # non-JSON values are refused parent-side
    assert await session.inject("bad", object()) is False


async def test_bind_result_names_extra_binding(session: SandboxSession) -> None:
    res = await session.run("result = [1.5, 2.5]", {}, bind_result="series")
    assert res.ok
    follow = await session.run("result = series[1]", {})
    assert follow.ok and follow.result == 2.5


async def test_timeout_kills_and_respawns_with_rehydration() -> None:
    session = SandboxSession(wall_timeout_seconds=10.0)
    try:
        await session.run("x = 1\nresult = None", {})
        hang = await session.run("while True:\n    pass", {}, timeout=1.0)
        assert hang.ok is False and hang.timed_out is True
        assert "restarted" in hang.error
        # Next call transparently respawns; JSON shadow (x) was restored.
        after = await session.run("result = x", {})
        assert after.ok and after.result == 1
        assert "restarted" in after.note
    finally:
        await session.close()


async def test_child_exits_on_stdin_eof() -> None:
    session = SandboxSession()
    try:
        assert (await session.run("result = 1", {})).ok
        proc = session._proc
        assert proc is not None
        assert proc.stdin is not None
        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        assert proc.returncode == 0
    finally:
        await session.close()


async def test_args_size_cap_and_non_json_args(session: SandboxSession) -> None:
    huge = {"blob": "x" * (2 * 1024 * 1024)}
    res = await session.run("result = 1", huge)
    assert res.ok is False and "byte cap" in res.error
    res2 = await session.run("result = 1", {"bad": object()})
    assert res2.ok is False and "JSON-serialisable" in res2.error


async def test_ensure_policy_extends_module_allowlist(session: SandboxSession) -> None:
    denied = await session.run("import numpy as np\nresult = 1", {})
    assert denied.ok is False and "not allowed" in denied.error
    await session.ensure_policy(["numpy"], "facade")
    allowed = await session.run("import numpy as np\nresult = int(np.ones(3).sum())", {})
    assert allowed.ok and allowed.result == 3


async def test_security_escapes_blocked(session: SandboxSession) -> None:
    os_import = await session.run("import os\nresult = 1", {})
    assert os_import.ok is False
    subprocess_import = await session.run("import subprocess\nresult = 1", {})
    assert subprocess_import.ok is False
    dunder_string = await session.run("result = '{0.__class__}'.format(1)", {})
    assert dunder_string.ok is False
    stdlib_module_reach = await session.run(
        "import statistics\nresult = statistics.sys", {}
    )
    assert stdlib_module_reach.ok is False  # facade drops re-exported modules


async def test_lock_serializes_concurrent_calls(session: SandboxSession) -> None:
    await session.run("counter = 0\nresult = None", {})
    code = "counter = counter + 1\nresult = counter"

    async def bump(times: int) -> None:
        for _ in range(times):
            res = await session.run(code, {})
            assert res.ok

    await asyncio.gather(bump(25), bump(25))
    final = await session.run("result = counter", {})
    assert final.result == 50


async def test_restore_shadow_seeds_first_spawn() -> None:
    session = SandboxSession()
    try:
        session.restore_shadow({"warm": 123})
        res = await session.run("result = warm", {})
        assert res.ok and res.result == 123
        assert "restarted" in res.note  # rehydration reported
    finally:
        await session.close()


async def test_close_is_idempotent_and_final() -> None:
    session = SandboxSession()
    assert (await session.run("result = 1", {})).ok
    await session.close()
    await session.close()
    res = await session.run("result = 1", {})
    assert res.ok is False and "closed" in res.error


async def test_holder_scopes_one_session() -> None:
    holder = SandboxSessionHolder()
    a = holder.get_or_create()
    b = holder.get_or_create()
    assert a is b
    assert holder.peek() is a
    await holder.aclose()
    assert holder.peek() is None
