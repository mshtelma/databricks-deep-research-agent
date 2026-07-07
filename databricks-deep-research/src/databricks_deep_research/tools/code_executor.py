"""Code-execution backends for deterministic ``python_function`` tools.

Two backends behind one ``CodeExecutor`` seam:

* :class:`SandboxSession` — the DEFAULT. One hardened subprocess per workflow
  run (the run's MemEx scratchpad): the exec namespace lives IN the child and
  persists across calls, so live objects (arrays, frames) survive between
  function invocations with no per-call marshaling. The parent keeps a
  JSON-able *shadow* (fed by per-exec deltas) used for prompt rendering,
  checkpointing, and crash rehydration. Security profile mirrors
  :class:`~databricks_deep_research.tools.builtins.skill_script_executor.ProcessSandbox`:
  ``python -I -B``, scrubbed env (no OBO/``DATABRICKS_*``/cloud creds),
  rlimits (CPU total / AS / FSIZE / NOFILE / **NPROC**), ``setsid`` + group
  ``SIGKILL`` on per-call wall timeout — after which the session is respawned
  and the JSON shadow re-injected (live objects are declared lost).

* :class:`RestrictedCodeExecutor` — opt-in for TRUSTED configurations only
  (host gate required): wraps a private in-process
  :class:`~databricks_deep_research.tools.builtins.compute.PythonComputeTool`
  engine. Low latency and live-object friendly, but per the compute security
  review it is NOT a hard boundary (un-killable worker threads, no memory
  cap), so untrusted/stored code must not select it.

A session serves exactly ONE workflow run (one user) and is never pooled or
reused across runs; ``isolate`` subworkflows get their own session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import tempfile
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from databricks_deep_research.tools.builtins._skill_script_runner import (
    ALLOWED_MODULES,
    DATA_LIBS,
    SkillScriptPolicyError,
    validate_script_source,
)
from databricks_deep_research.tools.builtins.skill_script_executor import (
    SandboxResult,
    sanitize_subprocess_env,
)

logger = logging.getLogger(__name__)

# Public re-exports: hosts (Designer validation, admin tooling) need the code
# policy surface without importing the private runner module directly.
__all__ = [
    "ALLOWED_MODULES",
    "DATA_LIBS",
    "CodeExecutor",
    "RestrictedCodeExecutor",
    "SandboxSession",
    "SandboxSessionHolder",
    "SkillScriptPolicyError",
    "validate_script_source",
]

_IS_POSIX = os.name == "posix"
_IS_LINUX = sys.platform.startswith("linux")

_RUNNER_NAME = "_skill_script_runner.py"

# Caps concurrent live sessions process-wide (bounds aggregate memory/PIDs to
# roughly N * max_memory_bytes). Sessions past the cap wait at spawn.
_DEFAULT_MAX_SESSIONS = 8
_SESSION_SEMAPHORE = asyncio.Semaphore(
    int(os.environ.get("DDR_SANDBOX_MAX_SESSIONS", str(_DEFAULT_MAX_SESSIONS)))
)

# Fork-bomb containment. RLIMIT_NPROC counts the uid's TASKS (threads incl.),
# so this must stay comfortably above the container's normal task count while
# still stopping exponential fork growth; override via env for tight hosts.
_DEFAULT_RLIMIT_NPROC = int(os.environ.get("DDR_SANDBOX_RLIMIT_NPROC", "2048"))

# Parent-side cap on a single call's JSON-encoded args payload.
_MAX_ARGS_BYTES = 1 * 1024 * 1024

_INIT_TIMEOUT_SECONDS = 15.0


@runtime_checkable
class CodeExecutor(Protocol):
    """Executes a fixed code snippet with JSON-able args; never raises for
    script-level failures (they land in the returned :class:`SandboxResult`)."""

    async def run(
        self, code: str, args: dict[str, Any], *, bind_result: str | None = None
    ) -> SandboxResult: ...


class SandboxSession:
    """Per-run persistent sandbox REPL (see module docstring)."""

    def __init__(
        self,
        *,
        wall_timeout_seconds: float = 10.0,
        cpu_seconds_total: int = 300,
        max_memory_bytes: int = 1024 * 1024 * 1024,
        max_output_bytes: int = 64 * 1024,
        max_file_bytes: int = 1 * 1024 * 1024,
        max_open_files: int = 64,
        extra_allowed_modules: Sequence[str] = (),
        data_lib_mode: str = "facade",
    ) -> None:
        self._wall_timeout_seconds = wall_timeout_seconds
        self._cpu_seconds_total = cpu_seconds_total
        self._max_memory_bytes = max_memory_bytes
        self._max_output_bytes = max_output_bytes
        self._max_file_bytes = max_file_bytes
        self._max_open_files = max_open_files
        self._modules: set[str] = {m for m in extra_allowed_modules if m}
        self._data_lib_mode = data_lib_mode
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._shadow: dict[str, Any] = {}
        self._described: dict[str, str] = {}
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._holds_semaphore = False
        self._closed = False
        self._rehydrated_note = ""

    # -- public surface ------------------------------------------------------

    async def run(
        self,
        code: str,
        args: dict[str, Any],
        *,
        bind_result: str | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        """Execute *code* in the persistent session namespace."""
        wall_timeout = timeout or self._wall_timeout_seconds
        try:
            encoded_args = json.dumps(args or {})
        except (TypeError, ValueError) as exc:
            return SandboxResult(
                ok=False,
                error=f"python_function args must be JSON-serialisable: {exc}",
                error_type="TypeError",
            )
        if len(encoded_args) > _MAX_ARGS_BYTES:
            return SandboxResult(
                ok=False,
                error=(
                    f"python_function args exceed the {_MAX_ARGS_BYTES} byte cap "
                    f"({len(encoded_args)} bytes); pass large data by namespace "
                    "variable or table reference instead"
                ),
                error_type="ValueError",
            )

        async with self._lock:
            if self._closed:
                return SandboxResult(
                    ok=False, error="sandbox session is closed", error_type="RuntimeError"
                )
            spawn_error = await self._ensure_proc_locked()
            if spawn_error is not None:
                return spawn_error
            reply = await self._request_locked(
                {"op": "exec", "code": code, "args": args or {}, "bind_result": bind_result},
                timeout=wall_timeout,
            )
            if reply is None:
                return SandboxResult(
                    ok=False,
                    error=(
                        f"python_function exceeded the {wall_timeout}s "
                        "time limit; the sandbox session was killed and will be "
                        "restarted (live objects lost, JSON state restored)"
                    ),
                    error_type="TimeoutError",
                    timed_out=True,
                )
            self._merge_shadow(reply)
            note = str(reply.get("note", ""))
            if self._rehydrated_note:
                note = f"{self._rehydrated_note}; {note}" if note else self._rehydrated_note
                self._rehydrated_note = ""
            return SandboxResult(
                ok=bool(reply.get("ok", False)),
                result=reply.get("result"),
                stdout=str(reply.get("stdout", "")),
                error=str(reply.get("error", "")),
                error_type=str(reply.get("error_type", "")),
                note=note,
            )

    async def ensure_policy(
        self, extra_allowed_modules: Sequence[str], data_lib_mode: str
    ) -> None:
        """Monotonically extend the session's module allowlist / mode."""
        requested = {m for m in extra_allowed_modules if m}
        upgrade_mode = data_lib_mode == "live" and self._data_lib_mode != "live"
        if requested.issubset(self._modules) and not upgrade_mode:
            return
        async with self._lock:
            self._modules |= requested
            if upgrade_mode:
                self._data_lib_mode = "live"
            if self._proc is not None and self._proc.returncode is None:
                await self._request_locked(self._init_payload(), timeout=_INIT_TIMEOUT_SECONDS)

    async def inject(self, name: str, value: Any) -> bool:
        """Bind a JSON-able value into the session namespace (and the shadow)."""
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return False
        async with self._lock:
            if self._closed:
                return False
            spawn_error = await self._ensure_proc_locked()
            if spawn_error is not None:
                return False
            reply = await self._request_locked(
                {"op": "inject", "name": name, "value": value},
                timeout=self._wall_timeout_seconds,
            )
            if reply is None or not reply.get("ok"):
                return False
            self._shadow[name] = value
            return True

    def known_names(self) -> set[str]:
        """Names currently bound in the session (JSON-able or described)."""
        return set(self._shadow) | set(self._described)

    def shadow(self) -> dict[str, Any]:
        """Parent-side JSON-able snapshot (checkpoint / prompt / rehydrate)."""
        return dict(self._shadow)

    def described(self) -> dict[str, str]:
        """Type descriptors for live-only (non-JSON-able) session bindings."""
        return dict(self._described)

    def restore_shadow(self, snapshot: dict[str, Any]) -> None:
        """Seed the shadow from a checkpoint; values reach the child lazily on
        the next spawn (rehydration)."""
        for key, value in snapshot.items():
            if isinstance(key, str) and key.isidentifier():
                self._shadow[key] = value

    async def close(self) -> None:
        """Shut the child down and release resources. Idempotent."""
        async with self._lock:
            self._closed = True
            proc = self._proc
            self._proc = None
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(Exception):
                    proc.stdin.write((json.dumps({"op": "shutdown"}) + "\n").encode())  # type: ignore[union-attr]
                    await asyncio.wait_for(proc.stdin.drain(), timeout=1.0)  # type: ignore[union-attr]
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
            if proc is not None and proc.returncode is None:
                await self._kill(proc)
            if self._holds_semaphore:
                _SESSION_SEMAPHORE.release()
                self._holds_semaphore = False
            if self._workdir is not None:
                with contextlib.suppress(Exception):
                    self._workdir.cleanup()
                self._workdir = None

    # -- internals (caller holds self._lock) -----------------------------------

    def _init_payload(self) -> dict[str, Any]:
        return {
            "op": "init",
            "extra_allowed_modules": sorted(self._modules),
            "data_lib_mode": self._data_lib_mode,
            "max_output_bytes": self._max_output_bytes,
        }

    def _preexec(self) -> None:  # pragma: no cover - runs in the forked child
        import resource

        os.setsid()
        resource.setrlimit(
            resource.RLIMIT_CPU, (self._cpu_seconds_total, self._cpu_seconds_total + 1)
        )
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (self._max_file_bytes, self._max_file_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (self._max_open_files, self._max_open_files)
        )
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_NPROC, (_DEFAULT_RLIMIT_NPROC, _DEFAULT_RLIMIT_NPROC)
            )
        if _IS_LINUX:
            resource.setrlimit(
                resource.RLIMIT_AS, (self._max_memory_bytes, self._max_memory_bytes)
            )

    async def _ensure_proc_locked(self) -> SandboxResult | None:
        if self._proc is not None and self._proc.returncode is None:
            return None
        if not self._holds_semaphore:
            await _SESSION_SEMAPHORE.acquire()
            self._holds_semaphore = True
        if self._workdir is None:
            self._workdir = tempfile.TemporaryDirectory(prefix="python_fn_session_")
        runner = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "builtins", _RUNNER_NAME
        )
        try:
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-B",
                runner,
                "--session",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=sanitize_subprocess_env(),
                cwd=self._workdir.name,
                preexec_fn=self._preexec if _IS_POSIX else None,  # noqa: PLW1509
            )
        except OSError as exc:
            logger.warning("SANDBOX_SESSION_SPAWN_FAILED err=%s", exc)
            return SandboxResult(
                ok=False,
                error=f"failed to start sandbox session: {exc}",
                error_type="OSError",
            )
        init_reply = await self._request_locked(
            self._init_payload(), timeout=_INIT_TIMEOUT_SECONDS
        )
        if init_reply is None or not init_reply.get("ok"):
            return SandboxResult(
                ok=False,
                error="sandbox session failed to initialize",
                error_type="RuntimeError",
            )
        if self._shadow:
            restored = 0
            for name, value in list(self._shadow.items()):
                reply = await self._request_locked(
                    {"op": "inject", "name": name, "value": value},
                    timeout=self._wall_timeout_seconds,
                )
                if reply is not None and reply.get("ok"):
                    restored += 1
            lost = sorted(self._described)
            self._described = {}
            self._rehydrated_note = (
                f"sandbox session restarted: {restored} JSON value(s) restored"
                + (f"; live objects lost: {', '.join(lost)}" if lost else "")
            )
            logger.info(
                "SANDBOX_SESSION_REHYDRATED restored=%d lost=%d", restored, len(lost)
            )
        return None

    async def _request_locked(
        self, payload: dict[str, Any], *, timeout: float
    ) -> dict[str, Any] | None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            return None
        try:
            proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            if not line:
                raise ConnectionResetError("session closed its stdout")
            reply = json.loads(line.decode("utf-8"))
            if not isinstance(reply, dict):
                raise ValueError("session protocol violation: non-object reply")
            return reply
        except (TimeoutError, ValueError, OSError, ConnectionResetError) as exc:
            logger.warning(
                "SANDBOX_SESSION_REQUEST_FAILED op=%s err=%s — killing session",
                payload.get("op"),
                exc,
            )
            await self._kill(proc)
            self._proc = None
            return None

    def _merge_shadow(self, reply: dict[str, Any]) -> None:
        delta = reply.get("shadow_delta")
        if isinstance(delta, dict):
            for key, value in delta.items():
                if isinstance(key, str):
                    self._shadow[key] = value
        described = reply.get("described")
        if isinstance(described, dict):
            self._described = {
                str(k): str(v) for k, v in described.items()
            }
            for key in described:
                self._shadow.pop(str(key), None)

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        try:
            if _IS_POSIX:
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - non-POSIX fallback
                proc.kill()
        except ProcessLookupError:
            pass
        except OSError:  # pragma: no cover - defensive
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=2.0)


class SandboxSessionHolder:
    """Run-scoped holder installed in the tool-factory extras.

    Lives at ``extras["_sandbox_session"]``; the isolate-subworkflow resolver
    builder EXCLUDES this key (like ``_resolver_cache``) so isolated children
    get their own session. The executor closes the holder at run end.
    """

    def __init__(self) -> None:
        self._session: SandboxSession | None = None

    def get_or_create(
        self,
        *,
        wall_timeout_seconds: float = 10.0,
        extra_allowed_modules: Sequence[str] = (),
        data_lib_mode: str = "facade",
    ) -> SandboxSession:
        if self._session is None:
            self._session = SandboxSession(
                wall_timeout_seconds=wall_timeout_seconds,
                extra_allowed_modules=extra_allowed_modules,
                data_lib_mode=data_lib_mode,
            )
        return self._session

    def peek(self) -> SandboxSession | None:
        return self._session

    async def aclose(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()


class RestrictedCodeExecutor:
    """Opt-in trusted backend over a PRIVATE in-process compute engine.

    NOT a hard boundary (un-killable threads, no memory cap) — hosts must gate
    its selection behind an operator-owned switch; the builtin factory enforces
    that fail-closed.
    """

    def __init__(
        self,
        *,
        extra_modules: list[str] | None = None,
        enable_dataframes: bool = False,
        max_execution_seconds: float = 10.0,
        max_output_chars: int = 10_000,
    ) -> None:
        from databricks_deep_research.tools.builtins.compute import PythonComputeTool

        self._engine = PythonComputeTool(
            name="python_function_restricted",
            extra_modules=extra_modules,
            enable_dataframes=enable_dataframes,
            max_execution_seconds=max_execution_seconds,
            max_output_chars=max_output_chars,
        )

    @property
    def engine(self) -> Any:
        """The private compute engine (live-object namespace access)."""
        return self._engine

    async def run(
        self, code: str, args: dict[str, Any], *, bind_result: str | None = None
    ) -> SandboxResult:
        from databricks_deep_research.tools.protocol import ToolContext

        for key, value in (args or {}).items():
            if isinstance(key, str) and key.isidentifier() and not key.startswith("__"):
                self._engine.inject_variable(key, value)
        tool_result = await self._engine.execute({"code": code}, ToolContext())
        result_value = self._engine.get_variable("result")
        if bind_result and isinstance(bind_result, str) and bind_result.isidentifier():
            self._engine.inject_variable(bind_result, result_value)
        content = tool_result.content if isinstance(tool_result.content, str) else str(
            tool_result.content
        )
        return SandboxResult(
            ok=tool_result.success,
            result=result_value,
            stdout=content,
            error=tool_result.error or "",
            error_type="" if tool_result.success else "ComputeError",
        )
