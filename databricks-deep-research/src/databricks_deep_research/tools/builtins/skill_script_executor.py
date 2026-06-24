"""Out-of-process sandbox for executing user-authored skill scripts.

:class:`ProcessSandbox` is the OS-level security boundary for the skill-script
feature (see :mod:`_skill_script_runner` for the in-subprocess policy and the
rationale for not reusing the compute threadpool). One ``run`` spawns a fresh,
short-lived ``python -I`` subprocess that:

* has its address space / CPU time / file size / open-fd count bounded by
  ``setrlimit`` (POSIX), installed in a ``preexec_fn`` that also calls
  ``os.setsid`` so the whole process group can be killed atomically;
* runs with a **scrubbed environment** — every secret-bearing variable
  (``*TOKEN*``/``*SECRET*``/``DATABRICKS_*``/``PG*``/cloud creds) is removed, so a
  script cannot exfiltrate credentials even if a policy gap were found;
* is **SIGKILL**-ed (whole group) on the wall-clock timeout — there is no orphaned
  thread, unlike the in-process threadpool path.

The result envelope is read back from the subprocess's stdout as JSON. The parent
ALSO runs the AST policy first (fail-fast) so obviously-malicious code never
spawns a process.

Rlimits and ``preexec_fn`` are POSIX-only; on a non-POSIX host the rlimits and
process-group kill are skipped and the wall-clock timeout + env scrub + AST policy
still apply (Databricks Apps run on Linux, where the full boundary is active).
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
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from databricks_deep_research.tools.builtins._skill_script_runner import (
    SkillScriptPolicyError,
    validate_script_source,
)

logger = logging.getLogger(__name__)

__all__ = ["ProcessSandbox", "SandboxResult", "sanitize_subprocess_env"]

_IS_POSIX = os.name == "posix"
_IS_LINUX = sys.platform.startswith("linux")

_RUNNER_PATH = Path(__file__).with_name("_skill_script_runner.py")

# Environment-variable name fragments that mark a secret-bearing variable. Match
# is case-insensitive and substring-based, so ``DATABRICKS_TOKEN``,
# ``PGPASSWORD``, ``AWS_SECRET_ACCESS_KEY``, ``OPENAI_API_KEY`` etc. are all
# dropped. PG* is matched by prefix (``PGHOST``/``PGUSER``/...).
_SENSITIVE_FRAGMENTS: tuple[str, ...] = (
    "token", "secret", "password", "passwd", "api_key", "apikey",
    "credential", "private", "session", "databricks", "client_secret",
)
_SENSITIVE_PREFIXES: tuple[str, ...] = ("PG", "AWS_", "AZURE_", "GCP_", "GOOGLE_")
# Variables that are safe (and useful) to forward so the subprocess behaves
# predictably (encoding, interpreter location).
_FORWARD_ALWAYS: tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "HOME")


def _is_sensitive_env_key(key: str) -> bool:
    """Return True if *key* names a secret-bearing variable to scrub."""
    upper = key.upper()
    if any(upper.startswith(prefix) for prefix in _SENSITIVE_PREFIXES):
        return True
    lower = key.lower()
    return any(fragment in lower for fragment in _SENSITIVE_FRAGMENTS)


def sanitize_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a minimal, secret-free environment for the sandbox subprocess.

    Forwards only an explicit set of innocuous variables (encoding/locale/PATH)
    and drops everything that matches :func:`_is_sensitive_env_key`. Never
    forwards an unknown variable, so newly added secrets are scrubbed by default.
    """
    source = os.environ if base is None else base
    env: dict[str, str] = {}
    for key in _FORWARD_ALWAYS:
        value = source.get(key)
        if value is not None and not _is_sensitive_env_key(key):
            env[key] = value
    # Guarantee a PATH even if the host had none (subprocess exec uses an absolute
    # interpreter path, but child tooling may still consult PATH).
    env.setdefault("PATH", "/usr/bin:/bin")
    # Force isolated, deterministic interpreter behaviour.
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class SandboxResult(BaseModel):
    """Outcome of a sandboxed skill-script run."""

    ok: bool
    result: Any = None
    stdout: str = ""
    error: str = ""
    error_type: str = ""
    note: str = ""
    timed_out: bool = False
    duration_seconds: float = Field(default=0.0, ge=0.0)


class ProcessSandbox:
    """Run a skill script in a hardened, short-lived subprocess.

    Parameters
    ----------
    cpu_seconds:
        ``RLIMIT_CPU`` ceiling (POSIX). A busy loop is terminated by ``SIGXCPU``.
    max_memory_bytes:
        ``RLIMIT_AS`` ceiling (Linux only — skipped on macOS where it can break
        the interpreter's own allocations).
    max_output_bytes:
        Cap on captured stdout / serialised result.
    max_file_bytes:
        ``RLIMIT_FSIZE`` ceiling (POSIX) — a backstop; the sandbox exposes no
        ``open``.
    max_open_files:
        ``RLIMIT_NOFILE`` ceiling (POSIX).
    wall_timeout_seconds:
        Hard wall-clock deadline; the process group is ``SIGKILL``-ed on expiry.
    """

    def __init__(
        self,
        *,
        cpu_seconds: int = 5,
        max_memory_bytes: int = 512 * 1024 * 1024,
        max_output_bytes: int = 64 * 1024,
        max_file_bytes: int = 1 * 1024 * 1024,
        max_open_files: int = 64,
        wall_timeout_seconds: float = 10.0,
    ) -> None:
        self._cpu_seconds = cpu_seconds
        self._max_memory_bytes = max_memory_bytes
        self._max_output_bytes = max_output_bytes
        self._max_file_bytes = max_file_bytes
        self._max_open_files = max_open_files
        self._wall_timeout_seconds = wall_timeout_seconds

    # -- preexec (child side, POSIX only) ------------------------------------

    def _preexec(self) -> None:  # pragma: no cover - runs in the forked child
        """Install rlimits and start a new session (POSIX child, pre-exec)."""
        import resource

        # New session/group so the whole subtree can be killed atomically.
        os.setsid()
        resource.setrlimit(
            resource.RLIMIT_CPU, (self._cpu_seconds, self._cpu_seconds + 1)
        )
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (self._max_file_bytes, self._max_file_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (self._max_open_files, self._max_open_files)
        )
        # RLIMIT_AS is enforced on Linux; on macOS it can abort the interpreter's
        # own mmaps, so it is skipped there (wall-clock timeout still bounds runs).
        if _IS_LINUX:
            resource.setrlimit(
                resource.RLIMIT_AS, (self._max_memory_bytes, self._max_memory_bytes)
            )

    # -- run -----------------------------------------------------------------

    async def run(self, code: str, args: dict[str, Any] | None = None) -> SandboxResult:
        """Validate, then execute *code* in a sandboxed subprocess.

        Never raises for script-level failures — every error (policy rejection,
        runtime exception, timeout, decode failure) is reported as a
        :class:`SandboxResult` with ``ok=False``.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()

        # Fail-fast: reject obviously-unsafe code before spawning a process.
        try:
            validate_script_source(code)
        except SkillScriptPolicyError as exc:
            logger.info("SKILL_SCRIPT_REJECTED reason=%s", str(exc)[:200])
            return SandboxResult(
                ok=False, error=str(exc), error_type="SkillScriptPolicyError"
            )

        payload = {
            "code": code,
            "args": args or {},
            "max_output_bytes": self._max_output_bytes,
        }
        cmd = [sys.executable, "-I", "-B", str(_RUNNER_PATH)]
        env = sanitize_subprocess_env()

        with tempfile.TemporaryDirectory(prefix="skill_script_") as workdir:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=workdir,
                    preexec_fn=self._preexec if _IS_POSIX else None,  # noqa: PLW1509
                )
            except OSError as exc:
                logger.warning("SKILL_SCRIPT_SPAWN_FAILED err=%s", exc)
                return SandboxResult(
                    ok=False, error=f"failed to start sandbox: {exc}", error_type="OSError"
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(json.dumps(payload).encode("utf-8")),
                    timeout=self._wall_timeout_seconds,
                )
            except TimeoutError:
                await self._kill(proc)
                duration = loop.time() - started
                logger.warning(
                    "SKILL_SCRIPT_TIMEOUT timeout=%.1fs killed", self._wall_timeout_seconds
                )
                return SandboxResult(
                    ok=False,
                    error=f"skill script exceeded the {self._wall_timeout_seconds}s time limit",
                    error_type="TimeoutError",
                    timed_out=True,
                    duration_seconds=duration,
                )

        duration = loop.time() - started
        return self._parse_envelope(stdout, stderr, proc.returncode, duration)

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the subprocess (whole group on POSIX) and reap it."""
        try:
            if _IS_POSIX:
                # The child is its own session leader (os.setsid in preexec), so
                # its pgid == pid; kill the group to catch any descendants.
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - non-POSIX fallback
                proc.kill()
        except ProcessLookupError:  # already exited
            pass
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("SKILL_SCRIPT_KILL_FAILED err=%s", exc)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=2.0)

    def _parse_envelope(
        self,
        stdout: bytes,
        stderr: bytes,
        returncode: int | None,
        duration: float,
    ) -> SandboxResult:
        """Decode the subprocess's JSON result envelope (defensively)."""
        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            detail = stderr.decode("utf-8", errors="replace").strip()[-500:]
            # An empty stdout with a non-zero/negative return code means the
            # interpreter was killed by a signal (e.g. SIGXCPU/SIGKILL/OOM).
            killed = returncode is not None and returncode < 0
            return SandboxResult(
                ok=False,
                error=detail or f"sandbox produced no output (exit={returncode})",
                error_type="SandboxError",
                timed_out=killed,
                duration_seconds=duration,
            )
        try:
            envelope = json.loads(text)
        except ValueError:
            return SandboxResult(
                ok=False,
                error=f"could not decode sandbox output: {text[:500]}",
                error_type="SandboxError",
                duration_seconds=duration,
            )
        return SandboxResult(
            ok=bool(envelope.get("ok", False)),
            result=envelope.get("result"),
            stdout=str(envelope.get("stdout", "")),
            error=str(envelope.get("error", "")),
            error_type=str(envelope.get("error_type", "")),
            note=str(envelope.get("note", "")),
            duration_seconds=duration,
        )
