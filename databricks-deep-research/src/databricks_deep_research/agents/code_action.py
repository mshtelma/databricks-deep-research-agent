"""MemEx code-action bridge — tools-as-functions inside the compute sandbox.

Spec: ``specs/unified-agent-architecture-plan.md`` §1.4. This is the
**highest-security-scrutiny** module in the roadmap: it lets agent-emitted
Python (running in the AST-guarded :class:`PythonComputeTool` sandbox) call its
*allowlisted* research tools AS FUNCTIONS and ``submit()`` a typed result.

Security model (Codex Top-Fixes 1–3 — see ``.omc/research/code_action_security_design.md``):

* **Allowlist, not blind-wrap (Top-Fix 1).** Only the framework-supplied,
  per-Cell ``code_action_tools`` allowlist gets a closure. The closure set is
  built from the loop's already-bound ``_all_tools`` — never from model-supplied
  names or reflection over arbitrary attributes.
* **One gated spine (Top-Fix 2).** A closure does NOT call ``tool.execute``
  directly. It schedules :meth:`ReactLoop._run_tool_gated` — the SAME coroutine
  the JSON tool-call path uses — onto the running event loop and blocks the
  sandbox worker thread for the result. So HITL, per-tool budget, admission,
  source/pool writes, tracing, and events all fire identically.
* **Capture nothing reachable (Top-Fix 3).** A closure binds only a
  ``weakref`` to the loop plus the tool-name string, and looks the tool up at
  call time. It closes over NO ``ExecutionContext`` / ``WorkspaceClient`` /
  ``user_token`` / auth object. Its return value is the admitted CONTENT,
  coerced to a plain object via the §1.1 :func:`coerce_to_object` helper —
  never the raw ``ToolResult``, context, or registry internals. ``submit``
  records ONLY a JSON-able value (non-serialisable submits raise).

The closures and ``submit`` are injected under reserved names; the compute AST
guard rejects reassignment of those names so sandbox code cannot shadow
``submit`` to exfiltrate (enforced in ``tools/builtins/compute.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from databricks_deep_research.agents.source_aware import tool_source_kind
from databricks_deep_research.agents.tool_offload import coerce_to_object
from databricks_deep_research.events.types import ToolCallEvent, ToolResultEvent
from databricks_deep_research.llm.client import ToolCall

if TYPE_CHECKING:
    from databricks_deep_research.agents.react_loop import ReactLoop

logger = logging.getLogger(__name__)

# Source kinds whose tools may be exposed as code-action callables. ``builtin``
# is excluded on purpose: ``compute`` itself is the host, and other builtins are
# framework internals, not data tools. The allowlist (``code_action_tools``) is
# the primary gate; this set is a second, source-kind-level guard so an
# allowlisted-but-non-data tool is still not bridged.
_CODE_ACTION_ALLOWED_SOURCE_KINDS: frozenset[str] = frozenset(
    {
        "web",
        "web_search",
        "web_crawl",
        "vector_index",
        "vector_search",
        "sql_analytics",
        "genie",
        "qa_assistant",
        "knowledge_assistant",
        "file",
        "uploaded_file",
        "text_table",
    }
)

# Reserved namespace name under which the typed-result sink is injected. Sandbox
# code calls ``submit(value)``; reassigning the name is blocked by the AST guard.
SUBMIT_NAME = "submit"

# Reserved namespace name for the GOVERNED spawn bridge (spec §3.3). Sandbox code
# calls ``spawn_agent(name, prompt)`` to run a DECLARED child subworkflow within
# Designer-declared bounds; reassigning the name is blocked by the AST guard
# (same as ``submit``) so sandbox code cannot shadow it to exfiltrate. Injected
# ONLY when spawning is enabled (declared subagents + a positive budget); the
# default code-action path never injects it.
SPAWN_NAME = "spawn_agent"


class CodeActionError(RuntimeError):
    """Raised inside the sandbox when a bridged tool call is denied / invalid.

    A plain ``RuntimeError`` subtype so the compute sandbox's generic
    ``except Exception`` handler surfaces it to the model as a normal tool
    error (``Error: CodeActionError: ...``) — exactly the design's
    "denial raises in-sandbox, no execute" behaviour — without coupling
    code-action to the text-table error taxonomy.
    """


class SubmitSink:
    """Holds the single JSON-able value a code-action Cell ``submit()``s.

    The sandbox never sees this object; it only sees the bound ``submit``
    closure. ``submitted`` distinguishes "no submit yet" from "submitted
    ``None``".
    """

    __slots__ = ("value", "submitted")

    def __init__(self) -> None:
        self.value: Any = None
        self.submitted: bool = False

    def record(self, value: Any) -> None:
        """Validate JSON-serialisability and store ``value``.

        Raises :class:`CodeActionError` (surfaced to the model) when ``value``
        is not JSON-serialisable, so the typed node output is always a plain
        JSON-able value — never a live object that could leak a tool handle,
        client, or context.
        """
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise CodeActionError(
                "submit() requires a JSON-serialisable value "
                f"(got {type(value).__name__}): {exc}"
            ) from exc
        if self.submitted:
            # Last-write-wins, but make a repeated submit() observable rather
            # than a silent overwrite (the Cell's authoritative output changed).
            logger.info(
                "CODE_ACTION_RESUBMIT submit() called again — last value wins"
            )
        self.value = value
        self.submitted = True


def _make_submit(sink: SubmitSink) -> Callable[[Any], None]:
    """Build the reserved ``submit(value)`` closure.

    Captures only the framework-owned :class:`SubmitSink` — nothing the model
    can reach. Returns ``None`` to the sandbox (the value is captured
    out-of-band) so the model cannot read it back from the namespace.
    """

    def submit(value: Any) -> None:
        sink.record(value)
        return None

    return submit


def _run_coro_blocking(
    event_loop: asyncio.AbstractEventLoop,
    coro: Any,
    *,
    timeout: float,
    tool_name: str,
) -> Any:
    """Schedule *coro* on *event_loop* from a worker thread and block for it.

    Runs on the compute sandbox's worker thread (NOT the loop thread). Bounds the
    wait to *timeout* seconds so the worker thread is GUARANTEED to release: a
    no-timeout block here would let a stuck gated call pin one of the compute
    pool's two worker threads forever (process-wide compute DoS). On timeout the
    orphaned task is cancelled on the loop thread (no leaked task) and a
    :class:`CodeActionError` is raised — surfaced to the model as a tool error.

    The task is created on the loop thread via ``call_soon_threadsafe`` (the only
    safe way to touch the loop from another thread). A done-callback relays the
    result/exception back over a :class:`threading.Event`; a cancelled task is
    relayed as a :class:`CodeActionError` (never a bare ``CancelledError``, which
    the sandbox's ``except Exception`` would not catch).
    """
    task_box: dict[str, asyncio.Task[Any]] = {}
    outcome: dict[str, Any] = {}
    done = threading.Event()

    def _start() -> None:
        task = event_loop.create_task(coro)
        task_box["task"] = task

        def _finish(completed: asyncio.Task[Any]) -> None:
            if completed.cancelled():
                outcome["error"] = CodeActionError(
                    f"tool '{tool_name}' call was cancelled"
                )
            else:
                try:
                    outcome["result"] = completed.result()
                except Exception as exc:  # noqa: BLE001 — relay to worker thread
                    outcome["error"] = exc
            done.set()

        task.add_done_callback(_finish)

    event_loop.call_soon_threadsafe(_start)
    if not done.wait(timeout):
        def _cancel() -> None:
            task = task_box.get("task")
            if task is not None and not task.done():
                task.cancel()

        event_loop.call_soon_threadsafe(_cancel)
        raise CodeActionError(
            f"tool '{tool_name}' call timed out after {timeout:.0f}s"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _run_gated_from_thread(
    loop_ref: weakref.ref[ReactLoop],
    tool_name: str,
    args: dict[str, Any],
) -> Any:
    """Body of a tool closure — runs the gated spine and returns the object.

    Executes on the compute sandbox's worker thread (``ThreadPoolExecutor``),
    NOT the event-loop thread, so blocking on the scheduled coroutine via
    :func:`asyncio.run_coroutine_threadsafe` cannot deadlock the loop.

    Binds only ``loop_ref`` (weak) + ``tool_name`` (str). Looks the loop, its
    event loop, and the tool up at call time; closes over no context/client/
    token. Returns the admitted CONTENT coerced to a plain object.
    """
    loop = loop_ref()
    if loop is None:  # pragma: no cover — loop GC'd mid-call is not reachable
        raise CodeActionError("code-action host is no longer available")

    event_loop = loop._code_action_event_loop
    if event_loop is None:  # pragma: no cover — set before hooks run
        raise CodeActionError("code-action event loop is not available")

    # Re-validate the allowlist at call time from the framework-owned set —
    # never trust a name the sandbox might have constructed.
    if tool_name not in loop._code_action_tool_names:
        raise CodeActionError(f"tool '{tool_name}' is not callable from code")

    tool = loop._all_tools.get(tool_name)
    if tool is None:
        raise CodeActionError(f"tool '{tool_name}' is not available")

    if not isinstance(args, dict):
        raise CodeActionError(
            f"tool arguments must be a dict, got {type(args).__name__}"
        )

    tc = ToolCall(
        id=f"code-action:{tool_name}:{loop._code_action_call_seq}",
        function_name=tool_name,
        arguments=json.dumps(args),
    )
    loop._code_action_call_seq += 1

    loop._code_action_events.append(
        ToolCallEvent(
            node_id=loop._node_id,
            timestamp=_iso_now(),
            tool_name=tool_name,
            arguments=dict(args),
        )
    )

    _tc_id, content, sources, meta = _run_coro_blocking(
        event_loop,
        loop._run_tool_gated(tc, tool, args, origin="code"),
        timeout=loop._code_action_call_timeout,
        tool_name=tool_name,
    )

    # Accepted sources flow to the pool via the loop's source sink — the same
    # path JSON-call sources take (folded into ``ReactResult.sources``). This is
    # what makes a code-action call achieve pool parity.
    if sources:
        loop._code_action_sources.extend(sources)

    loop._code_action_events.append(
        ToolResultEvent(
            node_id=loop._node_id,
            timestamp=_iso_now(),
            tool_name=tool_name,
            result_summary=content[:200],
            source_count=int(meta.get("accepted_source_count", 0)),
            raw_source_count=int(meta.get("raw_source_count", 0)),
            accepted_source_count=int(meta.get("accepted_source_count", 0)),
            rejected_source_count=int(meta.get("rejected_source_count", 0)),
            tool_success=bool(meta.get("tool_success", True)),
            tool_error=str(meta.get("tool_error", "") or ""),
        )
    )

    # A gate denial (HITL deny, per-tool budget exhausted, restriction) comes
    # back as success=False with a ``tool_error`` — raise so the sandbox sees a
    # normal tool error and no value is returned (no execute happened).
    if not bool(meta.get("tool_success", True)):
        raise CodeActionError(str(meta.get("tool_error") or content or "tool failed"))

    # Return the admitted CONTENT only, coerced to a plain object (reuses §1.1).
    # Never the raw ToolResult / context / registry internals.
    return coerce_to_object(content)


def _make_tool_closure(
    loop_ref: weakref.ref[ReactLoop],
    tool_name: str,
) -> Callable[..., Any]:
    """Build one synchronous tool closure for the sandbox.

    The returned callable accepts the same kwargs the model would pass as JSON
    tool arguments (``tool(query="...", num_results=5)``) plus an optional
    single positional dict (``tool({"query": "..."})``). It binds ONLY
    ``loop_ref`` (weak) + ``tool_name`` (str).
    """

    def _call(*args: Any, **kwargs: Any) -> Any:
        if args:
            if len(args) > 1 or kwargs:
                raise CodeActionError(
                    f"{tool_name}(...) takes either a single dict positional "
                    "argument OR keyword arguments, not both"
                )
            sole = args[0]
            if not isinstance(sole, dict):
                raise CodeActionError(
                    f"{tool_name}(...) positional argument must be a dict of "
                    f"tool arguments, got {type(sole).__name__}"
                )
            call_args: dict[str, Any] = dict(sole)
        else:
            call_args = dict(kwargs)
        return _run_gated_from_thread(loop_ref, tool_name, call_args)

    _call.__name__ = tool_name
    return _call


def _make_spawn_closure(
    loop_ref: weakref.ref[ReactLoop],
) -> Callable[[str, str], Any]:
    """Build the GOVERNED ``spawn_agent(name, prompt)`` closure (spec §3.3).

    Mirrors :func:`_make_tool_closure`'s capability model EXACTLY: the returned
    callable binds ONLY ``loop_ref`` (a weakref to the loop) — NO
    ``ExecutionContext`` / ``WorkspaceClient`` / ``user_token`` / spawn runner /
    declared-subagent dict. Everything is looked up at call time from the
    framework-owned loop object, so a model-constructed name can never be trusted
    and no privileged handle is reachable from the sandbox.

    Runs SYNCHRONOUSLY on the compute sandbox worker thread. It:

    a. Resolves the loop + its event loop (raising :class:`CodeActionError` if
       either is gone).
    b. Re-validates ``name`` against the framework-owned ``_spawnable_subagents``
       dict — an undeclared name is REJECTED (never trust a model-built name).
    c. Enforces the TOTAL spawn budget, incrementing the counter ONCE per attempt
       BEFORE the run so a failed/denied spawn still counts (no retry-storm).
    d. Drives ``loop._spawn_runner`` (the executor-bound declared-subworkflow
       runner) on the event-loop thread via :func:`_run_coro_blocking`, bounded by
       the same finite per-call timeout the tool closures use (so a stuck child
       cannot pin a compute worker thread forever).
    e. Returns the child's result COERCED to a plain object (scratchpad-handle
       style) — NEVER the raw child state / context / runtime internals.

    v1 spawns are sequential/blocking from the sandbox thread; ``spawn_agent``
    calls run one at a time and share the single total budget. Parallel spawn
    fan-out (bounded by ``max_concurrent_spawns``) is a DEFERRED enhancement.
    """

    def spawn_agent(name: str, prompt: str) -> Any:
        loop = loop_ref()
        if loop is None:  # pragma: no cover — loop GC'd mid-call is not reachable
            raise CodeActionError("code-action host is no longer available")

        event_loop = loop._code_action_event_loop
        if event_loop is None:  # pragma: no cover — set before hooks run
            raise CodeActionError("code-action event loop is not available")

        runner = loop._spawn_runner
        if runner is None:  # pragma: no cover — closure not injected unless set
            raise CodeActionError("spawn_agent is not available")

        # Re-validate the name from the framework-owned declared set at call time
        # — NEVER trust a name the sandbox might have constructed.
        if not isinstance(name, str):
            raise CodeActionError(
                f"spawn_agent(name, prompt) name must be a str, got {type(name).__name__}"
            )
        inline = loop._spawnable_subagents.get(name)
        if inline is None:
            raise CodeActionError(
                f"subagent '{name}' is not in the declared spawnable set"
            )

        # Budget: increment ONCE per attempt BEFORE the run (a failed spawn still
        # counts) so a denied/erroring spawn cannot drive a retry-storm.
        if loop._spawn_count >= loop._spawn_budget:
            raise CodeActionError("spawn budget exhausted")
        loop._spawn_count += 1

        # Future parallel fan-out ceiling — enforced as a guard even though v1
        # spawns block sequentially (so concurrency is always 1 here).
        if loop._max_concurrent_spawns < 1:  # pragma: no cover — clamped >=1 by config
            raise CodeActionError("max_concurrent_spawns must be >= 1")

        result = _run_coro_blocking(
            event_loop,
            runner(name=name, prompt=str(prompt), inline=inline),
            timeout=loop._code_action_call_timeout,
            tool_name=f"spawn:{name}",
        )

        # Return a plain, scratchpad-handle-style value: coerce a string child
        # output to a structured object; for a non-str result, return it only if
        # it is JSON-able, else coerce its string form. This makes the
        # "JSON-able value only" contract STRUCTURAL (mirrors submit()'s guard at
        # SubmitSink.record) rather than relying on the upstream invariant that a
        # child's primary output is always str/dict/list — so a live object /
        # handle / context can never round-trip back into the sandbox. NEVER the
        # raw child state / context / runtime internals.
        if isinstance(result, str):
            return coerce_to_object(result)
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            return coerce_to_object(str(result))
        return result

    spawn_agent.__name__ = SPAWN_NAME
    return spawn_agent


def select_code_action_tool_names(loop: ReactLoop) -> list[str]:
    """Return the bound tool names eligible to be exposed as code callables.

    Intersection of the per-Cell allowlist (``loop._code_action_allowlist``),
    the loop's actually-bound tools, and the allowed source kinds. ``compute``
    and other builtins are never bridged. Order follows the allowlist.
    """
    allowlist = loop._code_action_allowlist
    if not allowlist:
        return []
    selected: list[str] = []
    for name in allowlist:
        tool = loop._all_tools.get(name)
        if tool is None:
            logger.warning(
                "CODE_ACTION_TOOL_UNKNOWN node=%s tool=%s — not bound, skipped",
                loop._node_id,
                name,
            )
            continue
        kind = tool_source_kind(tool.definition)
        if kind not in _CODE_ACTION_ALLOWED_SOURCE_KINDS:
            logger.warning(
                "CODE_ACTION_TOOL_BLOCKED node=%s tool=%s source_kind=%s — "
                "not an allowed data tool, skipped",
                loop._node_id,
                name,
                kind,
            )
            continue
        if name not in selected:
            selected.append(name)
    return selected


def install_code_action_hook(
    loop: ReactLoop, compute: Any, *, spawn_enabled: bool = False
) -> SubmitSink:
    """Register the code-action closures + ``submit`` on the compute sandbox.

    Uses ``set_before_execute_hook`` (the proven text-table seam) so the
    closures are (re)injected immediately before every sandbox execution,
    keeping their weak loop reference and bound tool names current. Returns the
    :class:`SubmitSink` the loop reads after the Cell runs.

    The injected names — each bridged tool plus ``submit`` (and ``spawn_agent``
    when ``spawn_enabled``) — are the ONLY new namespace entries; no
    client/context/auth object is ever injected. ``spawn_agent`` is injected
    ONLY when ``spawn_enabled`` is True (the caller gates this on declared
    subagents + a positive budget), keeping the non-spawn path byte-identical.

    SECURITY: ``reserve_sandbox_names`` REPLACES the reserved set, so the
    ``submit`` / tool-closure / ``spawn_agent`` names are reserved together in a
    single call — otherwise reserving spawn separately would clear the
    code-action reservations and reopen the shadowing escape.
    """
    sink = SubmitSink()
    loop_ref: weakref.ref[ReactLoop] = weakref.ref(loop)
    tool_names = list(loop._code_action_tool_names)

    if not hasattr(compute, "set_before_execute_hook"):
        raise TypeError(
            "compute must expose set_before_execute_hook for code-action; "
            f"got {type(compute).__name__}"
        )

    # SECURITY: pandas/numpy are NOT auto-enabled by this code-action hook. The
    # raw ``numpy.load(allow_pickle=True)`` / ``pandas.read_pickle`` / ``read_csv``
    # reaches are arbitrary-code / file / network via PLAIN attribute calls (not
    # blocked dunders). The deferred US-109 §5.2 follow-up now ships a SAFE
    # pandas/numpy SUBSET (``tools/builtins/compute_dataframe.py``): a curated
    # module facade (only DataFrame/Series/concat/merge/... + numpy array/math/
    # stats — every ``read_*``/``load``/``save``/pickle/``eval`` omitted) plus an
    # AST instance-method denylist (``to_pickle``/``to_csv``/``to_sql``/
    # ``tofile``/``df.eval``/...). It is opt-in via the compute tool's
    # ``enable_dataframes`` flag (YAML ``config.enable_dataframes: true``), NOT
    # forced on by the bridge — so this hook stays dataframe-free unless a Cell's
    # compute tool was explicitly built with it. See the design doc (Codex F12).

    def _refresh(host_compute: Any) -> None:
        host_compute.inject_variable(SUBMIT_NAME, _make_submit(sink))
        for name in tool_names:
            host_compute.inject_variable(name, _make_tool_closure(loop_ref, name))
        if spawn_enabled:
            host_compute.inject_variable(SPAWN_NAME, _make_spawn_closure(loop_ref))

    compute.set_before_execute_hook("code_action_callables", _refresh)
    # Reserve the injected names so sandbox code can CALL but not rebind them
    # (the AST guard rejects shadowing ``submit`` / a tool closure / spawn_agent).
    # SECURITY: a single call — reserve_sandbox_names REPLACES the set.
    reserved = {SUBMIT_NAME, *tool_names}
    if spawn_enabled:
        reserved.add(SPAWN_NAME)
    if hasattr(compute, "reserve_sandbox_names"):
        compute.reserve_sandbox_names(frozenset(reserved))
    # Inject once now so the names exist even before the first hook fire.
    _refresh(compute)
    logger.info(
        "CODE_ACTION_INSTALLED node=%s tools=%s spawn_enabled=%s",
        loop._node_id,
        tool_names,
        spawn_enabled,
    )
    return sink


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).isoformat()


__all__ = [
    "SPAWN_NAME",
    "SUBMIT_NAME",
    "CodeActionError",
    "SubmitSink",
    "install_code_action_hook",
    "select_code_action_tool_names",
]
