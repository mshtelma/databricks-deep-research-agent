"""Save-time signature introspection for ``uc_function`` tool declarations.

When an author declares a ``uc_function`` tool, we introspect its signature via
``DESCRIBE FUNCTION`` under the caller's OBO identity and fill ``config.params``
(when missing) and ``config.returns_table`` before the definition is persisted.
DESCRIBE needs only the ``BROWSE`` privilege (unlike ``information_schema``, which
needs ``USE CATALOG``), so this also works for workshop/demo catalogs a user can
see but not use. Runtime stays introspection-free (the resolver rebuilds every
tool per request) and tolerates empty params, so this is a pure authoring
enhancement — **fail-soft by contract**: on any error the declaration is
persisted as-is and a warning is returned.

Placement: this runs at the agents_v2 *route* level (which can build the OBO
workspace client), inside ``asyncio.to_thread`` with a short cap, NEVER on the
synchronous ``AgentV2Service`` / ``normalize_ast`` path (the advisory-save design
keeps the request off the client's 30s timeout).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from deep_research.agent_designer.uc_metadata import FQN_RE, get_signature

if TYPE_CHECKING:
    from deep_research.agent_designer.uc_metadata import SqlExecutor

logger = logging.getLogger(__name__)


def _collect_targets(definition: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Collect ``(config, fqn)`` for every uc_function decl with a valid FQN.

    We introspect all of them (not only param-less ones) because
    ``returns_table`` must be correct even when the author supplied ``params`` —
    but the fill step never overwrites explicit non-empty ``params`` (author
    override wins). Identifiers are lowercased (UC resolves case-insensitively).
    """
    tools = definition.get("tools")
    if not isinstance(tools, list):
        return []
    targets: list[tuple[dict[str, Any], str]] = []
    for tool in tools:
        if not (isinstance(tool, dict) and tool.get("kind") == "uc_function"):
            continue
        config = tool.get("config")
        if not isinstance(config, dict):
            continue
        fqn = str(config.get("function") or "").strip().lower()
        if not FQN_RE.fullmatch(fqn):
            continue
        targets.append((config, fqn))
    return targets


def _introspect_all(
    sql_executor: SqlExecutor,
    targets: list[tuple[dict[str, Any], str]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str]]:
    """Synchronous body run in a worker thread: DESCRIBE each function, building
    the ``(config, signature)`` fills to apply on the caller (main) thread."""
    fills: list[tuple[dict[str, Any], dict[str, Any]]] = []
    warnings: list[str] = []
    for config, fqn in targets:
        try:
            sig = get_signature(sql_executor, fqn)
        except Exception as exc:  # noqa: BLE001 - fail-soft, one function at a time
            warnings.append(f"uc_function introspection failed for {fqn}: {exc}")
            continue
        if not sig.get("scalar", True):
            warnings.append(
                f"uc_function {fqn} has a non-scalar (array/map/struct) parameter; "
                "scalar params only in v1 — params left empty"
            )
        fills.append((config, sig))
    return fills, warnings


async def introspect_and_fill_uc_params(
    definition: dict[str, Any],
    sql_executor: SqlExecutor,
    *,
    timeout_seconds: float = 10.0,
) -> list[str]:
    """Fill ``config.params`` (when missing) and ``config.returns_table`` for
    uc_function decls.

    Mutates ``definition`` in place. Returns author-facing warnings. Never
    raises: a timeout or query error leaves the declaration as-is (runtime
    tolerates empty params) and is reported as a warning.
    """
    targets = _collect_targets(definition)
    if not targets:
        return []
    try:
        fills, warnings = await asyncio.wait_for(
            asyncio.to_thread(_introspect_all, sql_executor, targets),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning("UC_FUNCTION_INTROSPECT_TIMEOUT count=%d", len(targets))
        return [
            "uc_function parameter introspection timed out; declared params "
            "left empty (add config.params manually if the function takes "
            "arguments)"
        ]
    except Exception as exc:  # noqa: BLE001 - fail-soft by contract
        logger.warning("UC_FUNCTION_INTROSPECT_FAILED error=%s", str(exc)[:200])
        return [
            f"uc_function parameter introspection failed ({exc}); declared "
            "params left empty"
        ]
    for config, sig in fills:
        # returns_table is authoritative — it fixes the runtime SQL shape
        # (SELECT * FROM fn(..) vs SELECT fn(..)), so always set it.
        config["returns_table"] = bool(sig.get("returns_table", False))
        # Fill params only when the author didn't supply them (override wins) and
        # only for scalar signatures (non-scalar left empty -> untyped pass-through).
        existing = config.get("params")
        if not (isinstance(existing, list) and existing) and sig.get("scalar", True):
            config["params"] = sig.get("params", [])
    return warnings
