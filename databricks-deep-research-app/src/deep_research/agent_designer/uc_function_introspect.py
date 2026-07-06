"""Save-time parameter introspection for ``uc_function`` tool declarations.

When an author declares a ``uc_function`` tool without explicit
``config.params``, we discover the function's signature from
``<catalog>.information_schema.parameters`` under the caller's OBO identity and
fill ``config.params`` before the definition is persisted. Runtime stays
introspection-free (the resolver rebuilds every tool per request), and the
runtime tolerates empty params (it passes provided args through untyped), so
this is a pure authoring enhancement — **fail-soft by contract**: on any error
the declaration is persisted without params and a warning is returned.

Placement: this runs at the agents_v2 *route* level (which can build the OBO
workspace client), inside ``asyncio.to_thread`` with a short cap, NEVER on the
synchronous ``AgentV2Service`` / ``normalize_ast`` path (the advisory-save
design keeps the request off the client's 30s timeout).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from deep_research.agent_designer.uc_metadata import (
    FQN_RE,
    parse_parameter_rows,
    run_parameters_query,
)

if TYPE_CHECKING:
    from databricks_deep_research.tools.builtins.text_table.tools._common import (
        SqlExecutor,
    )

logger = logging.getLogger(__name__)


def _collect_targets(
    definition: dict[str, Any],
) -> dict[tuple[str, str], list[tuple[dict[str, Any], str]]]:
    """Group uc_function decls needing introspection by (catalog, schema).

    Skips decls with explicit non-empty ``config.params`` (author override
    wins) and decls with a malformed FQN (semantic validation reports those).
    Identifiers are lowercased — information_schema stores them lowercase.
    """
    tools = definition.get("tools")
    if not isinstance(tools, list):
        return {}
    targets: dict[tuple[str, str], list[tuple[dict[str, Any], str]]] = {}
    for tool in tools:
        if not (isinstance(tool, dict) and tool.get("kind") == "uc_function"):
            continue
        config = tool.get("config")
        if not isinstance(config, dict):
            continue
        existing = config.get("params")
        if isinstance(existing, list) and existing:
            continue
        fqn = str(config.get("function") or "").strip()
        if not FQN_RE.fullmatch(fqn):
            continue
        catalog, schema, fn = fqn.split(".")
        targets.setdefault((catalog.lower(), schema.lower()), []).append(
            (config, fn.lower())
        )
    return targets


def _params_by_fn(
    rows: list[dict[str, Any]],
    warnings: list[str],
    catalog: str,
    schema: str,
) -> dict[str, list[dict[str, Any]] | None]:
    """Parse parameter rows into per-function param lists, appending an author
    warning for each function rejected for a non-scalar (array/map/struct)
    parameter. Thin wrapper over the shared ``uc_metadata.parse_parameter_rows``."""
    by_fn = parse_parameter_rows(rows)
    for fn, params in by_fn.items():
        if params is None:
            warnings.append(
                f"uc_function {catalog}.{schema}.{fn} has a non-scalar "
                "(array/map/struct) parameter; scalar functions only in v1 — "
                "params left empty"
            )
    return by_fn


def _query_all(
    sql_executor: SqlExecutor,
    targets: dict[tuple[str, str], list[tuple[dict[str, Any], str]]],
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[str]]:
    """Synchronous body run in a worker thread: query each group, build the
    ``(config, params)`` fills to apply on the caller (main) thread."""
    fills: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    warnings: list[str] = []
    for (catalog, schema), items in targets.items():
        fn_names = sorted({fn for _cfg, fn in items})
        try:
            rows = run_parameters_query(sql_executor, catalog, schema, fn_names)
        except Exception as exc:  # noqa: BLE001 - fail-soft, one group at a time
            warnings.append(
                f"uc_function introspection query failed for {catalog}.{schema}: "
                f"{exc}"
            )
            continue
        by_fn = _params_by_fn(rows, warnings, catalog, schema)
        for config, fn in items:
            if fn in by_fn:
                params = by_fn[fn]
                if params is None:  # non-scalar param — already warned
                    continue
                fills.append((config, params))
            else:
                # No parameter rows: a no-arg function (or an unknown name that
                # runtime SQL will surface). Empty params => runtime passes
                # provided args through untyped.
                fills.append((config, []))
    return fills, warnings


async def introspect_and_fill_uc_params(
    definition: dict[str, Any],
    sql_executor: SqlExecutor,
    *,
    timeout_seconds: float = 10.0,
) -> list[str]:
    """Fill ``config.params`` for uc_function decls lacking explicit params.

    Mutates ``definition`` in place. Returns author-facing warnings. Never
    raises: a timeout or query error leaves params empty (runtime tolerates
    that) and is reported as a warning.
    """
    targets = _collect_targets(definition)
    if not targets:
        return []
    try:
        fills, warnings = await asyncio.wait_for(
            asyncio.to_thread(_query_all, sql_executor, targets),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning("UC_FUNCTION_INTROSPECT_TIMEOUT groups=%d", len(targets))
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
    for config, params in fills:
        config["params"] = params
    return warnings
