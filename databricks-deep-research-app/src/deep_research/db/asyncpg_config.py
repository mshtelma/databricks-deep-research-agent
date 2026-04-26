"""PgBouncer-safe asyncpg configuration for Databricks Lakebase.

Lakebase Postgres sits behind PgBouncer in transaction pooling mode. In that
mode, server-side prepared statements do not survive across queries, so
asyncpg's default named-prepared-statement cache causes:
  1. DuplicatePreparedStatementError on name collision, or
  2. indefinite `prepare()` hangs when the backend is multiplexed away.

The canonical fix is to zero asyncpg's prepared-statement cache, and when
the installed asyncpg version supports it, additionally force UNNAMED
prepared statements on the wire via ``prepared_statement_name_func``.

References:
  - https://magicstack.github.io/asyncpg/current/faq.html
  - asyncpg PR #846 — ``prepared_statement_name_func`` (v0.30+)

Version-awareness: older asyncpg (0.29 and below, and some 0.30/0.31
builds) does not accept ``prepared_statement_name_func`` at
``connect()``; passing it raises ``TypeError``. We probe the installed
asyncpg at import time and only include the kwarg when it is actually
supported. When unsupported, ``statement_cache_size=0`` alone is
sufficient for most PgBouncer workloads; the fallback still avoids the
DuplicatePreparedStatementError hazard described above.
"""

import inspect
from typing import Any

import asyncpg

from deep_research.core.config import Settings


def _unnamed_prepared_statement() -> str:
    """Force asyncpg to use unnamed prepared statements (PgBouncer-safe).

    Unnamed prepared statements are ephemeral at the Postgres protocol level
    and cannot outlive a transaction, which is exactly what PgBouncer
    transaction pooling requires.
    """
    return ""


def _asyncpg_supports_name_func() -> bool:
    """Probe whether the installed asyncpg accepts prepared_statement_name_func.

    The kwarg was added in asyncpg PR #846 and is present in newer
    releases. On older wheels it doesn't appear in the ``connect()``
    signature at all and raises ``TypeError`` if passed.

    We inspect the function signature directly via ``inspect.signature``
    so we survive decorators/``functools.wraps`` and aren't fooled by
    incidental string matches inside the source (a comment that mentions
    the kwarg would have falsely satisfied the prior substring check).
    Falls back to ``False`` on any reflection error — the conservative
    default, which still keeps PgBouncer-safe ``statement_cache_size=0``
    in effect.
    """
    try:
        params = inspect.signature(asyncpg.connect).parameters
    except (TypeError, ValueError):
        return False
    return "prepared_statement_name_func" in params


_SUPPORTS_NAME_FUNC: bool = _asyncpg_supports_name_func()


def lakebase_asyncpg_connect_args(settings: Settings) -> dict[str, Any]:
    """Return `connect_args` for SQLAlchemy `create_async_engine` against Lakebase.

    Returns an empty dict for non-Lakebase deployments so plain Postgres is
    unaffected. Only includes ``prepared_statement_name_func`` when the
    installed asyncpg actually supports it.

    Schema scoping is NOT applied here. The SQLAlchemy async engine is
    shared between the new storage-engine code (which writes to
    ``deep_research_state.*``) and the legacy SQLAlchemy ORM (which reads
    from ``public.*``). Applying ``search_path`` at the connect-args level
    would force every pooled connection to resolve unqualified legacy
    references (``custom_agents``, ``research_events``, …) into
    ``deep_research_state`` first — where tables of the same name exist
    with a completely different column layout — producing
    ``UndefinedColumnError``. The new storage engine instead
    fully-qualifies every table reference (``{schema}.chat_meta``),
    matching how ``SQLWarehouseBackend`` already emits ``${ns}.table``.
    """
    if not settings.use_lakebase:
        return {}
    args: dict[str, Any] = {
        "ssl": True,
        "statement_cache_size": 0,
        "command_timeout": settings.db_command_timeout,
    }
    if _SUPPORTS_NAME_FUNC:
        args["prepared_statement_name_func"] = _unnamed_prepared_statement
    return args


def lakebase_engine_kwargs(settings: Settings) -> dict[str, Any]:  # noqa: ARG001
    """Return SQLAlchemy engine-level kwargs for Lakebase.

    Historical note: an earlier version of this function returned
    ``{"prepared_statement_cache_size": 0}`` intending to zero the
    SQLAlchemy dialect cache. That key is only valid for the
    **psycopg2** dialect; the **asyncpg** dialect we use rejects it at
    ``create_async_engine`` time with::

        TypeError: Invalid argument(s) 'prepared_statement_cache_size'
        sent to create_engine()

    Under the asyncpg dialect, the dialect's own statement caching is
    already fully controlled by ``statement_cache_size=0`` in
    ``connect_args`` (see ``lakebase_asyncpg_connect_args``); no
    engine-level kwarg is needed. Returning an empty dict preserves
    backward compatibility with callers that splat ``**engine_kwargs``.
    """
    return {}


def lakebase_raw_asyncpg_kwargs() -> dict[str, Any]:
    """Return kwargs for raw `asyncpg.connect()` calls (e.g., bootstrap).

    Used by code paths that bypass SQLAlchemy. Version-aware: omits
    ``prepared_statement_name_func`` when the installed asyncpg does not
    support it, preventing startup failures like
    ``TypeError: connect() got an unexpected keyword argument
    'prepared_statement_name_func'`` on older asyncpg wheels.
    """
    kwargs: dict[str, Any] = {"statement_cache_size": 0}
    if _SUPPORTS_NAME_FUNC:
        kwargs["prepared_statement_name_func"] = _unnamed_prepared_statement
    return kwargs
