"""Static guard: ``db/grant_permissions.py`` must never touch the
``deep_research_state`` storage schema.

That schema is owned by the app SP and created at app lifespan by
``LakebaseBackend.migrate()``. Postgres on Lakebase rejects
cross-principal ``ALTER OWNER``; any developer-run ``CREATE SCHEMA`` /
``ALTER SCHEMA ... OWNER TO ...`` on it would leave the schema
developer-owned and the app would crash at startup with "must be owner
of table ...". The historical poisoned-schema state is documented in the
module docstring; this test is the regression guard.

We snapshot the SQL string literals in the module's grant function. If a
future edit adds a statement that mentions ``deep_research_state``, this
test fails before review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GRANT_PERMISSIONS_PY = (
    REPO_ROOT / "src" / "deep_research" / "db" / "grant_permissions.py"
)

# The storage schema name the app SP owns. Hardcoded to match
# ``Settings.storage_schema`` default.
APP_SP_OWNED_SCHEMA = "deep_research_state"


def _function_node(
    module_path: Path, function_name: str
) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Locate the named function in the module's AST."""
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
            and node.name == function_name
        ):
            return node
    pytest.fail(f"{function_name} not found in {module_path}")
    raise AssertionError("unreachable")  # for type-checker


def _collect_conn_execute_sql(
    fn_node: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[str]:
    """Return every string literal passed as the first positional arg to
    ``conn.execute(...)`` inside ``fn_node``.

    This captures the actual SQL we run, not docstrings or error
    messages that may incidentally contain the word ``GRANT``.
    """
    sql_literals: list[str] = []
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "execute"
            and isinstance(func.value, ast.Name)
            and func.value.id == "conn"
        ):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            sql_literals.append(arg.value)
        elif isinstance(arg, ast.JoinedStr):
            # f-string — concatenate literal parts, skip interpolations
            parts = []
            for value in arg.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    # placeholder for interpolated piece — we only need to
                    # see the surrounding SQL keywords, not the var.
                    parts.append("$VAR")
            sql_literals.append("".join(parts))
    return sql_literals


def test_grant_permissions_never_mentions_deep_research_state() -> None:
    fn = _function_node(GRANT_PERMISSIONS_PY, "grant_permissions_to_app")
    sql_statements = _collect_conn_execute_sql(fn)

    offenders = [s for s in sql_statements if APP_SP_OWNED_SCHEMA in s]

    assert not offenders, (
        f"grant_permissions_to_app must never reference the "
        f"{APP_SP_OWNED_SCHEMA!r} schema in conn.execute(...). The app SP "
        "owns it and creates it at lifespan; developer-run grants/DDL "
        "would poison ownership. Offending SQL: "
        f"{offenders}"
    )


def test_grant_permissions_only_touches_public_schema_and_db_level() -> None:
    """Whitelist check: every GRANT / ALTER DEFAULT PRIVILEGES statement
    that ``grant_permissions_to_app`` runs against the DB must target
    either schema ``public`` or be at the DATABASE level. If a future
    edit grants on a non-``public`` schema, this test fails.
    """
    fn = _function_node(GRANT_PERMISSIONS_PY, "grant_permissions_to_app")
    sql_statements = _collect_conn_execute_sql(fn)

    privilege_statements = [
        s
        for s in sql_statements
        if "GRANT " in s.upper() or "ALTER DEFAULT PRIVILEGES" in s.upper()
    ]
    assert privilege_statements, (
        "grant_permissions_to_app appears to contain no GRANT statements; "
        "the snapshot likely needs updating."
    )

    bad = []
    for stmt in privilege_statements:
        upper = stmt.upper()
        # DB-level grant ("GRANT ... ON DATABASE \"<db>\" TO ...") is fine.
        if "ON DATABASE" in upper:
            continue
        # Schema-scoped grant must reference public.
        if "SCHEMA PUBLIC" in upper:
            continue
        bad.append(stmt)

    assert not bad, (
        "grant_permissions_to_app contains GRANT/ALTER DEFAULT PRIVILEGES "
        "statements that target neither DATABASE-level nor schema "
        "``public``. Each privilege grant must be one of those two "
        "scopes — anything else risks touching app-SP-owned schemas. "
        f"Offending statements: {bad}"
    )
