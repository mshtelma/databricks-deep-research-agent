from __future__ import annotations

import pytest

from databricks_deep_research.tools.builtins.text_table.budgets import (
    Budget3D,
    BudgetExceeded,
    PER_STMT_LIMIT_BYTES,
    PER_STMT_LIMIT_GROUPS,
    PER_STMT_LIMIT_ROWS,
)
from databricks_deep_research.tools.builtins.text_table.error_codes import ErrorCode


def test_default_caps() -> None:
    b = Budget3D()
    assert b.max_calls == 30
    assert b.max_rows == 50_000
    assert b.max_wall_clock_s == pytest.approx(30.0)


def test_tick_accumulates() -> None:
    b = Budget3D()
    b.tick(rows=100, wall_clock_s=1.5)
    b.tick(rows=200, wall_clock_s=0.5)
    assert b.calls_used == 2
    assert b.rows_used == 300
    assert b.wall_clock_used_s == pytest.approx(2.0)


def test_call_cap_raises() -> None:
    b = Budget3D(max_calls=2)
    b.tick()
    b.tick()
    with pytest.raises(BudgetExceeded) as exc:
        b.tick()
    assert exc.value.dimension == "calls"
    assert exc.value.error.error_code is ErrorCode.BUDGET_EXCEEDED


def test_row_cap_raises() -> None:
    b = Budget3D(max_rows=100)
    b.tick(rows=50)
    with pytest.raises(BudgetExceeded) as exc:
        b.tick(rows=51)
    assert exc.value.dimension == "rows"


def test_wallclock_cap_raises() -> None:
    b = Budget3D(max_wall_clock_s=2.0)
    b.tick(wall_clock_s=1.0)
    with pytest.raises(BudgetExceeded) as exc:
        b.tick(wall_clock_s=1.5)
    assert exc.value.dimension == "wall_clock_s"


def test_per_statement_constants() -> None:
    assert PER_STMT_LIMIT_ROWS == 5_000
    assert PER_STMT_LIMIT_BYTES == 8 * 1024 * 1024
    assert PER_STMT_LIMIT_GROUPS == 1_000
