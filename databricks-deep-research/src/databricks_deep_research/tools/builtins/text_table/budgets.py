from __future__ import annotations

from dataclasses import dataclass, field

from .error_codes import ErrorCode, ToolError, ToolErrorException

# Per-statement hard caps (always enforced, not user-tunable).
PER_STMT_LIMIT_ROWS = 5_000
PER_STMT_LIMIT_BYTES = 8 * 1024 * 1024
PER_STMT_LIMIT_GROUPS = 1_000


class BudgetExceeded(ToolErrorException):
    """Raised when a Budget3D dimension is exceeded inside a compute turn."""

    def __init__(self, dimension: str, used: float, cap: float) -> None:
        self.dimension = dimension
        self.used = used
        self.cap = cap
        super().__init__(
            ToolError(
                error_code=ErrorCode.BUDGET_EXCEEDED,
                message=(
                    f"compute-turn budget exceeded on '{dimension}': "
                    f"used={used} cap={cap}"
                ),
                hint=(
                    "Either reduce the inner-call count, scope the SQL "
                    "with a stricter 'where', or split work across compute turns."
                ),
                details={"dimension": dimension, "used": used, "cap": cap},
            )
        )


@dataclass
class Budget3D:
    max_calls: int = 30
    max_rows: int = 50_000
    max_wall_clock_s: float = 30.0
    calls_used: int = field(default=0, init=False)
    rows_used: int = field(default=0, init=False)
    wall_clock_used_s: float = field(default=0.0, init=False)

    def tick(self, rows: int = 0, wall_clock_s: float = 0.0) -> None:
        next_calls = self.calls_used + 1
        next_rows = self.rows_used + rows
        next_wc = self.wall_clock_used_s + wall_clock_s
        if next_calls > self.max_calls:
            raise BudgetExceeded("calls", next_calls, self.max_calls)
        if next_rows > self.max_rows:
            raise BudgetExceeded("rows", next_rows, self.max_rows)
        if next_wc > self.max_wall_clock_s:
            raise BudgetExceeded("wall_clock_s", next_wc, self.max_wall_clock_s)
        self.calls_used = next_calls
        self.rows_used = next_rows
        self.wall_clock_used_s = next_wc
