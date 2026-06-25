"""Scored role inference for DISCOVERED bindings.

For each role we compute a feature score against every column and pick the
top-scoring column above its threshold. Required roles (content/id) raise
``INFERENCE_FAILED`` with the top-3 candidates if no column clears its bar;
optional roles (order/partition/type/date) are simply set to ``None``.

See spec §5.1 for the formulas.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .binding import RoleMap
from .error_codes import ErrorCode, ToolError, ToolErrorException
from .schema_cache import Schema, SchemaColumn

CONTENT_NAME_RE = re.compile(r"(content|text|chunk|body|passage)", re.IGNORECASE)
ID_NAME_RE = re.compile(r"(_id$|^id$)", re.IGNORECASE)
ORDER_NAME_RE = re.compile(r"(_index|_order|_position|_seq)", re.IGNORECASE)
PARTITION_NAME_RE = re.compile(
    r"(file_name|source|document|doc_name|url)", re.IGNORECASE
)
TYPE_NAME_RE = re.compile(r"(type|kind|category)", re.IGNORECASE)
DATE_NAME_RE = re.compile(r"(date|created_at|published_at)", re.IGNORECASE)

_STRING_TYPES = frozenset({"STRING", "VARCHAR", "CHAR"})
_INT_TYPES = frozenset({"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"})
_DATE_TYPES = frozenset({"DATE", "TIMESTAMP", "TIMESTAMP_NTZ"})

_TYPE_DISTINCT_PROBE_LIMIT = 16


@dataclass(frozen=True)
class RoleCandidate:
    """Top-N candidate surfaced when role inference fails / is reported."""

    column: str
    score: float
    rationale: str


@dataclass(frozen=True)
class _ColumnStats:
    avg_str_len: float
    null_rate: float
    distinct_ratio: float
    distinct_count: int


def _name_match(col: str, regex: re.Pattern[str]) -> float:
    return 1.0 if regex.search(col) else 0.0


def _normalize_type(t: str) -> str:
    return t.upper().split("(", 1)[0].strip()


def _is_string(col: SchemaColumn) -> bool:
    return _normalize_type(col.data_type) in _STRING_TYPES


def _is_int(col: SchemaColumn) -> bool:
    return _normalize_type(col.data_type) in _INT_TYPES


def _is_date(col: SchemaColumn) -> bool:
    return _normalize_type(col.data_type) in _DATE_TYPES


def _compute_stats(
    schema: Schema, sample_rows: Sequence[Mapping[str, Any]] | None
) -> dict[str, _ColumnStats]:
    """Compute per-column stats from sample rows.

    Degenerate defaults when ``sample_rows`` is ``None`` or empty:
    avg_str_len = 500 (so avg_str_len_norm = 0.5),
    null_rate = 0.0 (no evidence of nulls),
    distinct_ratio = 0.5 (neutral midpoint),
    distinct_count = 9999 (force-fail the type-column probe).
    """
    out: dict[str, _ColumnStats] = {}
    rows = list(sample_rows) if sample_rows else []
    n_rows = len(rows)

    if n_rows == 0:
        for col in schema.columns:
            out[col.name] = _ColumnStats(
                avg_str_len=500.0,
                null_rate=0.0,
                distinct_ratio=0.5,
                distinct_count=9999,
            )
        return out

    for col in schema.columns:
        values = [row.get(col.name) for row in rows]
        nulls = sum(1 for v in values if v is None)
        non_null = [v for v in values if v is not None]
        if non_null:
            distinct_set = {repr(v) for v in non_null}
            distinct_count = len(distinct_set)
            distinct_ratio = distinct_count / n_rows
            if _is_string(col):
                avg_str_len = sum(len(str(v)) for v in non_null) / len(non_null)
            else:
                avg_str_len = 0.0
        else:
            distinct_count = 0
            distinct_ratio = 0.0
            avg_str_len = 0.0
        null_rate = nulls / n_rows
        out[col.name] = _ColumnStats(
            avg_str_len=avg_str_len,
            null_rate=null_rate,
            distinct_ratio=distinct_ratio,
            distinct_count=distinct_count,
        )
    return out


def _score_content(col: SchemaColumn, stats: _ColumnStats) -> tuple[float, str]:
    if not _is_string(col):
        return 0.0, "type filter (STRING) excluded"
    avg_len_norm = min(stats.avg_str_len / 1000.0, 1.0)
    name = _name_match(col.name, CONTENT_NAME_RE)
    score = (
        0.4 * avg_len_norm
        + 0.3 * name
        + 0.2 * (1.0 - stats.null_rate)
        + 0.1 * (1.0 - stats.distinct_ratio)
    )
    rationale = (
        f"avg_len_norm={avg_len_norm:.2f} name_match={name:.0f} "
        f"non_null={1.0 - stats.null_rate:.2f} "
        f"non_distinct={1.0 - stats.distinct_ratio:.2f}"
    )
    return score, rationale


def _score_id(col: SchemaColumn, stats: _ColumnStats) -> tuple[float, str]:
    if not _is_string(col):
        return 0.0, "type filter (STRING) excluded"
    name = _name_match(col.name, ID_NAME_RE)
    score = (
        0.5 * name
        + 0.3 * stats.distinct_ratio
        + 0.2 * (1.0 - stats.null_rate)
    )
    rationale = (
        f"name_match={name:.0f} distinct_ratio={stats.distinct_ratio:.2f} "
        f"non_null={1.0 - stats.null_rate:.2f}"
    )
    return score, rationale


def _score_order(col: SchemaColumn, stats: _ColumnStats) -> tuple[float, str]:
    if not _is_int(col):
        return 0.0, "type filter (INT/BIGINT) excluded"
    name = _name_match(col.name, ORDER_NAME_RE)
    score = 0.7 * name + 0.3 * (1.0 - stats.null_rate)
    rationale = f"name_match={name:.0f} non_null={1.0 - stats.null_rate:.2f}"
    return score, rationale


def _score_partition(col: SchemaColumn, stats: _ColumnStats) -> tuple[float, str]:
    if not _is_string(col):
        return 0.0, "type filter (STRING) excluded"
    name = _name_match(col.name, PARTITION_NAME_RE)
    score = 0.7 * name + 0.3 * (1.0 - stats.null_rate)
    rationale = f"name_match={name:.0f} non_null={1.0 - stats.null_rate:.2f}"
    return score, rationale


def _score_type(col: SchemaColumn, stats: _ColumnStats) -> tuple[float, str]:
    if not _is_string(col):
        return 0.0, "type filter (STRING) excluded"
    if stats.distinct_count > _TYPE_DISTINCT_PROBE_LIMIT:
        return 0.0, f"distinct_count={stats.distinct_count} > {_TYPE_DISTINCT_PROBE_LIMIT}"
    name = _name_match(col.name, TYPE_NAME_RE)
    score = 0.7 * name + 0.3 * (1.0 - stats.null_rate)
    rationale = (
        f"name_match={name:.0f} distinct_count={stats.distinct_count} "
        f"non_null={1.0 - stats.null_rate:.2f}"
    )
    return score, rationale


def _score_date(col: SchemaColumn, stats: _ColumnStats) -> tuple[float, str]:
    if not _is_date(col):
        return 0.0, "type filter (DATE/TIMESTAMP) excluded"
    name = _name_match(col.name, DATE_NAME_RE)
    score = 0.7 * name + 0.3 * (1.0 - stats.null_rate)
    rationale = f"name_match={name:.0f} non_null={1.0 - stats.null_rate:.2f}"
    return score, rationale


_Scorer = Callable[[SchemaColumn, _ColumnStats], tuple[float, str]]


def _score_all(
    schema: Schema, stats: dict[str, _ColumnStats], scorer: _Scorer
) -> list[RoleCandidate]:
    out: list[RoleCandidate] = []
    for col in schema.columns:
        cs = stats[col.name]
        score, rationale = scorer(col, cs)
        out.append(RoleCandidate(column=col.name, score=score, rationale=rationale))
    out.sort(key=lambda c: -c.score)
    return out


def _pick_required(
    role: str,
    candidates: list[RoleCandidate],
    threshold: float,
) -> str:
    if not candidates or candidates[0].score < threshold:
        top3 = [
            {
                "column": c.column,
                "score": round(c.score, 3),
                "rationale": c.rationale,
            }
            for c in candidates[:3]
        ]
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INFERENCE_FAILED,
                message=f"could not infer required role {role!r} above threshold {threshold}",
                hint=f"Pass roles={{{role!r}: ...}} explicitly on next call.",
                details={
                    "role": role,
                    "threshold": threshold,
                    "candidates": top3,
                },
            )
        )
    return candidates[0].column


def _pick_optional(
    candidates: list[RoleCandidate], threshold: float
) -> str | None:
    if not candidates or candidates[0].score < threshold:
        return None
    return candidates[0].column


def infer_roles(
    schema: Schema,
    *,
    sample_rows: Sequence[Mapping[str, Any]] | None = None,
) -> RoleMap:
    """Infer column roles from schema + an optional sample.

    Required roles (``content_column``, ``id_column``) raise
    ``ToolErrorException(INFERENCE_FAILED)`` with the top-3 candidates if
    no column scores above their threshold.

    Optional roles (``order_column``, ``partition_column``, ``type_column``,
    ``date_column``) are set to ``None`` when no candidate clears the bar.
    """
    stats = _compute_stats(schema, sample_rows)

    content_candidates = _score_all(schema, stats, _score_content)
    id_candidates = _score_all(schema, stats, _score_id)
    order_candidates = _score_all(schema, stats, _score_order)
    partition_candidates = _score_all(schema, stats, _score_partition)
    type_candidates = _score_all(schema, stats, _score_type)
    date_candidates = _score_all(schema, stats, _score_date)

    content = _pick_required("content_column", content_candidates, 0.7)
    id_ = _pick_required("id_column", id_candidates, 0.6)

    return RoleMap(
        id_column=id_,
        content_column=content,
        order_column=_pick_optional(order_candidates, 0.6),
        partition_column=_pick_optional(partition_candidates, 0.6),
        type_column=_pick_optional(type_candidates, 0.6),
        date_column=_pick_optional(date_candidates, 0.5),
    )
