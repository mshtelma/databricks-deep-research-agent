"""HTML structured-passage parser.

If a ``<table>`` element is present, returns ``parsed`` as a list of row
dicts keyed by header. Otherwise returns ``parsed`` as the flat extracted
text.

Implementation note: ``beautifulsoup4`` is not a hard dependency of the
framework package (it lives in the app's ``uv.lock`` only). To stay
dependency-free we use a regex-based extractor that handles the row /
header / cell shapes the framework actually emits via the
``html_table_parser`` upstream stage.
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_tags(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment)
    text = _html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _extract_table_rows(html: str) -> list[dict[str, str]] | None:
    """Return a list of row dicts when ``html`` contains a parseable table.

    Returns ``None`` if the input has no ``<table>`` element.
    """
    table_match = _TABLE_RE.search(html)
    if table_match is None:
        return None
    table_html = table_match.group(1)
    tr_matches = _ROW_RE.findall(table_html)
    if not tr_matches:
        return []
    headers: list[str] = []
    body_trs: list[str] = list(tr_matches)
    th_cells = _TH_RE.findall(tr_matches[0])
    if th_cells:
        headers = [_strip_tags(c) for c in th_cells]
        body_trs = list(tr_matches[1:])
    else:
        first_tds = _TD_RE.findall(tr_matches[0])
        headers = [f"col_{i}" for i in range(len(first_tds))]
    rows: list[dict[str, str]] = []
    for tr_html in body_trs:
        td_cells = _TD_RE.findall(tr_html) or _TH_RE.findall(tr_html)
        values = [_strip_tags(c) for c in td_cells]
        row: dict[str, str] = {}
        for i, val in enumerate(values):
            key = headers[i] if i < len(headers) else f"col_{i}"
            row[key] = val
        rows.append(row)
    return rows


def parse_html(content: str) -> dict[str, Any]:
    rows = _extract_table_rows(content)
    if rows is not None:
        parsed: Any = rows
    else:
        parsed = _strip_tags(content)
    return {"raw": content, "parsed": parsed, "parser": "html"}
