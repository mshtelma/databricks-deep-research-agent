"""Structured-passage parsers for ``BindingInfo.structured_passages``.

The ``structured_passages`` map on a binding declares per-row content of a
particular ``type_value`` (e.g., the ``chunk_type`` cell in OfficeQA) as one
of a fixed set of structured payloads: ``html``, ``markdown``, or ``json``.
Each parser returns a uniform ``StructuredPassage`` shape so downstream
tools can treat them homogeneously.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from .html import parse_html
from .json import parse_json
from .markdown import parse_markdown

ParserName = Literal["html", "markdown", "json"]


class StructuredPassage(TypedDict):
    raw: str
    parsed: Any
    parser: ParserName


def _wrap_html(content: str) -> StructuredPassage:
    return _to_structured(parse_html(content))


def _wrap_markdown(content: str) -> StructuredPassage:
    return _to_structured(parse_markdown(content))


def _wrap_json(content: str) -> StructuredPassage:
    return _to_structured(parse_json(content))


def _to_structured(d: dict[str, Any]) -> StructuredPassage:
    return StructuredPassage(
        raw=d["raw"],
        parsed=d["parsed"],
        parser=d["parser"],
    )


_REGISTRY: dict[ParserName, Callable[[str], StructuredPassage]] = {
    "html": _wrap_html,
    "markdown": _wrap_markdown,
    "json": _wrap_json,
}


def get_parser(name: ParserName) -> Callable[[str], StructuredPassage]:
    """Return the parser callable for ``name``.

    Raises ``ValueError`` for unknown parser names.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown parser {name!r}; valid: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = ["ParserName", "StructuredPassage", "get_parser"]
