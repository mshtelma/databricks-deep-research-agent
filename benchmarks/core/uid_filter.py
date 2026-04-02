"""UID substring filtering for benchmark question/result selection."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def parse_uid_fragments(raw: str) -> list[str]:
    """Split comma-separated UID fragments, stripping whitespace.

    >>> parse_uid_fragments("0029,0030, 0057")
    ['0029', '0030', '0057']
    """
    return [f.strip() for f in raw.split(",") if f.strip()]


def filter_by_uid_fragments(
    items: list[T],
    fragments: list[str],
    uid_getter: Callable[[T], str],
) -> list[T]:
    """Keep items whose UID contains any fragment (substring match).

    Warns about fragments that matched nothing.
    """
    matched: list[T] = []
    hit_fragments: set[str] = set()

    for item in items:
        uid = uid_getter(item)
        for frag in fragments:
            if frag in uid:
                matched.append(item)
                hit_fragments.add(frag)
                break  # item matched — no need to check remaining fragments

    unmatched = [f for f in fragments if f not in hit_fragments]
    if unmatched:
        logger.warning(
            "UID_FILTER_NO_MATCH fragments=%s (matched %d of %d items)",
            ",".join(unmatched),
            len(matched),
            len(items),
        )

    return matched
