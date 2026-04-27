"""Thin Python wrappers around the framework's pool config dataclasses.

These wrappers provide an ergonomic Python API for configuring an agent's
pool reads / writes. They serialize 1:1 to ``PoolWriteConfig`` and
``PoolInjectConfig`` (``agents/config.py``) at compile time, preserving the
YAML round-trip guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PoolWriteSpec:
    """Describes how an :class:`Agent` writes items to a shared pool.

    Maps 1:1 to :class:`PoolWriteConfig` in the IR.
    """

    pool: str
    extract: str
    transform: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        d: dict[str, str | None] = {"pool": self.pool, "extract": self.extract}
        if self.transform is not None:
            d["transform"] = self.transform
        return d


@dataclass
class PoolInjectSpec:
    """Describes how pool contents are injected into an agent prompt.

    Maps 1:1 to :class:`PoolInjectConfig` in the IR.
    """

    pool: str
    threshold: float = 0.0
    format: str = "text"  # "text", "json", "markdown"
    max_items: int = 20
    max_item_chars: int = 0

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "pool": self.pool,
            "threshold": self.threshold,
            "format": self.format,
            "max_items": self.max_items,
            "max_item_chars": self.max_item_chars,
        }


__all__ = ["PoolWriteSpec", "PoolInjectSpec"]
