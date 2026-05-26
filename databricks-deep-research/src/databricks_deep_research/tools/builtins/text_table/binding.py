from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class BindingSource(StrEnum):
    BOUND = "bound"
    DISCOVERED = "discovered"


@dataclass(frozen=True)
class RoleMap:
    id_column: str
    content_column: str
    order_column: str | None = None
    partition_column: str | None = None
    label_column: str | None = None
    type_column: str | None = None
    date_column: str | None = None


@dataclass(frozen=True)
class BindingInfo:
    name: str
    fqn: str
    source: BindingSource
    description: str | None = None
    roles: RoleMap | None = None
    numeric_columns: tuple[str, ...] = ()
    structured_passages: Mapping[str, str] = field(default_factory=dict)
    verbose: bool = False
