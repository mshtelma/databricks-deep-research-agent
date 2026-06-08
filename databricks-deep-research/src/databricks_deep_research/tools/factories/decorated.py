"""Decorated tool factory — resolves YAML ``kind: decorated`` declarations.

Allows YAML workflows to reference Python ``@tool``-decorated callables by
import path::

    tools:
      - name: weather
        kind: decorated
        config:
          import: myapp.tools:weather   # module:attr

The factory imports the module, looks up the attribute, and returns the
existing :class:`_DecoratedTool` (which already conforms to
:class:`ResearchTool`). The attribute may also be a plain callable — in
that case the factory wraps it via :func:`tool` automatically using any
``description``, ``inject``, etc. supplied in the declaration's ``config``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any, ClassVar

from databricks_deep_research.tools.api import _DecoratedTool, tool
from databricks_deep_research.tools.catalog_types import CatalogCard, SafeProbe
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)


class DecoratedToolFactory:
    """Creates :class:`_DecoratedTool` instances from ``kind: decorated`` YAML."""

    SUPPORTED_KIND = "decorated"
    catalog_cards: ClassVar[Mapping[str, CatalogCard]] = {}
    safe_probes: ClassVar[Mapping[str, SafeProbe | None]] = {}

    def supports(self, kind: str) -> bool:
        return kind == self.SUPPORTED_KIND

    async def create(
        self,
        decl: ToolDeclaration,
        ctx: ToolFactoryContext,  # noqa: ARG002 — required by protocol
    ) -> ResearchTool:
        target = decl.config.get("import")
        if not target or not isinstance(target, str):
            raise ValueError(
                f"Decorated tool {decl.name!r} requires config.import "
                f"as a 'module:attr' string; got {target!r}"
            )

        module_path, _, attr = target.partition(":")
        if not module_path or not attr:
            raise ValueError(
                f"Decorated tool {decl.name!r} import must be 'module:attr'; "
                f"got {target!r}"
            )

        try:
            # TRUST BOUNDARY: module_path originates from YAML workflow definitions
            # authored at import-time by framework users (e.g. @tool-decorated callables
            # in application code), NOT from database rows or user-supplied HTTP input.
            # The factory is only instantiated during workflow loading — never at request
            # time from untrusted caller data — so importlib.import_module is safe here.
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"Decorated tool {decl.name!r}: failed to import {module_path!r}: {exc}"
            ) from exc

        try:
            value: Any = getattr(module, attr)
        except AttributeError as exc:
            raise ValueError(
                f"Decorated tool {decl.name!r}: module {module_path!r} has no attribute {attr!r}"
            ) from exc

        if isinstance(value, _DecoratedTool):
            # Optional name override from YAML.
            if decl.name and value.definition.name != decl.name:
                value._definition = value.definition.__class__(
                    name=decl.name,
                    description=value.definition.description,
                    parameters=value.definition.parameters,
                    source_type=value.definition.source_type,
                    source_kind=value.definition.source_kind,
                    metadata=value.definition.metadata,
                )
            return value

        if callable(value):
            kwargs: dict[str, Any] = {}
            if "description" in decl.config:
                kwargs["description"] = decl.config["description"]
            if decl.description and "description" not in kwargs:
                kwargs["description"] = decl.description
            if "inject" in decl.config:
                kwargs["inject"] = decl.config["inject"]
            if decl.config.get("requires_confirmation"):
                kwargs["requires_confirmation"] = True
            kwargs["name"] = decl.name
            return tool(value, **kwargs)

        raise ValueError(
            f"Decorated tool {decl.name!r}: import target {target!r} is neither "
            f"a @tool-decorated callable nor a plain function (got {type(value).__name__})"
        )


__all__ = ["DecoratedToolFactory"]
