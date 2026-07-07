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
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from databricks_deep_research.tools.api import _DecoratedTool, tool
from databricks_deep_research.tools.catalog_types import CatalogCard, SafeProbe
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)


class DecoratedToolFactory:
    """Creates :class:`_DecoratedTool` instances from ``kind: decorated`` YAML.

    TRUST BOUNDARY: importing a module executes its top-level code, so a
    ``decorated`` declaration is arbitrary code execution at tool-resolution
    time. ``allowed_import_prefixes`` gates which module paths may be
    imported:

    * ``None`` — historical allow-all. ONLY for hosts whose workflow YAML is
      authored at import time by the application developer (never database
      rows / user-supplied input). Must be opted into explicitly.
    * a sequence (possibly empty) — ``module`` must equal a prefix or live
      under ``prefix.``; empty means deny-all. The framework's DEFAULT
      executor factory chain uses deny-all, because default-chain hosts may
      feed it stored (database-sourced) workflow definitions.
    """

    SUPPORTED_KIND = "decorated"
    catalog_cards: ClassVar[Mapping[str, CatalogCard]] = {}
    safe_probes: ClassVar[Mapping[str, SafeProbe | None]] = {}

    def __init__(
        self, *, allowed_import_prefixes: Sequence[str] | None = None
    ) -> None:
        self._allowed_import_prefixes: tuple[str, ...] | None = (
            None
            if allowed_import_prefixes is None
            else tuple(p.rstrip(".") for p in allowed_import_prefixes if p)
        )

    def supports(self, kind: str) -> bool:
        return kind == self.SUPPORTED_KIND

    def _import_allowed(self, module_path: str) -> bool:
        if self._allowed_import_prefixes is None:
            return True
        return any(
            module_path == prefix or module_path.startswith(prefix + ".")
            for prefix in self._allowed_import_prefixes
        )

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

        if not self._import_allowed(module_path):
            allowed = (
                list(self._allowed_import_prefixes)
                if self._allowed_import_prefixes
                else "none"
            )
            raise ValueError(
                f"Decorated tool {decl.name!r}: import of {module_path!r} is not "
                f"allowed on this host (allowed prefixes: {allowed}). Importing "
                "a module executes code; stored workflow definitions must use "
                "'registered' tools or a host-configured allowlist instead."
            )

        try:
            # Gate above enforces the trust boundary; None (allow-all) is an
            # explicit host opt-in for import-time-authored YAML only.
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
