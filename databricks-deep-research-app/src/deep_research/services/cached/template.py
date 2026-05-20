"""Cache-backed ``ITemplateService`` — routes prompt-template CRUD through ``StorageStack``.

Template records live in the ``prompt_templates`` list table (cold-path
``_cold_list_rows`` / ``_cold_upsert_row`` / ``_cold_delete_row``).

DDL columns (both backends):
    template_id   TEXT/UUID PK
    owner_id      TEXT
    name          TEXT
    content       TEXT
    visibility    TEXT  ('private' | 'workspace')
    template_type TEXT
    metadata      JSONB/TEXT  — stores {is_default, description, variables, tags}
    created_at    TIMESTAMP
    updated_at    TIMESTAMP

Return shape: ``TemplateView`` dataclass that exposes the same attribute names
as the legacy ``PromptTemplate`` ORM object so all call-site code in
``api/v1/templates.py`` continues to work without modification.

Visibility semantics (mirrors legacy):
- ``get_for_user``          — owned by user_id only
- ``get_accessible``        — owned by user OR workspace-visible
- ``get_accessible_templates`` — same, with optional type/tags filter + pagination
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import ITemplateService

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)

_TABLE = "prompt_templates"
_PK = "template_id"

_WORKSPACE_VIS = "workspace"


# ---------------------------------------------------------------------------
# View object (legacy-compatible DTO)
# ---------------------------------------------------------------------------


@dataclass
class TemplateView:
    """Read-only DTO mirroring the legacy ``PromptTemplate`` ORM attribute surface."""

    id: UUID
    owner_id: str
    name: str
    type: str          # template_type column maps to .type on ORM
    content: str
    description: str | None
    variables: list[dict[str, Any]]
    tags: list[str]
    visibility: str
    is_default: bool
    created_at: datetime
    updated_at: datetime

    # Legacy ORM compat properties referenced by api/v1/templates.py
    def get_required_variables(self) -> list[str]:
        return [
            v["name"]
            for v in (self.variables or [])
            if v.get("required", True) and v.get("name")
        ]


# ---------------------------------------------------------------------------
# Row serialisation helpers
# ---------------------------------------------------------------------------


def _uuid(v: Any) -> UUID:
    if isinstance(v, UUID):
        return v
    return UUID(str(v))


def _dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v))


def _decode_metadata(raw: Any) -> dict[str, Any]:
    """Decode the ``metadata`` column — JSON string (Warehouse) or dict (Lakebase)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _encode_metadata(meta: dict[str, Any]) -> str:
    return json.dumps(meta, default=str)


def _row_to_view(row: dict[str, Any]) -> TemplateView:
    meta = _decode_metadata(row.get("metadata", {}))
    return TemplateView(
        id=_uuid(row[_PK]),
        owner_id=str(row["owner_id"]),
        name=str(row["name"]),
        type=str(row.get("template_type", "default")),
        content=str(row.get("content", "")),
        description=meta.get("description"),
        variables=meta.get("variables") or [],
        tags=meta.get("tags") or [],
        visibility=str(row.get("visibility", "private")),
        is_default=bool(meta.get("is_default", False)),
        created_at=_dt(row.get("created_at", datetime.now(UTC))),
        updated_at=_dt(row.get("updated_at", datetime.now(UTC))),
    )


def _view_to_row(view: TemplateView) -> dict[str, Any]:
    meta = _encode_metadata(
        {
            "is_default": view.is_default,
            "description": view.description,
            "variables": view.variables,
            "tags": view.tags,
        }
    )
    return {
        _PK: str(view.id),
        "owner_id": view.owner_id,
        "name": view.name,
        "template_type": view.type,
        "content": view.content,
        "visibility": view.visibility,
        "metadata": meta,
        "created_at": view.created_at.isoformat(),
        "updated_at": view.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CachedTemplateService(_CachedServiceBase, ITemplateService):
    """``ITemplateService`` backed by ``StorageStack`` cold-path list tables."""

    _service_name = "template"

    def __init__(self, stack: StorageStack) -> None:
        super().__init__(stack)

    # -- Reads ---------------------------------------------------------------

    async def get_for_user(
        self,
        template_id: UUID,
        user_id: str,
    ) -> TemplateView | None:
        rows = await self._cold_list_rows(
            _TABLE, {_PK: str(template_id), "owner_id": user_id}
        )
        if not rows:
            return None
        return _row_to_view(rows[0])

    async def get_accessible(
        self,
        template_id: UUID,
        user_id: str,
    ) -> TemplateView | None:
        rows = await self._cold_list_rows(_TABLE, {_PK: str(template_id)})
        if not rows:
            return None
        row = rows[0]
        owner = row.get("owner_id")
        vis = row.get("visibility", "private")
        if owner == user_id or vis == _WORKSPACE_VIS:
            return _row_to_view(row)
        return None

    async def get_by_name(
        self,
        owner_id: str,
        name: str,
    ) -> TemplateView | None:
        rows = await self._cold_list_rows(_TABLE, {"owner_id": owner_id, "name": name})
        if not rows:
            return None
        return _row_to_view(rows[0])

    async def get_accessible_templates(
        self,
        user_id: str,
        template_type: Any | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TemplateView], int]:
        """Return (views, total) — own + workspace templates with optional filters."""
        all_rows = await self._cold_list_rows(_TABLE)

        # Filter: accessible = owned OR workspace
        accessible: list[TemplateView] = []
        for row in all_rows:
            owner = row.get("owner_id")
            vis = row.get("visibility", "private")
            if owner != user_id and vis != _WORKSPACE_VIS:
                continue
            view = _row_to_view(row)
            # Optional type filter
            if template_type is not None:
                type_val = template_type.value if hasattr(template_type, "value") else str(template_type)
                if view.type != type_val:
                    continue
            # Optional tags filter (any match)
            if tags and not any(t in view.tags for t in tags):
                continue
            accessible.append(view)

        # Sort by name (mirrors legacy ORM query)
        accessible.sort(key=lambda v: v.name)
        total = len(accessible)
        page = accessible[offset: offset + limit]
        return page, total

    async def search_by_tags(
        self,
        user_id: str,
        tags: list[str],
        limit: int = 50,
    ) -> list[TemplateView]:
        views, _ = await self.get_accessible_templates(
            user_id=user_id, tags=tags, limit=limit
        )
        return views

    async def get_default_template(
        self,
        owner_id: str,
        template_type: Any,
    ) -> TemplateView | None:
        """Return user's own default first, then workspace default."""
        type_val = template_type.value if hasattr(template_type, "value") else str(template_type)
        all_rows = await self._cold_list_rows(_TABLE)

        # Own defaults first
        for row in all_rows:
            view = _row_to_view(row)
            if (
                row.get("owner_id") == owner_id
                and view.type == type_val
                and view.is_default
            ):
                return view

        # Workspace defaults, ordered by creation time (oldest first)
        workspace_defaults = []
        for row in all_rows:
            view = _row_to_view(row)
            if (
                row.get("visibility") == _WORKSPACE_VIS
                and view.type == type_val
                and view.is_default
            ):
                workspace_defaults.append(view)

        if workspace_defaults:
            workspace_defaults.sort(key=lambda v: v.created_at)
            return workspace_defaults[0]
        return None

    # -- Writes --------------------------------------------------------------

    async def create_template(
        self,
        owner_id: str,
        name: str,
        template_type: Any,
        content: str,
        description: str | None = None,
        variables: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        visibility: Any = "private",
        is_default: bool = False,
    ) -> TemplateView:
        type_val = template_type.value if hasattr(template_type, "value") else str(template_type)
        vis_val = visibility.value if hasattr(visibility, "value") else str(visibility)

        # Auto-extract variables from {{placeholder}} if not provided
        if variables is None:
            import re
            VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")
            names = sorted(set(VARIABLE_PATTERN.findall(content)))
            variables = [
                {"name": n, "type": "string", "required": True, "default": None, "description": None}
                for n in names
            ]

        # If setting as default, unset other defaults for this type/owner
        if is_default:
            await self._unset_defaults(owner_id, type_val)

        now = datetime.now(UTC)
        view = TemplateView(
            id=uuid4(),
            owner_id=owner_id,
            name=name,
            type=type_val,
            content=content,
            description=description,
            variables=variables,
            tags=tags or [],
            visibility=vis_val,
            is_default=is_default,
            created_at=now,
            updated_at=now,
        )
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        logger.info("Created template %s for owner %s", view.id, owner_id)
        return view

    async def update_template(self, template: Any) -> Any:
        """Persist changes from a TemplateView-like object."""
        if isinstance(template, TemplateView):
            template.updated_at = datetime.now(UTC)
            await self._cold_upsert_row(_TABLE, _view_to_row(template), pk=_PK)
            return template

        # ORM compat: translate legacy PromptTemplate ORM object
        tid = _uuid(template.id)
        rows = await self._cold_list_rows(_TABLE, {_PK: str(tid)})
        if not rows:
            return template
        view = _row_to_view(rows[0])
        for attr in ("name", "content", "description", "visibility"):
            val = getattr(template, attr, None)
            if val is not None:
                setattr(view, attr, val)
        view.is_default = getattr(template, "is_default", view.is_default)
        tags = getattr(template, "tags", None)
        if tags is not None:
            view.tags = list(tags)
        variables = getattr(template, "variables", None)
        if variables is not None:
            view.variables = list(variables)
        view.updated_at = datetime.now(UTC)
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        return view

    # Alias used by api/v1/templates.py (calls svc.update(template))
    async def update(self, template: Any) -> Any:
        return await self.update_template(template)

    async def delete_template(self, template: Any) -> None:
        tid = str(_uuid(template.id))
        await self._cold_delete_row(_TABLE, tid, pk=_PK)
        logger.info("Deleted template %s", tid)

    # Alias used by api/v1/templates.py (calls svc.delete(template))
    async def delete(self, template: Any) -> None:
        await self.delete_template(template)

    async def set_default_template(
        self,
        template_id: UUID,
        owner_id: str,
    ) -> TemplateView | None:
        view = await self.get_for_user(template_id, owner_id)
        if view is None:
            return None
        await self.set_as_default(template_id=view.id, owner_id=owner_id, type_=view.type)
        # Reload so caller sees fresh state.
        return await self.get_for_user(template_id, owner_id)

    async def set_as_default(
        self,
        template_id: UUID,
        owner_id: str,
        type_: Any,
    ) -> None:
        """Flip the default flag for ``(owner_id, type)`` to point at
        ``template_id`` and unset all others.

        Cached path: there is no atomic SQL transaction, but the cold
        upserts run sequentially against a single backend connection
        within one process. Worst case is one stale read between unset
        and set; downstream reads on the same process see the final
        state because ``_cold_upsert_row`` invalidates the cold cache.
        """
        type_val = type_.value if hasattr(type_, "value") else str(type_)
        rows = await self._cold_list_rows(_TABLE, {"owner_id": owner_id})
        now = datetime.now(UTC)
        for row in rows:
            view = _row_to_view(row)
            if view.type != type_val:
                continue
            should_be_default = view.id == template_id
            if view.is_default == should_be_default:
                continue
            view.is_default = should_be_default
            view.updated_at = now
            await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)

    # Sync helpers used by api/v1/templates.py directly on the service
    def render_template(
        self,
        template: Any,
        variables: dict[str, Any],
    ) -> tuple[str, list[str], list[str]]:
        """Delegate to SafeTemplateRenderer (identical to legacy impl)."""
        from deep_research.services.template_renderer import SafeTemplateRenderer

        renderer = SafeTemplateRenderer()
        content = getattr(template, "content", "")
        var_list = getattr(template, "variables", []) or []
        missing_variables: list[str] = []
        used_defaults: list[str] = []
        var_metadata: dict[str, dict[str, Any]] = {
            v.get("name", ""): v for v in var_list if v.get("name")
        }
        context: dict[str, Any] = {}
        content_variables = renderer.extract_variables(content)
        for var_name in content_variables:
            if var_name in variables:
                context[var_name] = variables[var_name]
            else:
                metadata = var_metadata.get(var_name, {})
                default_value = metadata.get("default")
                is_required = metadata.get("required", True)
                if default_value is not None:
                    context[var_name] = default_value
                    used_defaults.append(var_name)
                elif is_required:
                    missing_variables.append(var_name)
        rendered = renderer.render(content, context)
        return rendered, missing_variables, used_defaults

    def validate_variables(
        self,
        template: Any,
        variables: dict[str, Any],
    ) -> list[str]:
        required = getattr(template, "get_required_variables", lambda: [])()
        return [name for name in required if name not in variables]

    # -- Internal ------------------------------------------------------------

    async def _unset_defaults(self, owner_id: str, type_val: str) -> None:
        """Unset is_default on all templates of given type for owner."""
        rows = await self._cold_list_rows(_TABLE, {"owner_id": owner_id})
        for row in rows:
            view = _row_to_view(row)
            if view.type == type_val and view.is_default:
                view.is_default = False
                view.updated_at = datetime.now(UTC)
                await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
