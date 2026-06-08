"""Cache-backed ``IDataSourceService`` — stores data source metadata via ``StorageStack``.

Metadata rows live in the ``user_data_sources`` cold-path list table.
OBO validation is delegated to ``OBODatabricksClient`` (identical to the
legacy service) because it talks to Databricks APIs, not the storage layer.

Return shape: ``_DataSourceView`` dataclass mirrors the ``UserDataSource``
ORM attribute surface so all call sites work without modification.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.models.data_source import (
    DataSourceType,
    DataSourceValidationStatus,
    DataSourceVisibility,
)
from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IDataSourceService

if TYPE_CHECKING:
    from deep_research.services.obo_client import OBODatabricksClient
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)

_DS_TABLE = "user_data_sources"


# ---------------------------------------------------------------------------
# View object (legacy-compatible DTO)
# ---------------------------------------------------------------------------


class _DataSourceView:
    """Lightweight DTO mirroring ``UserDataSource`` ORM attribute surface."""

    __slots__ = (
        "id", "owner_id", "type", "name", "description",
        "endpoint_identifier", "config", "visibility",
        "validation_status", "last_validated_at",
        "created_at", "updated_at",
    )

    def __init__(
        self,
        id: UUID,
        owner_id: str,
        type: str,
        name: str,
        endpoint_identifier: str,
        config: dict[str, Any],
        visibility: str = DataSourceVisibility.PRIVATE.value,
        validation_status: str = DataSourceValidationStatus.PENDING.value,
        description: str | None = None,
        last_validated_at: datetime | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.id = id
        self.owner_id = owner_id
        self.type = type
        self.name = name
        self.description = description
        self.endpoint_identifier = endpoint_identifier
        self.config: dict[str, Any] = config
        self.visibility = visibility
        self.validation_status = validation_status
        self.last_validated_at = last_validated_at
        now = datetime.now(UTC)
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    # ORM-compatible properties
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType(self.type)

    @property
    def visibility_level(self) -> DataSourceVisibility:
        return DataSourceVisibility(self.visibility)

    @property
    def status(self) -> DataSourceValidationStatus:
        return DataSourceValidationStatus(self.validation_status)

    @property
    def is_valid(self) -> bool:
        return self.validation_status == DataSourceValidationStatus.VALID.value

    @property
    def is_workspace_visible(self) -> bool:
        return self.visibility == DataSourceVisibility.WORKSPACE.value

    def mark_valid(self) -> None:
        self.validation_status = DataSourceValidationStatus.VALID.value
        self.last_validated_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def mark_invalid(self) -> None:
        self.validation_status = DataSourceValidationStatus.INVALID.value
        self.last_validated_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def mark_expired(self) -> None:
        self.validation_status = DataSourceValidationStatus.EXPIRED.value
        self.updated_at = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Row serialisation helpers
# ---------------------------------------------------------------------------


def _row_to_view(row: dict[str, Any]) -> _DataSourceView:
    raw_id = row["id"]
    ds_id = raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
    raw_lv = row.get("last_validated_at")
    last_validated_at: datetime | None = None
    if raw_lv:
        last_validated_at = (
            raw_lv if isinstance(raw_lv, datetime)
            else datetime.fromisoformat(str(raw_lv))
        )
    raw_created = row.get("created_at")
    created_at: datetime | None = None
    if raw_created:
        created_at = (
            raw_created if isinstance(raw_created, datetime)
            else datetime.fromisoformat(str(raw_created))
        )
    raw_updated = row.get("updated_at")
    updated_at: datetime | None = None
    if raw_updated:
        updated_at = (
            raw_updated if isinstance(raw_updated, datetime)
            else datetime.fromisoformat(str(raw_updated))
        )
    return _DataSourceView(
        id=ds_id,
        owner_id=str(row["owner_id"]),
        type=str(row.get("type", DataSourceType.VECTOR_SEARCH.value)),
        name=str(row.get("name", "")),
        description=row.get("description"),
        endpoint_identifier=str(row.get("endpoint_identifier", "")),
        config=row.get("config") or {},
        visibility=str(row.get("visibility", DataSourceVisibility.PRIVATE.value)),
        validation_status=str(row.get("validation_status", DataSourceValidationStatus.PENDING.value)),
        last_validated_at=last_validated_at,
        created_at=created_at,
        updated_at=updated_at,
    )


def _view_to_row(v: _DataSourceView) -> dict[str, Any]:
    return {
        "id": str(v.id),
        "owner_id": v.owner_id,
        "type": v.type,
        "name": v.name,
        "description": v.description,
        "endpoint_identifier": v.endpoint_identifier,
        "config": v.config,
        "visibility": v.visibility,
        "validation_status": v.validation_status,
        "last_validated_at": v.last_validated_at.isoformat() if v.last_validated_at else None,
        "created_at": v.created_at.isoformat() if v.created_at else None,
        "updated_at": v.updated_at.isoformat() if v.updated_at else None,
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CachedDataSourceService(_CachedServiceBase, IDataSourceService):
    """``IDataSourceService`` backed by ``StorageStack`` cold-path list tables."""

    _service_name = "data_source"

    def __init__(
        self,
        stack: StorageStack,
        obo_client: OBODatabricksClient | None = None,
    ) -> None:
        super().__init__(stack)
        if obo_client is None:
            from deep_research.services.obo_client import OBODatabricksClient
            obo_client = OBODatabricksClient()
        self._obo_client = obo_client

    # -- Create methods -------------------------------------------------------

    async def create_vector_search_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        endpoint_name: str,
        index_name: str,
        description: str | None = None,
        visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE,
        enable_hybrid: bool = True,
        enable_reranking: bool = True,
        num_results: int = 10,
    ) -> tuple[_DataSourceView | None, str | None]:
        has_access, error = await self._obo_client.validate_vector_search_access(
            user_token, endpoint_name, index_name
        )
        if not has_access:
            return None, error

        schema = await self._obo_client.get_vector_search_index_schema(
            user_token, endpoint_name, index_name
        )
        columns = [col["name"] for col in (schema or {}).get("columns", [])]
        text_columns = (schema or {}).get("text_columns", [])

        config = {
            "endpoint_name": endpoint_name,
            "index_name": index_name,
            "columns": columns,
            "columns_to_rerank": text_columns[:2] if text_columns else [],
            "enable_hybrid": enable_hybrid,
            "enable_reranking": enable_reranking,
            "num_results": num_results,
        }
        if isinstance(visibility, DataSourceVisibility):
            vis_value = visibility.value
        else:
            vis_value = str(visibility)

        view = _DataSourceView(
            id=uuid4(),
            owner_id=owner_id,
            type=DataSourceType.VECTOR_SEARCH.value,
            name=name,
            description=description,
            endpoint_identifier=index_name,
            config=config,
            visibility=vis_value,
            validation_status=DataSourceValidationStatus.VALID.value,
            last_validated_at=datetime.now(UTC),
        )
        await self._cold_upsert_row(_DS_TABLE, _view_to_row(view), pk="id")
        logger.info("Created VS source id=%s owner=%s", view.id, owner_id)
        return view, None

    async def create_genie_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        space_id: str,
        description: str | None = None,
        example_questions: list[str] | None = None,
        visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE,
    ) -> tuple[_DataSourceView | None, str | None]:
        has_access, error = await self._obo_client.validate_genie_access(
            user_token, space_id
        )
        if not has_access:
            return None, error

        config = {
            "space_id": space_id,
            "example_questions": example_questions or [],
        }
        if isinstance(visibility, DataSourceVisibility):
            vis_value = visibility.value
        else:
            vis_value = str(visibility)

        view = _DataSourceView(
            id=uuid4(),
            owner_id=owner_id,
            type=DataSourceType.GENIE.value,
            name=name,
            description=description,
            endpoint_identifier=space_id,
            config=config,
            visibility=vis_value,
            validation_status=DataSourceValidationStatus.VALID.value,
            last_validated_at=datetime.now(UTC),
        )
        await self._cold_upsert_row(_DS_TABLE, _view_to_row(view), pk="id")
        logger.info("Created Genie source id=%s owner=%s", view.id, owner_id)
        return view, None

    async def create_assistant_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        endpoint_name: str,
        description: str | None = None,
        pass_context: bool = True,
        visibility: DataSourceVisibility = DataSourceVisibility.PRIVATE,
    ) -> tuple[_DataSourceView | None, str | None]:
        has_access, error = await self._obo_client.validate_assistant_access(
            user_token, endpoint_name
        )
        if not has_access:
            return None, error

        config = {
            "endpoint_name": endpoint_name,
            "pass_context": pass_context,
        }
        if isinstance(visibility, DataSourceVisibility):
            vis_value = visibility.value
        else:
            vis_value = str(visibility)

        view = _DataSourceView(
            id=uuid4(),
            owner_id=owner_id,
            type=DataSourceType.KNOWLEDGE_ASSISTANT.value,
            name=name,
            description=description,
            endpoint_identifier=endpoint_name,
            config=config,
            visibility=vis_value,
            validation_status=DataSourceValidationStatus.VALID.value,
            last_validated_at=datetime.now(UTC),
        )
        await self._cold_upsert_row(_DS_TABLE, _view_to_row(view), pk="id")
        logger.info("Created KA source id=%s owner=%s", view.id, owner_id)
        return view, None

    # -- Query methods --------------------------------------------------------

    async def get_accessible_sources(
        self,
        user_id: str,
        source_type: DataSourceType | None = None,
        only_valid: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[_DataSourceView], int]:
        all_rows = await self._cold_list_rows(_DS_TABLE)
        views = [_row_to_view(r) for r in all_rows]

        # Access filter: owned OR (workspace-visible AND valid)
        def _is_accessible(v: _DataSourceView) -> bool:
            if v.owner_id == user_id:
                return not only_valid or v.is_valid
            return (
                v.visibility == DataSourceVisibility.WORKSPACE.value
                and v.is_valid
            )

        filtered = [v for v in views if _is_accessible(v)]

        if source_type is not None:
            type_val = source_type.value if isinstance(source_type, DataSourceType) else str(source_type)
            filtered = [v for v in filtered if v.type == type_val]

        # Sort by name for determinism
        filtered.sort(key=lambda v: v.name)

        total = len(filtered)
        page = filtered[offset: offset + limit]
        return page, total

    async def get_for_user(
        self, source_id: UUID, user_id: str
    ) -> _DataSourceView | None:
        rows = await self._cold_list_rows(
            _DS_TABLE, {"id": str(source_id), "owner_id": user_id}
        )
        if not rows:
            return None
        return _row_to_view(rows[0])

    async def get_accessible(
        self, source_id: UUID, user_id: str
    ) -> _DataSourceView | None:
        rows = await self._cold_list_rows(_DS_TABLE, {"id": str(source_id)})
        if not rows:
            return None
        v = _row_to_view(rows[0])
        if v.owner_id == user_id:
            return v
        if v.is_workspace_visible and v.is_valid:
            return v
        return None

    # -- Validation -----------------------------------------------------------

    async def revalidate_source(
        self,
        source: _DataSourceView,
        user_token: str,
    ) -> tuple[bool, str | None]:
        source_type = DataSourceType(source.type)
        config = source.config

        if source_type == DataSourceType.VECTOR_SEARCH:
            has_access, error = await self._obo_client.validate_vector_search_access(
                user_token,
                config.get("endpoint_name", ""),
                config.get("index_name", source.endpoint_identifier),
            )
        elif source_type == DataSourceType.GENIE:
            has_access, error = await self._obo_client.validate_genie_access(
                user_token, source.endpoint_identifier
            )
        elif source_type == DataSourceType.KNOWLEDGE_ASSISTANT:
            has_access, error = await self._obo_client.validate_assistant_access(
                user_token, source.endpoint_identifier
            )
        else:
            has_access, error = True, None

        if has_access:
            source.mark_valid()
        else:
            source.mark_invalid()

        await self._cold_upsert_row(_DS_TABLE, _view_to_row(source), pk="id")
        return has_access, error

    async def mark_sources_expired(self, owner_id: str) -> int:
        rows = await self._cold_list_rows(_DS_TABLE, {"owner_id": owner_id})
        count = 0
        for row in rows:
            v = _row_to_view(row)
            if v.validation_status == DataSourceValidationStatus.VALID.value:
                v.mark_expired()
                await self._cold_upsert_row(_DS_TABLE, _view_to_row(v), pk="id")
                count += 1
        return count

    # -- Mutations ------------------------------------------------------------

    async def update(self, source: _DataSourceView) -> _DataSourceView:
        source.updated_at = datetime.now(UTC)
        await self._cold_upsert_row(_DS_TABLE, _view_to_row(source), pk="id")
        return source

    async def delete(self, source: _DataSourceView) -> None:
        await self._cold_delete_row(_DS_TABLE, str(source.id), pk="id")
