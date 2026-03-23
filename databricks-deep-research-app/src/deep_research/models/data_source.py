"""UserDataSource SQLAlchemy model for enterprise data source integration.

Allows users to configure their own data sources (Vector Search indexes,
Genie spaces, Knowledge Assistants) for use in research.

Part of 007-enterprise-data-sources feature (T010).
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel

if TYPE_CHECKING:
    pass


class DataSourceType(StrEnum):
    """Types of queryable data sources.

    Each type has different capabilities and query patterns:
    - VECTOR_SEARCH: Semantic similarity search with optional filtering
    - GENIE: Natural language to SQL for analytics
    - KNOWLEDGE_ASSISTANT: Domain expert Q&A
    - WEB_SEARCH: External web search (Brave) - system only
    - UPLOADED_FILE: User-uploaded documents - session only
    - CUSTOM: Plugin-provided custom sources
    """

    VECTOR_SEARCH = "vector_search"
    GENIE = "genie"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"
    WEB_SEARCH = "web_search"
    UPLOADED_FILE = "uploaded_file"
    CUSTOM = "custom"


class DataSourceVisibility(StrEnum):
    """Visibility levels for user-configured data sources."""

    PRIVATE = "private"  # Only creator can see/use
    WORKSPACE = "workspace"  # All workspace users (with OBO access)


class DataSourceValidationStatus(StrEnum):
    """Validation status for user data sources."""

    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"


class UserDataSource(BaseModel):
    """User-configured data source for research.

    Stores configuration for user-added data sources (Vector Search indexes,
    Genie spaces, Knowledge Assistants). Configuration is stored in the
    `config` JSONB column, with schema auto-detected on creation.

    Attributes:
        owner_id: Databricks workspace user ID who created this source.
        type: Type of data source (vector_search, genie, etc.).
        name: Display name for the source (unique per owner).
        description: Human-readable description.
        endpoint_identifier: Primary identifier (index name, space ID, etc.).
        config: Type-specific configuration (JSONB, schema varies by type).
        visibility: Who can see/use this source.
        validation_status: Whether OBO access is currently valid.
        last_validated_at: When OBO access was last validated.
    """

    __tablename__ = "user_data_sources"

    # Owner identification (Databricks workspace user ID)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Source type
    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Display name (unique per owner for clarity)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Description for UI and LLM
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Primary identifier (index name for VS, space ID for Genie, etc.)
    endpoint_identifier: Mapped[str] = mapped_column(String(500), nullable=False)

    # Type-specific configuration (schema varies by type)
    # Vector Search: endpoint_name, index_name, columns, columns_to_rerank, enable_hybrid, enable_reranking, num_results
    # Genie: space_id, example_questions
    # Knowledge Assistant: endpoint_name, pass_context
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # Visibility level
    visibility: Mapped[str] = mapped_column(
        String(20),
        default=DataSourceVisibility.PRIVATE.value,
        nullable=False,
    )

    # Validation status (OBO access check)
    validation_status: Mapped[str] = mapped_column(
        String(20),
        default=DataSourceValidationStatus.PENDING.value,
        nullable=False,
    )

    # When access was last validated
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Indexes
    __table_args__ = (
        # Fast lookup by owner
        Index("idx_user_data_sources_owner", "owner_id"),
        # Fast lookup by type
        Index("idx_user_data_sources_type", "type"),
        # Fast lookup for workspace-visible sources
        Index("idx_user_data_sources_visibility", "visibility"),
        # Composite for listing accessible sources
        Index("idx_user_data_sources_owner_visibility", "owner_id", "visibility"),
        # Unique name per owner to prevent confusion
        Index(
            "uq_user_data_sources_owner_name",
            "owner_id",
            "name",
            unique=True,
        ),
    )

    @property
    def source_type(self) -> DataSourceType:
        """Get type as enum."""
        return DataSourceType(self.type)

    @property
    def visibility_level(self) -> DataSourceVisibility:
        """Get visibility as enum."""
        return DataSourceVisibility(self.visibility)

    @property
    def status(self) -> DataSourceValidationStatus:
        """Get validation status as enum."""
        return DataSourceValidationStatus(self.validation_status)

    @property
    def is_valid(self) -> bool:
        """Check if source has valid OBO access."""
        return self.validation_status == DataSourceValidationStatus.VALID.value

    @property
    def is_workspace_visible(self) -> bool:
        """Check if source is visible to workspace users."""
        return self.visibility == DataSourceVisibility.WORKSPACE.value

    def mark_valid(self) -> None:
        """Mark source as having valid OBO access."""
        self.validation_status = DataSourceValidationStatus.VALID.value
        self.last_validated_at = datetime.now(UTC)

    def mark_invalid(self) -> None:
        """Mark source as having invalid OBO access."""
        self.validation_status = DataSourceValidationStatus.INVALID.value
        self.last_validated_at = datetime.now(UTC)

    def mark_expired(self) -> None:
        """Mark source validation as expired (needs revalidation)."""
        self.validation_status = DataSourceValidationStatus.EXPIRED.value
