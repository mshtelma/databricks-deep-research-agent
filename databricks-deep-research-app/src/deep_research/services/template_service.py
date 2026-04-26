"""TemplateService - CRUD operations for prompt templates.

Manages user-created prompt templates with variable validation,
rendering support, and default template management.

Part of US5 - Custom Prompt Template Library (T067).
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update

from deep_research.models.prompt_template import (
    PromptTemplate,
    TemplateType,
    TemplateVisibility,
)
from deep_research.services.base import BaseRepository
from deep_research.services.template_renderer import SafeTemplateRenderer

logger = logging.getLogger(__name__)

# Regex pattern for template variables: {{variable_name}}
VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")

# Safe template renderer instance (used for rendering user-provided templates)
_safe_renderer = SafeTemplateRenderer()


class TemplateService(BaseRepository[PromptTemplate]):
    """Service for managing prompt templates.

    Extends BaseRepository[PromptTemplate] for standard CRUD operations.
    Provides specialized methods for:
    - Creating templates with variable extraction
    - Rendering templates with variable substitution
    - Managing default templates
    - Listing accessible templates (own + workspace)
    """

    model = PromptTemplate

    # =========================================================================
    # Template Creation
    # =========================================================================

    async def create_template(
        self,
        owner_id: str,
        name: str,
        template_type: TemplateType,
        content: str,
        description: str | None = None,
        variables: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        visibility: TemplateVisibility = TemplateVisibility.PRIVATE,
        is_default: bool = False,
    ) -> PromptTemplate:
        """Create a new prompt template.

        Args:
            owner_id: Databricks workspace user ID.
            name: Display name for the template.
            template_type: Type of template (system, step, synthesis, query).
            content: Template content with {{variable}} placeholders.
            description: Optional description for the template.
            variables: Variable definitions with metadata.
            tags: Tags for filtering.
            visibility: Visibility level.
            is_default: Whether to set as default for this type.

        Returns:
            Created template.
        """
        # Extract variables from content if not provided
        if variables is None:
            variables = self._extract_variables(content)

        # If setting as default, unset other defaults for this type/owner
        if is_default:
            await self._unset_defaults(owner_id, template_type)

        template = PromptTemplate(
            owner_id=owner_id,
            name=name,
            type=template_type.value,
            content=content,
            description=description,
            variables=variables,
            tags=tags or [],
            visibility=visibility.value,
            is_default=is_default,
        )

        template = await self.add(template)
        logger.info(
            "Created prompt template",
            extra={
                "template_id": str(template.id),
                "owner_id": owner_id,
                "type": template_type.value,
                "name": name,
            },
        )
        return template

    def _extract_variables(self, content: str) -> list[dict[str, Any]]:
        """Extract variable names from template content.

        Args:
            content: Template content with {{variable}} placeholders.

        Returns:
            List of variable definitions with default metadata.
        """
        variable_names = set(VARIABLE_PATTERN.findall(content))
        return [
            {
                "name": name,
                "type": "string",
                "required": True,
                "default": None,
                "description": None,
            }
            for name in sorted(variable_names)
        ]

    async def _unset_defaults(self, owner_id: str, template_type: TemplateType) -> None:
        """Unset default flag on all templates of given type for owner.

        Args:
            owner_id: Owner user ID.
            template_type: Template type.
        """
        result = await self._session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.owner_id == owner_id,
                    PromptTemplate.type == template_type.value,
                    PromptTemplate.is_default.is_(True),
                )
            )
        )
        templates = list(result.scalars().all())

        for template in templates:
            template.is_default = False

        await self._session.flush()

    # =========================================================================
    # Template Rendering
    # =========================================================================

    def render_template(
        self,
        template: PromptTemplate,
        variables: dict[str, Any],
    ) -> tuple[str, list[str], list[str]]:
        """Render a template with variable substitution.

        Uses SafeTemplateRenderer to prevent SSTI attacks while supporting
        {{variable}}, {{#if}}/{{/if}}, {{#for}}/{{/for}}, and {{var|length}}.

        Args:
            template: Template to render.
            variables: Variable name to value mapping.

        Returns:
            Tuple of (rendered_content, missing_variables, used_defaults).
        """
        content = template.content
        missing_variables: list[str] = []
        used_defaults: list[str] = []

        # Build variable lookup from metadata
        var_metadata: dict[str, dict[str, Any]] = {
            var.get("name", ""): var
            for var in (template.variables or [])
            if var.get("name")
        }

        # Build the full context with defaults applied
        context: dict[str, Any] = {}
        content_variables = _safe_renderer.extract_variables(content)

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

        # Render using the safe template renderer
        rendered = _safe_renderer.render(content, context)

        return rendered, missing_variables, used_defaults

    def validate_variables(
        self,
        template: PromptTemplate,
        variables: dict[str, Any],
    ) -> list[str]:
        """Validate that all required variables are provided.

        Args:
            template: Template to validate against.
            variables: Variable name to value mapping.

        Returns:
            List of missing required variable names.
        """
        required = template.get_required_variables()
        return [name for name in required if name not in variables]

    # =========================================================================
    # Default Template Management
    # =========================================================================

    async def get_default_template(
        self,
        owner_id: str,
        template_type: TemplateType,
    ) -> PromptTemplate | None:
        """Get the default template of a given type for a user.

        First checks user's own defaults, then falls back to workspace defaults.

        Args:
            owner_id: User ID.
            template_type: Template type.

        Returns:
            Default template if found, None otherwise.
        """
        # First check user's own default
        result = await self._session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.owner_id == owner_id,
                    PromptTemplate.type == template_type.value,
                    PromptTemplate.is_default.is_(True),
                )
            )
        )
        template = result.scalar_one_or_none()

        if template:
            return template

        # Fall back to any workspace default
        result = await self._session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.visibility == TemplateVisibility.WORKSPACE.value,
                    PromptTemplate.type == template_type.value,
                    PromptTemplate.is_default.is_(True),
                )
            ).order_by(PromptTemplate.created_at)
        )
        return result.scalar_one_or_none()

    async def set_default_template(
        self,
        template_id: UUID,
        owner_id: str,
    ) -> PromptTemplate | None:
        """Set a template as the default for its type.

        Args:
            template_id: Template ID to set as default.
            owner_id: User ID (must own the template).

        Returns:
            Updated template if found and owned, None otherwise.
        """
        template = await self.get_for_user(template_id, owner_id)
        if not template:
            return None

        # Atomic flip: single UPDATE sets is_default=True for this template
        # and is_default=False for every other template of the same
        # (owner_id, type). Race-free against concurrent flips.
        await self.set_as_default(
            template_id=template.id,
            owner_id=owner_id,
            type_=TemplateType(template.type),
        )

        # Reload so the caller sees fresh state.
        return await self.get_for_user(template_id, owner_id)

    async def set_as_default(
        self,
        template_id: UUID,
        owner_id: str,
        type_: Any,
    ) -> None:
        """Atomically flip the default flag for ``(owner_id, type)``.

        Issues a single SQL ``UPDATE`` that sets ``is_default = (id =
        :template_id)`` for every row matching ``owner_id`` + ``type``,
        which simultaneously promotes the chosen template and demotes
        the previous default. Race-free against concurrent invocations
        because PostgreSQL serializes the update on the matching rows.
        """
        type_val = type_.value if hasattr(type_, "value") else str(type_)
        await self._session.execute(
            update(PromptTemplate)
            .where(
                and_(
                    PromptTemplate.owner_id == owner_id,
                    PromptTemplate.type == type_val,
                )
            )
            .values(
                is_default=(PromptTemplate.id == template_id),
                updated_at=datetime.now(UTC),
            )
        )
        await self._session.flush()

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_accessible_templates(
        self,
        user_id: str,
        template_type: TemplateType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[PromptTemplate], int]:
        """Get templates accessible to a user.

        Returns templates that are:
        - Owned by the user (any visibility)
        - OR workspace-visible

        Args:
            user_id: Databricks workspace user ID.
            template_type: Optional filter by template type.
            tags: Optional filter by tags (any match).
            limit: Maximum number of templates.
            offset: Number of templates to skip.

        Returns:
            Tuple of (templates, total_count).
        """
        # Build conditions: own templates OR workspace visible
        access_conditions = or_(
            PromptTemplate.owner_id == user_id,
            PromptTemplate.visibility == TemplateVisibility.WORKSPACE.value,
        )

        conditions = [access_conditions]

        if template_type:
            conditions.append(PromptTemplate.type == template_type.value)

        # Tag filtering uses JSONB containment
        if tags:
            # Check if any of the provided tags are in the template's tags
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(
                    PromptTemplate.tags.contains([tag])
                )
            if tag_conditions:
                conditions.append(or_(*tag_conditions))

        # Get total count
        count_query = select(func.count(PromptTemplate.id)).where(and_(*conditions))
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        # Get templates
        query = (
            select(PromptTemplate)
            .where(and_(*conditions))
            .order_by(PromptTemplate.name)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(query)
        templates = list(result.scalars().all())

        return templates, total

    async def get_for_user(
        self,
        template_id: UUID,
        user_id: str,
    ) -> PromptTemplate | None:
        """Get a template by ID with user ownership check.

        Args:
            template_id: Template ID.
            user_id: User ID (for ownership check).

        Returns:
            Template if found and owned by user, None otherwise.
        """
        result = await self._session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.id == template_id,
                    PromptTemplate.owner_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_accessible(
        self,
        template_id: UUID,
        user_id: str,
    ) -> PromptTemplate | None:
        """Get a template by ID if accessible to user.

        Template is accessible if:
        - Owned by the user
        - OR workspace-visible

        Args:
            template_id: Template ID.
            user_id: User ID.

        Returns:
            Template if accessible, None otherwise.
        """
        result = await self._session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.id == template_id,
                    or_(
                        PromptTemplate.owner_id == user_id,
                        PromptTemplate.visibility == TemplateVisibility.WORKSPACE.value,
                    ),
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_by_name(
        self,
        owner_id: str,
        name: str,
    ) -> PromptTemplate | None:
        """Get a template by name for a specific owner.

        Args:
            owner_id: Owner user ID.
            name: Template name.

        Returns:
            Template if found, None otherwise.
        """
        result = await self._session.execute(
            select(PromptTemplate).where(
                and_(
                    PromptTemplate.owner_id == owner_id,
                    PromptTemplate.name == name,
                )
            )
        )
        return result.scalar_one_or_none()

    async def search_by_tags(
        self,
        user_id: str,
        tags: list[str],
        limit: int = 50,
    ) -> list[PromptTemplate]:
        """Search templates by tags.

        Args:
            user_id: User ID.
            tags: Tags to search for (any match).
            limit: Maximum results.

        Returns:
            Matching templates.
        """
        templates, _ = await self.get_accessible_templates(
            user_id=user_id,
            tags=tags,
            limit=limit,
        )
        return templates
