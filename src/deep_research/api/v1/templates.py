"""Prompt Template API endpoints.

Provides CRUD operations for user-created prompt templates with
variable validation and rendering support.

Part of US5 - Custom Prompt Template Library (T068).
"""

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.exceptions import NotFoundError, ValidationError
from deep_research.db.session import get_db
from deep_research.middleware.auth import AuthenticatedUser, CurrentUser
from deep_research.models.prompt_template import (
    TemplateType as ModelTemplateType,
)
from deep_research.models.prompt_template import (
    TemplateVisibility as ModelTemplateVisibility,
)
from deep_research.schemas.template import (
    CreateTemplateRequest,
    RenderTemplateRequest,
    RenderTemplateResponse,
    TemplateListResponse,
    TemplateResponse,
    TemplateType,
    TemplateVariable,
    TemplateVisibility,
    UpdateTemplateRequest,
)
from deep_research.services.template_service import TemplateService

router = APIRouter(prefix="/templates", tags=["Templates"])


def _template_to_response(template: object) -> TemplateResponse:
    """Convert PromptTemplate model to response schema.

    Args:
        template: PromptTemplate model instance.

    Returns:
        TemplateResponse schema.
    """
    # Access attributes dynamically since we're using Any type
    variables_data = getattr(template, "variables", []) or []
    variables = [
        TemplateVariable(
            name=v.get("name", ""),
            type=v.get("type", "string"),
            required=v.get("required", True),
            default=v.get("default"),
            description=v.get("description"),
        )
        for v in variables_data
    ]

    return TemplateResponse(
        id=getattr(template, "id", None),
        owner_id=getattr(template, "owner_id", ""),
        name=getattr(template, "name", ""),
        type=TemplateType(getattr(template, "type", "system")),
        content=getattr(template, "content", ""),
        description=getattr(template, "description", None),
        variables=variables,
        tags=getattr(template, "tags", []) or [],
        visibility=TemplateVisibility(getattr(template, "visibility", "private")),
        is_default=getattr(template, "is_default", False),
        origin="user",
        created_at=getattr(template, "created_at", datetime.now(UTC)),
        updated_at=getattr(template, "updated_at", datetime.now(UTC)),
    )


# =============================================================================
# List and Get Endpoints
# =============================================================================


@router.get("", response_model=TemplateListResponse)
async def list_templates(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
    template_type: TemplateType | None = Query(None, description="Filter by template type"),
    tags: list[str] | None = Query(None, description="Filter by tags (any match)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> TemplateListResponse:
    """List prompt templates accessible to the current user.

    Returns both user-owned templates and workspace-visible templates.
    """
    service = TemplateService(db)

    # Convert schema enum to model enum if provided
    model_type = ModelTemplateType(template_type.value) if template_type else None

    templates, total = await service.get_accessible_templates(
        user_id=user.user_id,
        template_type=model_type,
        tags=tags,
        limit=limit,
        offset=offset,
    )

    # Count user vs workspace templates
    user_templates = sum(1 for t in templates if t.owner_id == user.user_id)
    workspace_templates = len(templates) - user_templates

    return TemplateListResponse(
        templates=[_template_to_response(t) for t in templates],
        total=total,
        user_templates=user_templates,
        workspace_templates=workspace_templates,
    )


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> TemplateResponse:
    """Get details of a specific prompt template.

    Returns the template if owned by user or workspace-visible.
    """
    service = TemplateService(db)
    template = await service.get_accessible(template_id, user.user_id)

    if not template:
        raise NotFoundError("Template", str(template_id))

    return _template_to_response(template)


# =============================================================================
# Create Endpoint
# =============================================================================


@router.post("", response_model=TemplateResponse, status_code=201)
async def create_template(
    request_body: CreateTemplateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> TemplateResponse:
    """Create a new prompt template.

    Variables are automatically extracted from {{placeholder}} syntax in content
    if not explicitly provided.
    """
    service = TemplateService(db)

    # Check for duplicate name
    existing = await service.get_by_name(user.user_id, request_body.name)
    if existing:
        raise ValidationError(f"Template with name '{request_body.name}' already exists")

    # Convert schema types to model types
    template_type = ModelTemplateType(request_body.type.value)
    visibility = ModelTemplateVisibility(request_body.visibility.value)

    # Convert variable schemas to dicts
    variables = [v.model_dump() for v in request_body.variables] if request_body.variables else None

    template = await service.create_template(
        owner_id=user.user_id,
        name=request_body.name,
        template_type=template_type,
        content=request_body.content,
        description=request_body.description,
        variables=variables,
        tags=request_body.tags,
        visibility=visibility,
        is_default=request_body.is_default,
    )

    await db.commit()
    return _template_to_response(template)


# =============================================================================
# Update Endpoint
# =============================================================================


@router.patch("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: UUID,
    request_body: UpdateTemplateRequest,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> TemplateResponse:
    """Update a prompt template.

    Only the template owner can update.
    """
    service = TemplateService(db)
    template = await service.get_for_user(template_id, user.user_id)

    if not template:
        raise NotFoundError("Template", str(template_id))

    # Check for duplicate name if changing
    if request_body.name is not None and request_body.name != template.name:
        existing = await service.get_by_name(user.user_id, request_body.name)
        if existing:
            raise ValidationError(f"Template with name '{request_body.name}' already exists")

    # Update fields
    if request_body.name is not None:
        template.name = request_body.name
    if request_body.description is not None:
        template.description = request_body.description
    if request_body.content is not None:
        template.content = request_body.content
    if request_body.variables is not None:
        template.variables = [v.model_dump() for v in request_body.variables]
    if request_body.tags is not None:
        template.tags = request_body.tags
    if request_body.visibility is not None:
        template.visibility = request_body.visibility.value
    if request_body.is_default is not None:
        if request_body.is_default:
            # Unset other defaults first
            await service._unset_defaults(user.user_id, ModelTemplateType(template.type))
        template.is_default = request_body.is_default

    template.updated_at = datetime.now(UTC)
    await service.update(template)
    await db.commit()

    return _template_to_response(template)


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: UUID,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a prompt template.

    Only the template owner can delete.
    """
    service = TemplateService(db)
    template = await service.get_for_user(template_id, user.user_id)

    if not template:
        raise NotFoundError("Template", str(template_id))

    await service.delete(template)
    await db.commit()


# =============================================================================
# Render Endpoint
# =============================================================================


@router.post("/{template_id}/render", response_model=RenderTemplateResponse)
async def render_template(
    template_id: UUID,
    request_body: RenderTemplateRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> RenderTemplateResponse:
    """Render a template with variable substitution.

    Returns the rendered content along with any missing required variables
    and variables that used default values.
    """
    service = TemplateService(db)
    template = await service.get_accessible(template_id, user.user_id)

    if not template:
        raise NotFoundError("Template", str(template_id))

    rendered_content, missing_variables, used_defaults = service.render_template(
        template, request_body.variables
    )

    return RenderTemplateResponse(
        rendered_content=rendered_content,
        missing_variables=missing_variables,
        used_defaults=used_defaults,
    )


# =============================================================================
# Default Template Endpoints
# =============================================================================


@router.get("/defaults/{template_type}", response_model=TemplateResponse | None)
async def get_default_template(
    template_type: TemplateType,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> TemplateResponse | None:
    """Get the default template for a given type.

    First checks user's own defaults, then workspace defaults.
    """
    service = TemplateService(db)
    model_type = ModelTemplateType(template_type.value)
    template = await service.get_default_template(user.user_id, model_type)

    if not template:
        return None

    return _template_to_response(template)


@router.post("/{template_id}/set-default", response_model=TemplateResponse)
async def set_default_template(
    template_id: UUID,
    user: AuthenticatedUser,
    db: AsyncSession = Depends(get_db),
) -> TemplateResponse:
    """Set a template as the default for its type.

    Only the template owner can set their templates as default.
    """
    service = TemplateService(db)
    template = await service.set_default_template(template_id, user.user_id)

    if not template:
        raise NotFoundError("Template", str(template_id))

    await db.commit()
    return _template_to_response(template)
