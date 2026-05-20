"""FastAPI dependency injectors for the storage stack + cached services.

One `get_storage(request)` returns the process-singleton `StorageStack`
constructed in `main.py`'s lifespan. Per-service `get_<svc>_service`
dispatchers wrap `_impl_factory.make_<svc>_service` so routes receive the
correct impl based on `settings.storage_service_impl` without knowing the
concrete class.

Usage in a route:

    @router.get("/feedback/{message_id}")
    async def read_feedback(
        message_id: UUID,
        svc: IFeedbackService = Depends(get_feedback_service),
    ) -> list[dict]:
        return await svc.get_feedback_for_message(message_id)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request

from deep_research.core.config import Settings, get_settings
from deep_research.services._impl_factory import (
    make_chat_service,
    make_data_source_service,
    make_export_service,
    make_feedback_service,
    make_file_upload_service,
    make_preferences_service,
    make_research_event_service,
    make_session_service,
    make_template_service,
    make_user_service,
)
from deep_research.services._protocols import (
    IChatService,
    IDataSourceService,
    IExportService,
    IFeedbackService,
    IFileUploadService,
    IPreferencesService,
    IResearchEventService,
    ISessionService,
    ITemplateService,
    IUserService,
)

if TYPE_CHECKING:

    from deep_research.storage.factory import StorageStack


# --- Storage stack --------------------------------------------------------


def get_storage(request: Request) -> StorageStack:
    """Return the process-singleton `StorageStack`.

    Raises HTTP 503 if the stack was not initialized (either cached mode is
    disabled or startup failed). Routes that only work under cached mode
    should declare this dependency directly.
    """
    stack = getattr(request.app.state, "storage_stack", None)
    if stack is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "storage stack not initialized "
                "(STORAGE_SERVICE_IMPL!=cached or startup failed)"
            ),
        )
    return stack


def get_storage_optional(request: Request) -> StorageStack | None:
    """Return the stack or None — use in services that support both impls."""
    return getattr(request.app.state, "storage_stack", None)


# --- Per-service dispatchers ----------------------------------------------
#
# Each dispatcher takes the settings, the optional stack, and a legacy
# AsyncSession (injected via the existing `get_db` helper that already lives
# in `db.session`). The factory picks the impl based on `storage_service_impl`.
#
# Lazy-imports keep the dep module light — avoids pulling in SQLAlchemy,
# the Databricks SDK, etc. at FastAPI startup just for type-hint resolution.


def get_chat_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IChatService:
    """Return the chat service matching `storage_service_impl`.

    Routes through `make_chat_service` — cached path requires a `StorageStack`,
    legacy path requires an `AsyncSession` attached to `request.state.db_session`.
    """
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_chat_service(settings, stack)
    return _with_legacy_session(make_chat_service, settings, request, get_db)


def get_feedback_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IFeedbackService:
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_feedback_service(settings, stack)
    # Legacy path still needs a live session.
    return _with_legacy_session(make_feedback_service, settings, request, get_db)


def get_user_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IUserService:
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_user_service(settings, stack)
    return _with_legacy_session(make_user_service, settings, request, get_db)


def get_preferences_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IPreferencesService:
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_preferences_service(settings, stack)
    return _with_legacy_session(make_preferences_service, settings, request, get_db)


def get_research_event_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IResearchEventService:
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_research_event_service(settings, stack)
    return _with_legacy_session(
        make_research_event_service, settings, request, get_db
    )


def get_file_upload_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IFileUploadService:
    """Return the file-upload service matching ``storage_service_impl``."""
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_file_upload_service(settings, stack)
    return _with_legacy_session(make_file_upload_service, settings, request, get_db)


def get_data_source_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IDataSourceService:
    """Return the data-source service matching ``storage_service_impl``."""
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_data_source_service(settings, stack)
    return _with_legacy_session(make_data_source_service, settings, request, get_db)


def get_template_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ITemplateService:
    """Return the template service matching ``storage_service_impl``."""
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_template_service(settings, stack)
    return _with_legacy_session(make_template_service, settings, request, get_db)


def get_session_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ISessionService:
    """Return the incognito-session service matching ``storage_service_impl``."""
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_session_service(settings, stack)
    return _with_legacy_session(make_session_service, settings, request, get_db)


def get_export_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> IExportService:
    """Return the export service matching ``storage_service_impl``.

    Cached path: ``CachedExportService`` — pure composition over already-cached
    chat, message, research-session, and source services.

    Legacy path: ``ExportService(session)`` — direct ORM reads.
    """
    from deep_research.db.session import get_db

    stack = get_storage_optional(request)
    if settings.storage_service_impl == "cached":
        return make_export_service(settings, stack)
    return _with_legacy_session(make_export_service, settings, request, get_db)


# --- Internal -------------------------------------------------------------


def _with_legacy_session(factory, settings, request, _get_db):  # type: ignore[no-untyped-def]
    """Helper: call a factory with a legacy session if one is already
    attached to the request (e.g. via the existing Depends(get_db) in the
    route signature). Otherwise raise a 503 — the legacy path requires a
    session that only `Depends(get_db)` knows how to open.

    Most routes already declare `db: AsyncSession = Depends(get_db)` and
    pass it through; this helper is a fallback for routes that don't.
    """
    session = getattr(request.state, "db_session", None)
    if session is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "legacy service requested but no AsyncSession on request.state.db_session; "
                "declare `db: AsyncSession = Depends(get_db)` in the route "
                "and pass it through, or set STORAGE_SERVICE_IMPL=cached."
            ),
        )
    return factory(settings, None, session=session)
