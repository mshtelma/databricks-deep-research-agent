"""Service-implementation factory — selects legacy vs cached per service.

Single entry point the app uses to build a service instance. Reads
`settings.storage_service_impl` and returns the appropriate concrete class
while preserving the service's Protocol type so callers are oblivious.

Usage:

    # In a FastAPI dep:
    def get_feedback_service(
        session: AsyncSession = Depends(get_db),
        stack: StorageStack = Depends(get_storage),
        settings: Settings = Depends(get_settings),
    ) -> IFeedbackService:
        return make_feedback_service(settings, stack, session=session)

Extending for a new service:
1. Import its `I*` Protocol, `SQLAlchemy*` class, and `Cached*` class.
2. Add a `make_*_service(settings, stack, session=...)` function below.
3. Wire a FastAPI dep in `core/deps.py` that calls the factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deep_research.services._protocols import (
    IAuditLogService,
    IChatMemoryService,
    IChatService,
    ICustomAgentService,
    IDataSourceService,
    IExportService,
    IFeedbackService,
    IFileUploadService,
    IMessageService,
    IPreferencesService,
    IResearchEventService,
    IResearchSessionService,
    ISessionService,
    ISourceService,
    ITemplateService,
    IUserService,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from deep_research.core.config import Settings
    from deep_research.storage.factory import StorageStack


# --- Ready — cached impls exist ---------------------------------------------


def make_research_event_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IResearchEventService:
    """Return the research-event service matching `storage_service_impl`."""
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError(
                "cached research_event service requires a StorageStack"
            )
        from deep_research.services.cached.research_event import (
            CachedResearchEventService,
        )

        return CachedResearchEventService(stack)

    from deep_research.services.research_event_service import (
        ResearchEventService,
    )

    if session is None:
        raise ValueError("legacy research_event service requires an AsyncSession")
    return ResearchEventService(session)


def make_chat_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IChatService:
    """Return the chat service matching `storage_service_impl`.

    Cached path: `CachedChatService(stack)` — reads/writes go through the
    `StorageStack` (WriteQueue + ChatStateCache + backend). No SQLAlchemy.

    Legacy path: `ChatService(session)` — reads/writes go through the
    existing ORM on `public.chats`.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached chat service requires a StorageStack")
        from deep_research.services.cached.chat import CachedChatService

        return CachedChatService(stack)

    from deep_research.services.chat_service import ChatService

    if session is None:
        raise ValueError("legacy chat service requires an AsyncSession")
    return ChatService(session)


# --- Stubs — Wave 5 follow-ups add cached impls + wire them here -----------


def _not_yet_cached(service_name: str) -> None:
    raise NotImplementedError(
        f"{service_name} cached impl not yet landed — "
        "set STORAGE_SERVICE_IMPL=sqlalchemy_legacy until Wave 5 completes for this service."
    )


def make_feedback_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IFeedbackService:
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached feedback service requires a StorageStack")
        from deep_research.services.cached.feedback import CachedFeedbackService

        return CachedFeedbackService(stack)
    from deep_research.services.feedback_service import FeedbackService

    if session is None:
        raise ValueError("legacy feedback service requires an AsyncSession")
    return FeedbackService(session)


def make_user_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IUserService:
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached user service requires a StorageStack")
        from deep_research.services.cached.user import CachedUserService

        return CachedUserService(stack)
    from deep_research.services.user_service import UserService

    if session is None:
        raise ValueError("legacy user service requires an AsyncSession")
    return UserService(session)


def make_preferences_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IPreferencesService:
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached preferences service requires a StorageStack")
        from deep_research.services.cached.preferences import (
            CachedPreferencesService,
        )

        return CachedPreferencesService(stack)
    from deep_research.services.preferences_service import PreferencesService

    if session is None:
        raise ValueError("legacy preferences service requires an AsyncSession")
    return PreferencesService(session)


def make_chat_memory_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
    embedder: Any = None,
    llm: Any = None,
) -> IChatMemoryService:
    """Return the chat-memory service — the plugin-critical path.

    Legacy path keeps the existing `ChatMemoryService(session, embedder, llm=)`
    constructor; cached path uses `CachedChatMemoryService(stack, embedder, llm=)`
    which inherits the legacy render/snapshot/search/account_candidates code
    (Strategy A — see plan §Wave-5c).
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached chat_memory service requires a StorageStack")
        from deep_research.services.cached.chat_memory import (
            CachedChatMemoryService,
        )

        return CachedChatMemoryService(stack, embedder=embedder, llm=llm)
    from deep_research.services.chat_memory_service import ChatMemoryService

    if session is None:
        raise ValueError("legacy chat_memory service requires an AsyncSession")
    return ChatMemoryService(session, embedder=embedder, llm=llm)


def make_message_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IMessageService:
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached message service requires a StorageStack")
        from deep_research.services.cached.message import CachedMessageService

        return CachedMessageService(stack)
    from deep_research.services.message_service import MessageService

    if session is None:
        raise ValueError("legacy message service requires an AsyncSession")
    return MessageService(session)


def make_source_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> ISourceService:
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached source service requires a StorageStack")
        from deep_research.services.cached.source import CachedSourceService

        return CachedSourceService(stack)
    from deep_research.services.source_service import SourceService

    if session is None:
        raise ValueError("legacy source service requires an AsyncSession")
    return SourceService(session)


def make_research_session_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IResearchSessionService:
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached research_session service requires a StorageStack")
        from deep_research.services.cached.research_session import (
            CachedResearchSessionService,
        )

        return CachedResearchSessionService(stack)
    from deep_research.services.research_session_service import (
        ResearchSessionService,
    )

    if session is None:
        raise ValueError("legacy research_session service requires an AsyncSession")
    return ResearchSessionService(session)


def make_audit_log_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IAuditLogService:
    """Return the audit-log service matching ``storage_service_impl``.

    Cached path: ``CachedAuditLogService(stack)`` — appends rows to the
    ``audit_log`` table via ``WriteQueue`` (fire-and-forget, never raises).

    Legacy path: raises ``NotImplementedError`` — there is no legacy
    ``AuditLogService`` concrete class; the audit_log table was always
    written directly via ORM snippets. Until a legacy class is extracted,
    callers must use the cached path.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached audit_log service requires a StorageStack")
        from deep_research.services.cached.audit_log import CachedAuditLogService

        return CachedAuditLogService(stack)

    raise NotImplementedError(
        "AuditLogService legacy impl not yet extracted — "
        "set STORAGE_SERVICE_IMPL=cached to use the cached audit_log path."
    )


def make_custom_agent_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> ICustomAgentService:
    """Return the custom-agent service matching ``storage_service_impl``.

    Cached path: ``CachedCustomAgentService(stack)`` — reads/writes go through
    the ``StorageStack`` cold-path list tables.

    Legacy path: ``CustomAgentService(session)`` — reads/writes go through the
    existing ORM on ``public.custom_agents`` + ``public.agent_preset_steps``.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached custom_agent service requires a StorageStack")
        from deep_research.services.cached.custom_agent import CachedCustomAgentService

        return CachedCustomAgentService(stack)

    from deep_research.services.custom_agent_service import CustomAgentService

    if session is None:
        raise ValueError("legacy custom_agent service requires an AsyncSession")
    return CustomAgentService(session)


def make_file_upload_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
    storage_path: str | None = None,
) -> IFileUploadService:
    """Return the file-upload service matching ``storage_service_impl``.

    Cached path: ``CachedFileUploadService(stack, storage_path=...)`` —
    metadata goes through cold-path list tables; chunks through
    ``append_events`` (batched at 1000 rows/call).

    Legacy path: ``FileUploadService(session, storage_path=...)`` — ORM tables.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached file_upload service requires a StorageStack")
        from deep_research.services.cached.file_upload import CachedFileUploadService

        return CachedFileUploadService(stack, storage_path=storage_path)

    from deep_research.services.file_upload_service import FileUploadService

    if session is None:
        raise ValueError("legacy file_upload service requires an AsyncSession")
    return FileUploadService(session, storage_path=storage_path)


def make_data_source_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
    obo_client: Any = None,
) -> IDataSourceService:
    """Return the data-source service matching ``storage_service_impl``.

    Cached path: ``CachedDataSourceService(stack, obo_client=...)`` —
    metadata goes through cold-path list tables; OBO validation is still
    delegated to ``OBODatabricksClient``.

    Legacy path: ``DataSourceService(session, obo_client=...)`` — ORM tables.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached data_source service requires a StorageStack")
        from deep_research.services.cached.data_source import CachedDataSourceService

        return CachedDataSourceService(stack, obo_client=obo_client)

    from deep_research.services.data_source_service import DataSourceService

    if session is None:
        raise ValueError("legacy data_source service requires an AsyncSession")
    return DataSourceService(session, obo_client=obo_client)


def make_template_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> ITemplateService:
    """Return the template service matching ``storage_service_impl``.

    Cached path: ``CachedTemplateService(stack)`` — reads/writes go through
    the ``StorageStack`` cold-path list tables (``prompt_templates``).

    Legacy path: ``TemplateService(session)`` — ORM on ``public.prompt_templates``.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached template service requires a StorageStack")
        from deep_research.services.cached.template import CachedTemplateService

        return CachedTemplateService(stack)

    from deep_research.services.template_service import TemplateService

    if session is None:
        raise ValueError("legacy template service requires an AsyncSession")
    return TemplateService(session)


def make_export_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> IExportService:
    """Return the export service matching ``storage_service_impl``.

    Cached path: ``CachedExportService`` — composes reads from the already-cached
    chat, message, research-session, and source services. No direct SQL.

    Legacy path: ``ExportService(session)`` — direct ORM reads on the legacy
    Postgres tables.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached export service requires a StorageStack")
        from deep_research.services.cached.chat import CachedChatService
        from deep_research.services.cached.export import CachedExportService
        from deep_research.services.cached.message import CachedMessageService
        from deep_research.services.cached.research_session import CachedResearchSessionService
        from deep_research.services.cached.source import CachedSourceService

        return CachedExportService(
            chat_service=CachedChatService(stack),
            message_service=CachedMessageService(stack),
            research_session_service=CachedResearchSessionService(stack),
            source_service=CachedSourceService(stack),
        )

    from deep_research.services.export_service import ExportService

    if session is None:
        raise ValueError("legacy export service requires an AsyncSession")
    return ExportService(session)


def make_session_service(
    settings: "Settings",
    stack: "StorageStack | None" = None,
    *,
    session: "AsyncSession | None" = None,
) -> ISessionService:
    """Return the incognito-session service matching ``storage_service_impl``.

    Cached path: ``CachedSessionService(stack)`` — reads/writes go through
    the ``StorageStack`` cold-path list tables (``incognito_sessions``).

    Legacy path: ``SessionService(session)`` — ORM on ``public.incognito_sessions``.
    """
    if settings.storage_service_impl == "cached":
        if stack is None:
            raise ValueError("cached session service requires a StorageStack")
        from deep_research.services.cached.session import CachedSessionService

        return CachedSessionService(stack)

    from deep_research.services.session_service import SessionService

    if session is None:
        raise ValueError("legacy session service requires an AsyncSession")
    return SessionService(session)


__all__ = [
    "make_audit_log_service",
    "make_chat_service",
    "make_chat_memory_service",
    "make_custom_agent_service",
    "make_data_source_service",
    "make_export_service",
    "make_feedback_service",
    "make_file_upload_service",
    "make_user_service",
    "make_preferences_service",
    "make_research_event_service",
    "make_message_service",
    "make_source_service",
    "make_research_session_service",
    "make_template_service",
    "make_session_service",
]
