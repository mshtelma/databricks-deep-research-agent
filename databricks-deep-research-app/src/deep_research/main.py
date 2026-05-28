"""FastAPI application entry point."""

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import databricks_deep_research._fips_compat  # noqa: F401  # FIPS md5 patch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from deep_research.api.v1 import router as api_v1_router
from deep_research.core.app_config import get_app_config
from deep_research.core.config import get_settings
from deep_research.core.exceptions import (
    AppException,
    app_exception_handler,
    http_exception_handler,
)
from deep_research.db.session import close_db, log_lakebase_connection_self_test
from deep_research.middleware.csrf import CSRFMiddleware
from deep_research.middleware.logging import RequestLoggingMiddleware, setup_logging
from deep_research.middleware.security import SecurityHeadersMiddleware
from deep_research.static_files import setup_static_files

logger = logging.getLogger(__name__)

# Session cleanup interval in seconds (5 minutes)
SESSION_CLEANUP_INTERVAL_SECONDS = 300


async def cleanup_expired_sessions_task(
    session_maker: Any,
    storage_stack: Any | None = None,
) -> None:
    """Background task to clean up expired incognito sessions periodically.

    Runs every 5 minutes to delete expired sessions and their associated chats.
    This prevents storage leaks and ensures privacy by removing incognito data
    after session expiry.

    F-OTHER.2: when `storage_stack` is provided and `storage_service_impl=cached`,
    routes through the cached `ISessionService` (no legacy session_maker needed).
    Legacy path stays as fallback.
    """
    from deep_research.core.config import get_settings
    from deep_research.services._impl_factory import make_session_service

    settings = get_settings()

    while True:
        try:
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
            if settings.storage_service_impl == "cached" and storage_stack is not None:
                service = make_session_service(settings, storage_stack)
                count = await service.cleanup_expired()
                if count > 0:
                    logger.info(f"Cleaned up {count} expired incognito sessions (cached)")
            else:
                async with session_maker() as db:
                    service = make_session_service(settings, None, session=db)
                    count = await service.cleanup_expired()
                    if count > 0:
                        await db.commit()
                        logger.info(f"Cleaned up {count} expired incognito sessions")
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
            raise
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}", exc_info=True)
            # Continue running despite errors


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings.log_level)

    # Deploy-here diagnostic banner — visible in app logs at startup so we
    # can confirm which build is live. Bump the marker on every iteration of
    # the deploy-here debugging loop.
    logger.info(
        "DEPLOY_HERE_BUILD_MARKER version=T2-diag-1 ts_module_loaded=%s",
        os.environ.get("DATABRICKS_APP_PORT", "no-app-port"),
    )

    # Surface missing required secrets at startup, not at synthesis end.
    # Workflows that declare web tools (web_research, brave_search, web_crawl)
    # fail loudly under strict_tool_resolution=True when BRAVE_API_KEY is
    # missing; corpus-only workflows are unaffected. Empty string and unset
    # are both treated as missing.
    if not (os.environ.get("BRAVE_API_KEY") or "").strip():
        logger.warning(
            "STARTUP_BRAVE_API_KEY_MISSING — web_research / brave_search / "
            "web_crawl tools will be unresolvable. Set BRAVE_API_KEY in the "
            "app env (via secret scope binding) for web workflows to function."
        )

    # Layer 2 of the layered tool-context validation: walk the static catalog
    # of tool kinds declared by the agent_designer registry and warn early
    # for any kind whose required ctx fields are *guaranteed* unsatisfiable
    # at process scope (e.g. text-table tools when STORAGE_WAREHOUSE_ID is
    # unset). We log instead of raise here because (a) per-request OBO
    # workspace_client is not yet available, and (b) workflows that don't
    # declare these kinds should still boot — the per-request Layer 3 guard
    # (``ToolResolver.validate_all``) catches actual unsatisfiable
    # declarations before LLM tokens are spent.
    try:
        from databricks_deep_research import required_ctx_fields_for_kind

        from deep_research.agent_designer.registry import _TOOL_KIND_META

        process_ctx_present: dict[str, bool] = {
            "search_client": bool((os.environ.get("BRAVE_API_KEY") or "").strip()),
            "schema_cache": bool(
                (os.environ.get("STORAGE_WAREHOUSE_ID") or "").strip()
                or (os.environ.get("TABLE_TOOLS_WAREHOUSE_ID") or "").strip()
            ),
            "sql_executor": bool(
                (os.environ.get("STORAGE_WAREHOUSE_ID") or "").strip()
                or (os.environ.get("TABLE_TOOLS_WAREHOUSE_ID") or "").strip()
            ),
            # workspace_client / table_registry / table_discovery_provider
            # are wired per-request by build_app_workflow_runner; we cannot
            # determine their presence at boot time.
        }
        unmet: dict[str, set[str]] = {}
        for kind in _TOOL_KIND_META:
            for field in required_ctx_fields_for_kind(kind):
                if field in process_ctx_present and not process_ctx_present[field]:
                    unmet.setdefault(field, set()).add(kind)
        if unmet:
            details = "; ".join(
                f"{field} unsatisfied (env unset) blocks: {', '.join(sorted(kinds))}"
                for field, kinds in sorted(unmet.items())
            )
            logger.warning(
                "STARTUP_TOOL_CATALOG_UNSATISFIABLE %s. Workflows declaring "
                "these kinds will fail at ToolResolver.validate_all() before "
                "execution. See preflight.resolve_warehouse_id_or_fail (for "
                "STORAGE_WAREHOUSE_ID) / BRAVE_API_KEY secret binding.",
                details,
            )
    except Exception:  # noqa: BLE001 — boot-diagnostic must never crash the app
        logger.exception("STARTUP_TOOL_CATALOG_SCAN_FAILED")

    # NOTE: Database migrations are NOT run here.
    # The app's service principal has limited permissions (CAN_CONNECT_AND_CREATE)
    # but cannot create tables in the public schema.
    # Migrations must be run remotely with developer credentials via:
    #   make deploy TARGET=dev  (runs migrations as part of deployment)
    #   make db-migrate-remote TARGET=dev  (manual migration only)

    # Validate central configuration (fail fast on startup)
    try:
        app_config = get_app_config()
        logger.info(
            "Central configuration loaded: %d endpoints, %d roles, default_role=%s",
            len(app_config.endpoints),
            len(app_config.models),
            app_config.default_role,
        )
    except Exception as e:
        logger.critical("Failed to load central configuration: %s", e)
        raise SystemExit(1) from e

    # Export Brave concurrency knobs as env vars so the framework's
    # BraveSearchAdapter (process-wide semaphore, retry count) picks them up.
    # The framework cannot import app config directly, so we bridge via env.
    os.environ.setdefault("BRAVE_MAX_CONCURRENCY", str(app_config.search.brave.max_concurrency))
    os.environ.setdefault("BRAVE_MAX_RETRIES", str(app_config.search.brave.max_retries))
    os.environ.setdefault(
        "BRAVE_INTER_CALL_JITTER_SECONDS",
        str(app_config.search.brave.inter_call_jitter_seconds),
    )

    # Setup tracing (if available)
    try:
        from deep_research.core.tracing import setup_tracing

        setup_tracing()
    except ImportError:
        pass

    # T1 fix verification: confirm deployment template directories resolved
    # to existing paths at startup. The actual import failure would surface
    # at first deploy attempt; logging here makes packaging regressions
    # observable proactively without crashing the app for users who don't
    # exercise the deployment feature.
    try:
        from deep_research.services.deployment.batch import _BATCH_TEMPLATE_DIR
        from deep_research.services.deployment.shell_app import _TEMPLATE_DIR

        logger.info(
            "DEPLOYMENT_TEMPLATES_RESOLVED shell=%s shell_exists=%s batch=%s batch_exists=%s",
            _TEMPLATE_DIR,
            _TEMPLATE_DIR.is_dir(),
            _BATCH_TEMPLATE_DIR,
            _BATCH_TEMPLATE_DIR.is_dir(),
        )
    except Exception:
        logger.exception("DEPLOYMENT_TEMPLATES_RESOLUTION_FAILED")

    # Initialize Lakebase credential provider if configured
    # NOTE: Credential pre-generation disabled - will generate on first DB request
    # if settings.use_lakebase:
    #     provider = get_credential_provider(settings)
    #     if provider:
    #         # Pre-generate credential to fail fast on startup
    #         provider.get_credential()
    #         logger.info("Lakebase OAuth credential initialized")
    logger.info("Lakebase credential will be generated on first database request")

    # Initialize shared services
    from deep_research.agent.tools.web_crawler import WebCrawler
    from deep_research.services.llm.client import LLMClient
    from deep_research.services.llm.config import ModelConfig
    from deep_research.services.search.brave import BraveSearchClient

    app.state.model_config = ModelConfig()
    app.state.llm_client = LLMClient(app.state.model_config)
    app.state.brave_client = BraveSearchClient(verify_ssl=settings.brave_verify_ssl)
    app.state.web_crawler = WebCrawler(verify_ssl=settings.brave_verify_ssl)

    # Initialize plugin manager (discovers and loads plugins via entry points)
    from deep_research.plugins.manager import PluginManager

    plugin_manager = PluginManager()
    try:
        plugin_manager.discover_and_load(app_config)
        app.state.plugin_manager = plugin_manager
        logger.info(
            "PluginManager initialized: %d plugins, %d phases",
            len(plugin_manager),
            len(plugin_manager.get_all_phases()),
        )
        # DIAGNOSTIC: Log plugin_manager storage for instance comparison
        logger.info(
            f"MAIN_PLUGIN_MANAGER_STORED instance_id={id(plugin_manager)} num_plugins={len(plugin_manager)} num_phases={len(plugin_manager.get_all_phases())} has_customization={plugin_manager.get_pipeline_customization() is not None}"
        )
    except Exception as e:
        logger.warning("PluginManager initialization failed: %s", e)
        app.state.plugin_manager = None

    # User-sync is now synchronous on cache miss (see middleware/auth.py
    # ``_ensure_user_synced``); no background task pool to drain.

    # Initialize the storage stack when the cached service layer is active.
    # Plan §Wave-6: under STORAGE_SERVICE_IMPL=cached the app runs on the
    # chat-document cache + WriteQueue. For the legacy impl (default) this
    # block is a no-op so the existing SQLAlchemy path is unaffected.
    app.state.storage_stack = None
    if settings.storage_service_impl == "cached":
        from deep_research.storage.factory import create_storage_stack

        try:
            storage_stack = create_storage_stack(settings)
            await storage_stack.start()
            storage_stack.install_signal_handlers()
            app.state.storage_stack = storage_stack
            logger.info(
                "StorageStack started: backend=%s, flush_interval=%ss",
                settings.storage_backend,
                settings.storage_flush_interval_sec,
            )
        except Exception as exc:
            logger.critical("StorageStack startup failed: %s", exc)
            raise

    # Initialize background job manager
    from deep_research.db.session import get_session_maker
    from deep_research.services.job_manager import initialize_job_manager

    job_manager = initialize_job_manager()
    session_maker = get_session_maker(settings)
    await log_lakebase_connection_self_test(settings)
    if app.state.storage_stack is not None:
        # Cached storage mode: hand the stack to JobManager so background
        # research jobs can thread it into `stream_research(...)`.
        job_manager.set_storage_stack(app.state.storage_stack)
    await job_manager.start(session_maker)
    app.state.job_manager = job_manager
    logger.info(
        "Job manager started: worker_id=%s",
        job_manager.worker_id,
    )

    # W12: async DeploymentJobRunner — separate from JobManager because the
    # research-job pool quota / lifecycle is sized for a different
    # workload. Janitor + orphan recovery start on .start().
    from deep_research.services.deployment.job_runner import DeploymentJobRunner

    deployment_runner = DeploymentJobRunner(session_factory=session_maker)
    await deployment_runner.start()
    app.state.deployment_runner = deployment_runner
    logger.info("DeploymentJobRunner started")

    # Start session cleanup background task
    cleanup_task = asyncio.create_task(
        cleanup_expired_sessions_task(session_maker, app.state.storage_stack)
    )
    app.state.cleanup_task = cleanup_task
    logger.info(
        "Session cleanup task started (runs every %d seconds)", SESSION_CLEANUP_INTERVAL_SECONDS
    )

    # Initialize HITL approval broker + periodic cleanup task (PR1/PR2 fix C3).
    # The broker is process-local; no cross-replica state. Cleanup reclaims
    # entries past the broker's grace window so long-running deployments do
    # not leak memory under sustained approval traffic.
    from databricks_deep_research.api.approval import InProcessApprovalBroker

    approval_broker = InProcessApprovalBroker()
    app.state.approval_broker = approval_broker
    APPROVAL_CLEANUP_INTERVAL_SECONDS = 60

    async def _approval_cleanup_loop() -> None:
        # CancelledError (BaseException) propagates through `except Exception`
        # and exits the loop on shutdown; non-cancel exceptions are logged.
        while True:
            try:
                await asyncio.sleep(APPROVAL_CLEANUP_INTERVAL_SECONDS)
                approval_broker.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Approval broker cleanup tick failed: %s", exc)

    app.state.approval_cleanup_task = asyncio.create_task(_approval_cleanup_loop())
    logger.info(
        "HITL approval broker initialized; cleanup runs every %d seconds",
        APPROVAL_CLEANUP_INTERVAL_SECONDS,
    )

    if settings.is_databricks_app and not settings.is_production:
        logger.warning(
            "DEBUG_ENDPOINTS_EXPOSED: Databricks App with APP_ENV=%s. "
            "Set APP_ENV=production for production deployments.",
            settings.app_env,
        )

    logger.info(
        "Application started: env=%s, is_databricks_app=%s, port=%s",
        settings.app_env,
        settings.is_databricks_app,
        settings.server_port,
    )

    yield

    # Graceful shutdown - Databricks Apps requires completion within 15 seconds
    logger.info("Shutdown signal received, cleaning up...")

    # Cancel session cleanup task
    if hasattr(app.state, "cleanup_task") and app.state.cleanup_task:
        app.state.cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.cleanup_task
        logger.info("Session cleanup task stopped")

    # Cancel HITL approval broker cleanup task
    if hasattr(app.state, "approval_cleanup_task") and app.state.approval_cleanup_task:
        app.state.approval_cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.approval_cleanup_task
        logger.info("Approval broker cleanup task stopped")

    # Stop job manager first (cancels running jobs)
    if hasattr(app.state, "job_manager") and app.state.job_manager:
        await app.state.job_manager.stop()
        logger.info("Job manager stopped")

    # W12: stop the DeploymentJobRunner — gracefully cancels in-flight
    # deployments and marks survivors FAILED with error_message=
    # "server_shutdown" so the UI doesn't poll them indefinitely.
    if hasattr(app.state, "deployment_runner") and app.state.deployment_runner:
        await app.state.deployment_runner.shutdown()
        logger.info("DeploymentJobRunner stopped")

    deploy_here_tasks = getattr(app.state, "deploy_here_tasks", None)
    if deploy_here_tasks:
        for task in list(deploy_here_tasks):
            task.cancel()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*deploy_here_tasks, return_exceptions=True),
                timeout=10.0,
            )
        logger.info("Deploy-here background tasks stopped")

    # User-sync is synchronous on cache miss now — nothing to drain here.

    # Drain the storage stack (if running). 15 s cap matches the Databricks
    # Apps shutdown budget; queue drops any remaining writes beyond that and
    # emits `storage_queue_backlog_at_shutdown`.
    if getattr(app.state, "storage_stack", None):
        try:
            await app.state.storage_stack.stop(timeout=15.0)
            logger.info("StorageStack stopped")
        except Exception as exc:
            logger.warning("StorageStack shutdown failed: %s", exc)

    # Flush MLflow traces before closing connections
    try:
        from deep_research.core.tracing import shutdown_tracing

        shutdown_tracing()
    except ImportError:
        pass

    # Cleanup shared services
    await app.state.llm_client.close()
    await app.state.web_crawler.close()
    await app.state.brave_client.close()
    logger.info("Shared services closed")

    # Cleanup database
    await close_db()
    logger.info("Database connections closed - shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Deep Research Agent API - Multi-agent research with step-by-step reflection",
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Content Security Policy ──
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    # Middleware is added in reverse execution order
    # Execution: SecurityHeaders → CORS → CSRF → Logging → Route

    # 1. Request logging (innermost — added first)
    app.add_middleware(RequestLoggingMiddleware)

    # 2. CSRF protection via Origin header validation
    csrf_origins = settings.csrf_allowed_origins
    if not csrf_origins:
        logger.warning(
            "CSRF_NO_ORIGINS No allowed origins configured; "
            "only same-origin requests will be permitted for state-changing methods"
        )
    app.add_middleware(
        CSRFMiddleware,
        allowed_origins=csrf_origins,
        enforce_https=settings.is_production,
    )

    # 3. CORS (tightened from wildcard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "x-forwarded-access-token",
            "X-Request-ID",
            "Accept",
        ],
    )

    # 4. Security response headers (outermost — added last)
    app.add_middleware(
        SecurityHeadersMiddleware,
        csp_policy=csp_policy,
        report_only=settings.csp_report_only,
        enable_hsts=settings.is_production,
    )

    # Register exception handlers
    app.add_exception_handler(AppException, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]

    # Include API routers
    app.include_router(api_v1_router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint for load balancers."""
        return {"status": "healthy", "service": "deep-research-agent"}

    # Setup static file serving for SPA (must be last - catch-all route)
    setup_static_files(app)

    return app


# Create application instance
app = create_app()
