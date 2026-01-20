"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from deep_research.api.v1 import router as api_v1_router
from deep_research.core.app_config import get_app_config
from deep_research.core.config import get_settings
from deep_research.core.exceptions import AppException, app_exception_handler, http_exception_handler
from deep_research.db.session import close_db, get_credential_provider
from deep_research.middleware.logging import RequestLoggingMiddleware, setup_logging
from deep_research.static_files import setup_static_files

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings.log_level)

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

    # Setup tracing (if available)
    try:
        from deep_research.core.tracing import setup_tracing

        setup_tracing()
    except ImportError:
        pass

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
    app.state.brave_client = BraveSearchClient()
    app.state.web_crawler = WebCrawler()

    # Initialize background job manager
    from deep_research.db.session import get_session_maker
    from deep_research.services.job_manager import initialize_job_manager

    job_manager = initialize_job_manager()
    session_maker = get_session_maker(settings)
    await job_manager.start(session_maker)
    app.state.job_manager = job_manager
    logger.info(
        "Job manager started: worker_id=%s",
        job_manager.worker_id,
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

    # Stop job manager first (cancels running jobs)
    if hasattr(app.state, "job_manager") and app.state.job_manager:
        await app.state.job_manager.stop()
        logger.info("Job manager stopped")

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

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Register exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Include API routers
    app.include_router(api_v1_router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers."""
        return {"status": "healthy", "service": "deep-research-agent"}

    # Setup static file serving for SPA (must be last - catch-all route)
    setup_static_files(app)

    return app


# Create application instance
app = create_app()
