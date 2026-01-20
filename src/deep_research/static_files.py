"""Static file serving for SPA deployment.

This module provides functionality to serve the React frontend from FastAPI
when running in production mode. The frontend is built to static assets
and served directly by the backend, eliminating the need for a separate
Node.js runtime.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Static directory is at project root /static/ (sibling to src/)
# Path: static_files.py -> deep_research/ -> src/ -> PROJECT_ROOT -> static/
STATIC_DIR = Path(__file__).parent.parent.parent / "static"


def is_static_serving_enabled() -> bool:
    """Check if static files should be served.

    Static serving is enabled when:
    1. SERVE_STATIC environment variable is set to "true"
    2. The static/index.html file exists (frontend has been built)

    Returns:
        True if static serving should be enabled.
    """
    serve_static_env = os.environ.get("SERVE_STATIC", "").lower() == "true"
    static_exists = (STATIC_DIR / "index.html").exists()
    return serve_static_env and static_exists


def setup_static_files(app: FastAPI) -> None:
    """Configure static file serving for SPA.

    This must be called AFTER all API routes are registered, as it adds
    a catch-all route that serves index.html for client-side routing.

    Args:
        app: FastAPI application instance.
    """
    if not is_static_serving_enabled():
        return

    # Mount /assets for bundled JS, CSS, and images
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="static-assets",
        )

    # Serve favicon/icons at root level
    @app.get("/vite.svg", include_in_schema=False)
    async def serve_vite_icon() -> FileResponse:
        """Serve the Vite icon."""
        icon_path = STATIC_DIR / "vite.svg"
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/svg+xml")
        raise HTTPException(status_code=404)

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon() -> FileResponse:
        """Serve favicon if it exists."""
        favicon_path = STATIC_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/x-icon")
        raise HTTPException(status_code=404)

    # SPA catch-all: serve index.html for all non-API routes
    # This enables client-side routing (React Router)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(request: Request, full_path: str) -> HTMLResponse:
        """Serve index.html for SPA client-side routing.

        This catch-all route serves index.html for any path that doesn't
        match an API route. This allows React Router to handle the routing
        on the client side.

        Args:
            request: The incoming request.
            full_path: The requested path.

        Returns:
            HTML response with index.html content.

        Raises:
            HTTPException: If the path is an API route or index.html doesn't exist.
        """
        # Don't serve index.html for API routes, docs, or health checks
        # These should 404 if not matched by their actual handlers
        if full_path.startswith((
            "api/",
            "docs",
            "redoc",
            "openapi.json",
            "health",
        )):
            raise HTTPException(status_code=404, detail="Not found")

        # Serve index.html for all other routes
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(
                content=index_path.read_text(encoding="utf-8"),
                status_code=200,
            )

        raise HTTPException(
            status_code=404,
            detail="Frontend not built. Run 'make build' first.",
        )
