# Installation

This guide sets up Deep Research Agent for **local development**. To ship it to a
workspace instead, see [Deploy to Databricks](../deploy.md).

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend build & dev server |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager |
| Databricks CLI | latest | Only if deploying to Databricks |
| Brave Search API key | optional | Only for `provider: brave` — the default `databricks` provider uses built-in web search and needs **no key** |

!!! tip "No external search subscription required"
    Out of the box the agent uses Databricks' **built-in web search**, so it works
    on any Databricks workspace with no Brave/Jina account. See
    [Web Search Providers](../guides/web-search-providers.md).

## 1. Install dependencies

```bash
git clone https://github.com/mshtelma/databricks-deep-research-agent.git
cd databricks-deep-research-agent

# Installs backend (framework + app), frontend, and E2E dependencies
make install
```

## 2. Configure your environment

Create a `.env` file in the repository root. All `.env*` files live at the repo
root (never inside the app directory).

```bash
# --- Databricks authentication (choose ONE) ---
DATABRICKS_CONFIG_PROFILE=e2-demo-west       # recommended: profile-based
# OR
# DATABRICKS_HOST=https://your-workspace.databricks.com
# DATABRICKS_TOKEN=your-personal-access-token

# --- Lakebase (conversation persistence) ---
LAKEBASE_INSTANCE_NAME=deep-research-lakebase
LAKEBASE_DATABASE=deep_research

# --- Web search (optional) ---
# Default provider is `databricks` (built-in, no key). Only set this for Brave:
# BRAVE_API_KEY=your-brave-api-key

# --- Optional ---
APP_CONFIG_PATH=config/app.yaml
LOG_LEVEL=INFO
```

!!! note "Local PostgreSQL fallback"
    Local development uses **remote Lakebase** by default. For fully offline work
    you can point at a local database instead:

    ```bash
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/postgres
    ```

    Start it with `make db-local` (requires Docker).

## 3. Set up the database

=== "Remote Lakebase (default)"

    ```bash
    make db-migrate            # run migrations on the configured Lakebase
    make db-status             # check current migration status
    ```

    Use `DB_SUFFIX=dev` to isolate a database (targets `deep_research_dev`).

=== "Local PostgreSQL"

    ```bash
    make db-local              # start local PostgreSQL via Docker
    make db-migrate
    ```

## 4. Verify

```bash
make test                     # fast, mocked unit tests — no credentials needed
```

If that passes, you're ready to run the app.

[Continue to the Quickstart :octicons-arrow-right-24:](quickstart.md){ .md-button .md-button--primary }
