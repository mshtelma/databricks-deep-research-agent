# Deploy to Databricks

Deep Research Agent deploys as a **native Databricks App**. A single command
provisions everything: the Lakebase database for conversation persistence, model
serving endpoint permissions, secret scope bindings, and the app itself. There are no
clusters to manage and no separate database servers to maintain.

## Prerequisites

1. **Databricks CLI configured** with a workspace profile:

    ```bash
    databricks configure --profile e2-demo-west
    databricks auth describe --profile e2-demo-west
    ```

2. **Model endpoints available** in your workspace for the simple / analytical /
   complex tiers (e.g. Claude Haiku / Sonnet / Opus class models).

3. **Brave secret** *(optional)* — only if a web tool pins `provider: brave`. The
   default `databricks` provider needs no secret:

    ```bash
    databricks secrets create-scope your-secret-scope
    databricks secrets put-secret your-secret-scope BRAVE_API_KEY
    ```

## One-command deployment

```bash
make deploy TARGET=dev        # development workspace
make deploy TARGET=ais        # production workspace

# Only if pinning Brave:
# make deploy TARGET=dev BRAVE_SCOPE=your-secret-scope
```

| TARGET | CLI profile | Description |
|--------|-------------|-------------|
| `dev` | `e2-demo-west` | Development workspace |
| `ais` | `ais` | Production workspace |

!!! note "Deploy gate"
    `make deploy` and `make app-deploy` are gated by `typecheck-framework` (strict
    mypy on the framework). Use the `*-unchecked` variants only for emergency reverts.
    Always rebuild the UI on deploy with `make app-deploy BUILD=1`.

## What the command does

The single command runs the full pipeline:

1. Build the frontend (`npm run build` → `static/`)
2. Generate `requirements.txt` from `pyproject.toml`
3. Bootstrap-deploy with `postgres` (creates the Lakebase instance)
4. Wait for Lakebase to be ready (~30–60 s for new instances)
5. Create the `deep_research` database
6. Re-deploy the bundle pointed at `deep_research`
7. Run migrations with developer credentials
8. Grant table permissions to the app service principal
9. Start the app and print a deployment summary

### Why two phases?

The app needs `LAKEBASE_DATABASE=deep_research`, but that database doesn't exist until
the Lakebase instance is created by the bundle deploy — a chicken-and-egg problem. The
solution is to deploy twice: first with `postgres` (always exists), then with
`deep_research` after creating it.

### Permission model

Tables are owned by the developer who runs migrations, not the app's service principal,
so the app needs explicit grants:

```sql
GRANT ALL ON ALL TABLES IN SCHEMA public TO <app_service_principal>;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO <app_service_principal>;
```

This is handled automatically during deployment.

## Operations

```bash
make logs TARGET=dev                       # fetch logs once
make logs TARGET=dev FOLLOW=-f             # follow in real time
make logs TARGET=dev SEARCH="--search ERROR"

databricks bundle run -t dev deep_research_agent   # restart after config change
databricks bundle summary -t dev                   # deployment status
```

## Run it as a library instead

To embed research into notebooks, jobs, or your own app, install the framework as a
standalone Python package and run workflows programmatically — same multi-agent
architecture, citation verification, and data-source access:

```bash
pip install "databricks-deep-research @ git+https://github.com/mshtelma/databricks-deep-research-agent.git@main#subdirectory=databricks-deep-research"
```

## Go deeper

- [Deployment (full docs)](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/docs/deployment.md)
- [Deploying a designed agent](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/docs/agent-deployment.md)
