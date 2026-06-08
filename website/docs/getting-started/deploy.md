# Deploy to Databricks

The Deep Research Agent runs as a native **Databricks App** — deploying it to your
workspace *is* the install. Three steps:

!!! info "Prerequisites"
    - A **Databricks workspace** with [Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) enabled — these provide both the LLMs **and** the built-in web search, so **no external AI or search API keys are required**.
    - The [**Databricks CLI**](https://docs.databricks.com/en/dev-tools/cli/index.html) configured with a profile (`databricks configure`).
    - The repo cloned, plus **Node.js 18+**, [**uv**](https://docs.astral.sh/uv/), and **jq** (used by the build).

## 1. Point it at your workspace

Add a target for your workspace in `databricks-deep-research-app/databricks.yml`:

```yaml
targets:
  my-workspace:
    mode: development
    workspace:
      profile: my-profile        # a profile from ~/.databrickscfg
    variables:
      resource_suffix: "prod"    # app is named deep-research-agent-prod
```

`profile` points at an entry in your `~/.databrickscfg` — create one with:

```bash
databricks configure --profile my-profile   # prompts for host + token
```

## 2. Deploy

```bash
make deploy TARGET=my-workspace
```

This one command builds the frontend, provisions a **Lakebase** Postgres database for
conversation history, deploys the app, runs migrations, grants the app access, and
starts it.

## 3. Open the app

Find the app URL under **Compute → Apps** in your workspace (or run
`make logs TARGET=my-workspace`), open it, and ask your first question. 🎉

[What to do next :octicons-arrow-right-24:](quickstart.md){ .md-button .md-button--primary }

---

That's the whole happy path. Everything below is optional depth.

??? note "What `make deploy` actually does"
    Deploying a Databricks App backed by Lakebase is a two-phase process — the database
    must exist before the app can bind to it — so `make deploy` runs it end to end:

    1. **Discover** the Lakebase Autoscaling endpoint for your target and wait until it's ready.
    2. **Create** the `deep_research` database (idempotent).
    3. **Resolve** the bundle's Postgres binding variables (preflight).
    4. **Deploy** the Databricks Asset Bundle — the app, its Lakebase binding, and model-endpoint grants.
    5. **Migrate** the database schema using your developer credentials.
    6. **Grant** the app's service principal access to the tables, and write `../.env.<target>`.
    7. **Start** the app with `databricks bundle run` (this injects the resolved DB connection into the app config).

    Re-running `make deploy` is safe — every step is idempotent.

### Required model endpoints

The app routes work across [model tiers](../concepts/architecture.md#tiered-model-routing)
defined in `databricks-deep-research-app/config/app.yaml` — Claude, GPT, and Gemini class
models served by the Databricks Foundation Model APIs. They're declared as resource grants
in `databricks.yml`, so the app is granted `CAN_QUERY` on each at deploy time. If a model
isn't available in your workspace, the app **falls back** across the tier, so deployment
still succeeds.

### Updating a deployment

After the first full deploy, use the fast path for code/UI changes:

```bash
make app-deploy TARGET=my-workspace BUILD=1   # BUILD=1 rebuilds the UI
```

It skips Lakebase provisioning, migrations, and permission grants.

### Operations

```bash
make logs TARGET=my-workspace                 # fetch logs once
make logs TARGET=my-workspace FOLLOW=-f       # follow live
databricks bundle run -t my-workspace deep_research_agent   # restart
databricks bundle summary -t my-workspace                   # deployment status
```

### Optional: use Brave or Jina for web search

By default the agent uses **Databricks' built-in web search** — nothing to set up. To use
Brave instead:

1. Create a secret scope and store your key:
   ```bash
   databricks secrets create-scope my-scope
   databricks secrets put-secret my-scope BRAVE_API_KEY
   ```
2. Deploy with the scope: `make deploy TARGET=my-workspace BRAVE_SCOPE=my-scope`
3. Set `search.provider: brave` in `config/app.yaml`.

See [Web Search Providers](../guides/web-search-providers.md) for the full picture.

!!! note "Governance"
    Every data-source query runs under **On-Behalf-Of** authentication — the user's own
    token — so Unity Catalog permissions, row-level security, and column masking all
    apply. Database tables are owned by whoever runs the migrations; the deploy grants the
    app's service principal access to them.

## Run it as a library instead

To embed research in notebooks or jobs rather than deploy the app, install the framework
package:

```bash
pip install "databricks-deep-research @ git+https://github.com/mshtelma/databricks-deep-research-agent.git@main#subdirectory=databricks-deep-research"
```
