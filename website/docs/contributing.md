# Contributing & local development

This project is open source under the [Apache License 2.0](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/LICENSE).
Issues, docs, and pull requests are welcome — see the full
[CONTRIBUTING guide](https://github.com/mshtelma/databricks-deep-research-agent/blob/main/CONTRIBUTING.md)
on GitHub.

## Local development

You need **Python 3.11+**, **Node.js 18+**, and [**uv**](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mshtelma/databricks-deep-research-agent.git
cd databricks-deep-research-agent

make install        # backend (framework + app) + frontend + e2e deps
make dev            # backend :8000 + frontend :5173, hot reload
```

Local development connects to a Databricks **Lakebase** by default (a local PostgreSQL
fallback is available via `make db-local`). See [Configuration](guides/configuration.md)
for connection details.

## Quality gates

Run these before opening a PR — they mirror CI and the deploy gate:

```bash
make test           # unit tests (fast, mocked, no credentials)
make typecheck      # mypy (framework strict) + tsc
make lint           # ruff + eslint
make format         # ruff format + prettier
```

The framework must stay type-clean: `make deploy` is gated on `typecheck-framework`.

## Working on these docs

The published site lives in `website/` (MkDocs + Material):

```bash
cd website
pip install -r requirements.txt
mkdocs serve              # live preview at http://127.0.0.1:8000
mkdocs build --strict     # must be warning-free (CI enforces this)
```

Changes under `website/**` deploy to GitHub Pages automatically when merged to `main`.

## Pull requests

1. Branch off `main` (or `make worktree BRANCH=my-feature`).
2. Keep PRs focused; add or update tests for behavior changes.
3. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, …).
4. Ensure `make test`, `make typecheck`, and `make lint` pass.
5. Open the PR against `main` and describe how you verified it.
