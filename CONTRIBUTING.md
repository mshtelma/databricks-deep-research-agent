# Contributing

Thanks for your interest in improving the **Databricks Deep Research Agent**! This
project is open source under the [Apache License 2.0](LICENSE), and contributions —
issues, docs, and pull requests — are welcome.

> 📖 Full documentation: <https://mshtelma.github.io/databricks-deep-research-agent/>

## Ways to contribute

- **Report a bug** or request a feature via GitHub Issues.
- **Improve the docs** under `website/` (the published site) or the in-repo `docs/`.
- **Send a pull request** — bug fixes, new tools/agents, search providers, tests.

## Development setup

Local development needs **Python 3.11+**, **Node.js 18+**, and [**uv**](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mshtelma/databricks-deep-research-agent.git
cd databricks-deep-research-agent

make install        # backend (framework + app), frontend, and e2e deps
make dev            # backend :8000 + frontend :5173, with hot reload
```

Local development connects to a Databricks **Lakebase** by default; see the
[Configuration](https://mshtelma.github.io/databricks-deep-research-agent/guides/configuration/)
docs and `.env.example` for connection options (a local PostgreSQL fallback is available
via `make db-local`).

### Use a worktree for each change

The project is set up for git worktrees so the main checkout stays clean:

```bash
make worktree BRANCH=my-feature INSTALL=1
cd ../.worktrees/my-feature
```

## Quality gates

Run these before opening a PR — they mirror CI and the deploy gate:

```bash
make test           # unit tests (fast, mocked, no credentials)
make typecheck      # mypy (framework is strict) + tsc
make lint           # ruff + eslint
make format         # ruff format + prettier
```

`make test-all` additionally runs the frontend tests; `make e2e` runs Playwright
end-to-end suites (these need workspace credentials).

> **Deploy gate:** `make deploy` / `make app-deploy` require `typecheck-framework`
> (strict mypy on the framework) to pass. Keep the framework type-clean.

## Working on the docs site

The published site lives in `website/` (MkDocs + Material):

```bash
cd website
pip install -r requirements.txt
mkdocs serve              # live preview at http://127.0.0.1:8000
mkdocs build --strict     # must be warning-free (CI enforces this)
```

Changes to `website/**` deploy automatically to GitHub Pages when merged to `main`.

## Pull request guidelines

1. Branch off `main` (or use `make worktree`).
2. Keep PRs focused; add or update tests for behavior changes.
3. Use clear commit messages — this repo follows
   [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
   `docs:`, `chore:`, …).
4. Make sure `make test`, `make typecheck`, and `make lint` pass.
5. Open the PR against `main` and describe the change and how you verified it.

Project conventions and architecture notes live in [`CLAUDE.md`](CLAUDE.md) and the
[documentation site](https://mshtelma.github.io/databricks-deep-research-agent/).

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
