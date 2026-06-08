# Documentation site

This directory holds the [MkDocs](https://www.mkdocs.org/) +
[Material](https://squidfunk.github.io/mkdocs-material/) source for the project's
GitHub Pages site, published at
<https://mshtelma.github.io/databricks-deep-research-agent/>.

It is self-contained — it does **not** read from the repo's other `docs/` trees,
so building it never touches application or framework documentation.

## Build & preview locally

```bash
cd website
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

mkdocs serve          # live-reload preview at http://127.0.0.1:8000
mkdocs build --strict # production build into ./site (must be warning-free)
```

`--strict` turns broken links and missing nav entries into errors — the same
gate CI enforces.

## Structure

```
website/
├── mkdocs.yml          # site config, nav, theme, markdown extensions
├── requirements.txt    # build dependencies
├── overrides/
│   └── home.html       # custom landing-page hero (extends Material's base)
└── docs/
    ├── index.md        # landing page
    ├── assets/         # logo, favicon, extra.css
    ├── getting-started/
    ├── concepts/
    ├── guides/
    ├── deploy.md
    └── benchmarks.md
```

## Deployment

`.github/workflows/docs.yml` builds this site and publishes it to GitHub Pages on
every push to `main` that touches `website/**`. The first deploy requires a
one-time setting: **repo Settings → Pages → Source → "GitHub Actions"**.
