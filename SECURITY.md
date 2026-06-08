# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab → **Report a vulnerability**
   (or <https://github.com/mshtelma/databricks-deep-research-agent/security/advisories/new>).
2. Describe the issue, the affected component (framework, app, or deploy path), and
   steps to reproduce.

We aim to acknowledge reports within a few business days and will keep you updated as
we investigate and prepare a fix. Please give us a reasonable window to remediate
before any public disclosure.

## Supported versions

This project is pre-1.0 and under active development. Security fixes target the latest
released versions:

| Package | Version |
|---------|---------|
| `databricks-deep-research` (framework) | 0.2.x |
| `databricks-deep-research-app` (app) | 0.1.x |

## Scope and posture

The application is designed to run as a **Databricks App** and inherits the workspace's
security model:

- **On-Behalf-Of (OBO) authentication** — data-source queries use the end user's own
  OAuth token, so Unity Catalog permissions, row-level security, and column masking all
  apply. The app does not bypass governance with a shared service account.
- The LLM never sees raw URLs (only opaque integer references), reducing injection risk.
- CSRF protection and security headers are enabled on the API.

When reporting, it helps to note whether an issue involves the framework library, the
FastAPI/React application, the deployment tooling, or workspace configuration.

Please **do not** include real secrets, tokens, or customer data in a report — redact
them and describe the class of issue instead.
