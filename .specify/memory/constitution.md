<!--
Sync Impact Report
==================
Version change: N/A (initial) → 1.0.0
Modified principles: N/A (initial ratification)
Added sections:
  - Core Principles (4 principles)
  - Tooling Requirements
  - Development Workflow
  - Governance
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ compatible (Constitution Check section exists)
  - .specify/templates/spec-template.md: ✅ compatible (requirements format aligns)
  - .specify/templates/tasks-template.md: ✅ compatible (no principle-specific task types needed)
Follow-up TODOs: None
-->

# Databricks Deep Research Agent Constitution

## Core Principles

### I. Clients and Workspace Integration

All LLM interactions MUST use the OpenAI client or DSPy directly. All Databricks workspace
interactions MUST use `WorkspaceClient`.

**Requirements:**
- Prefer obtaining or configuring the OpenAI client through `WorkspaceClient` when available
  to keep authentication and configuration consistent.
- Design authentication with the assumption that a Databricks CLI profile may be used
  (now or later) for workspace access.
- Direct API calls bypassing these clients are prohibited unless explicitly justified.

**Rationale:** Consistent client usage ensures authentication flows remain centralized and
testable, reducing credential sprawl and simplifying configuration management across
development and production environments.

### II. Typing-First Python

Treat Python as a *statically typed* language for this project.

**Requirements:**
- All functions, methods, and public APIs MUST have type annotations for parameters and
  return types.
- Use precise types for complex structures: `Mapping[str, Any]`, `Sequence[T]`,
  dataclasses, and TypedDicts instead of generic `dict`/`list` where possible.
- Type annotations MUST accurately reflect runtime behavior; misleading annotations are
  treated as defects.

**Rationale:** Static typing surfaces bugs at development time rather than runtime, improves
IDE tooling and refactoring confidence, and serves as executable documentation for APIs.

### III. Avoid Runtime Introspection for Type Guarantees

Avoid `hasattr`, `isinstance` duck-typing, and similar runtime attribute checks for type
safety.

**Requirements:**
- Prefer explicit interfaces, protocols (`typing.Protocol`), and well-defined types over
  defensive runtime probing.
- Make type expectations explicit at boundaries (I/O, API calls, workspace interactions).
- When external data must be validated, use structured validation (e.g., Pydantic,
  dataclass parsing) rather than scattered attribute checks.

**Rationale:** Runtime introspection obscures type contracts, defeats static analysis, and
makes code harder to reason about. Explicit types at boundaries provide the same safety
with better tooling support.

### IV. Linting and Static Type Enforcement

Use linting and static type checking to surface typing issues early.

**Requirements:**
- Static type checking MUST pass before code is merged.
- Treat static typing errors as first-class engineering signals and address them
  proactively.
- Every function MUST make it obvious what goes in and what comes out through its
  signature.
- Type: ignore comments MUST include a justification comment explaining why suppression
  is necessary.

**Rationale:** Static enforcement catches type drift, missing annotations, and API
mismatches automatically, reducing review burden and preventing runtime TypeErrors.

## Tooling Requirements

**Language/Version:** Python 3.11+

**Required Static Analysis:**
- Type checker: mypy or pyright in strict mode
- Linter: ruff or flake8 with type-checking plugins

**Dependency Management:**
- Use `pyproject.toml` for project configuration
- Pin dependency versions in lock files for reproducibility

**Authentication:**
- Support Databricks CLI profile-based authentication
- Support environment variable fallback for CI/CD environments

## Development Workflow

**Before Commit:**
1. All type annotations present and accurate
2. Static type checker passes with no errors
3. Linter passes with no errors
4. No `hasattr` or unstructured runtime type checks introduced

**Code Review Gates:**
- Type coverage: All public APIs fully annotated
- No new `# type: ignore` without justification
- Client usage follows Principle I patterns

**Testing:**
- Tests MUST use typed fixtures and assertions where practical
- Integration tests MUST verify WorkspaceClient interactions work with expected
  authentication methods

## Governance

This constitution supersedes ad-hoc practices. Amendments require:
1. Written proposal documenting the change
2. Rationale explaining why the change is necessary
3. Impact assessment on existing code
4. Version increment following semantic versioning

**Versioning Policy:**
- MAJOR: Backward-incompatible principle removals or redefinitions
- MINOR: New principles added or materially expanded guidance
- PATCH: Clarifications, wording fixes, non-semantic refinements

**Compliance Review:**
- All PRs MUST be verified against these principles
- Complexity or principle violations MUST be justified in PR description
- Runtime guidance in README.md and docs/ MUST reference this constitution

**Version**: 1.0.0 | **Ratified**: 2025-12-20 | **Last Amended**: 2025-12-20
