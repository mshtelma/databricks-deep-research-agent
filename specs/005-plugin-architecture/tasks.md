# Tasks: Plugin Architecture

**Branch**: `005-plugin-architecture` | **Generated**: 2026-01-17 | **Spec**: [spec.md](./spec.md)

## Overview

This document contains the implementation tasks for the Plugin Architecture feature, organized by implementation phase and user story. Total estimated tasks: ~115.

## Critical Path

```
Setup → Foundational → US5 (Package) → US3 (Plugins) ──┬─→ US1 (Vector Search) → US7 (Citations)
                                                        ├─→ US2 (Knowledge Assistant)
                                                        ├─→ US8 (Pipeline) → US10 (Conversation)
                                                        ├─→ US9 (Output Types)
                                                        ├─→ US6 (Deployment)
                                                        └─→ US4 (Frontend)
```

## Legend

- `[ ]` Task not started
- `[~]` Task in progress
- `[x]` Task completed
- `[!]` Task blocked
- **P1**: Priority 1 (required for MVP)
- **P2**: Priority 2 (required but can follow)
- `FR-XXX`: Functional Requirement reference
- `US-X`: User Story reference

---

## Phase 0: Setup

### 0.1 Environment Setup

| Task | Description | FR | Status |
|------|-------------|----|----|
| 0.1.1 | Verify Python 3.11+ and development environment | - | [x] |
| 0.1.2 | Verify Databricks SDK >= 0.30.0 is available | - | [x] |
| 0.1.3 | Create feature branch `005-plugin-architecture` | - | [x] |

### 0.2 Project Structure Verification

| Task | Description | FR | Status |
|------|-------------|----|----|
| 0.2.1 | Audit current `src/` directory structure for restructure planning | FR-240 | [x] |
| 0.2.2 | Audit all imports to prepare bulk find-replace | FR-241 | [x] |
| 0.2.3 | Verify `pyproject.toml` current state and entry point support | FR-242 | [x] |

---

## Phase 1: Foundational Infrastructure

**Blocks**: All User Stories

### 1.1 Package Restructure (FR-240 series) - Week 1

| Task | Description | FR | Status |
|------|-------------|----|----|
| 1.1.1 | Create `src/deep_research/` directory | FR-240 | [x] |
| 1.1.2 | Move all files from `src/` to `src/deep_research/` | FR-240 | [x] |
| 1.1.3 | Update all Python imports to `from deep_research.xxx import yyy` format | FR-241 | [x] |
| 1.1.4 | Update all relative imports to absolute imports | FR-241 | [x] |
| 1.1.5 | Update `pyproject.toml` with new package structure | FR-242 | [x] |
| 1.1.6 | Verify `pip install -e .` works in clean environment | FR-243 | [x] |
| 1.1.7 | Create `src/deep_research/__init__.py` with public API exports | FR-244 | [x] |
| 1.1.8 | Update `alembic.ini` for new package path | FR-252 | [x] |
| 1.1.9 | Update all test imports in `tests/` directory | FR-241 | [x] |
| 1.1.10 | Update `main.py` and `static_files.py` for new structure | FR-240 | [x] |
| 1.1.11 | Run full test suite to verify no regressions | - | [x] |
| 1.1.12 | Update `Makefile` for new package structure | - | [x] |
| 1.1.13 | Update `CLAUDE.md` documentation for new import paths | - | [x] |

### 1.2 Tool Infrastructure (FR-200 series) - Week 2

| Task | Description | FR | Status |
|------|-------------|----|----|
| 1.2.1 | Create `src/deep_research/agent/tools/` directory structure | FR-200 | [x] |
| 1.2.2 | Implement `ToolDefinition` dataclass in `base.py` | FR-200 | [x] |
| 1.2.3 | Implement `ToolResult` dataclass in `base.py` | FR-200 | [x] |
| 1.2.4 | Implement `ResearchTool` protocol in `base.py` | FR-200 | [x] |
| 1.2.5 | Implement `ResearchContext` dataclass in `base.py` | FR-203 | [x] |
| 1.2.6 | Implement `ToolRegistry` class in `registry.py` | FR-201 | [x] |
| 1.2.7 | Add `register()` method with conflict detection | FR-201 | [x] |
| 1.2.8 | Add `register_with_prefix()` for conflict resolution | FR-201 | [x] |
| 1.2.9 | Add `get()`, `list_tools()`, `get_openai_tools()` methods | FR-201 | [x] |
| 1.2.10 | Refactor `web_search` tool to implement `ResearchTool` protocol | FR-202 | [x] |
| 1.2.11 | Refactor `web_crawl` tool to implement `ResearchTool` protocol | FR-202 | [x] |
| 1.2.12 | Write unit tests for `ToolRegistry` | - | [x] |
| 1.2.13 | Write unit tests for refactored core tools | - | [x] |

### 1.3 Plugin System Core (FR-230 series) - Week 3

| Task | Description | FR | Status |
|------|-------------|----|----|
| 1.3.1 | Create `src/deep_research/plugins/` directory structure | FR-230 | [x] |
| 1.3.2 | Implement `ResearchPlugin` protocol in `base.py` | FR-230 | [x] |
| 1.3.3 | Implement `ToolProvider` protocol in `base.py` | FR-231 | [x] |
| 1.3.4 | Implement `PromptProvider` protocol in `base.py` | FR-232 | [x] |
| 1.3.5 | Implement `PluginManager` class in `manager.py` | FR-233 | [x] |
| 1.3.6 | Implement `discover_plugins()` function in `discovery.py` | FR-234 | [x] |
| 1.3.7 | Add entry point discovery using `importlib.metadata` | FR-234 | [x] |
| 1.3.8 | Implement plugin initialization with error isolation | FR-235 | [x] |
| 1.3.9 | Implement plugin shutdown lifecycle | FR-230 | [x] |
| 1.3.10 | Add plugin-specific configuration loading from `app.yaml` | FR-236 | [x] |
| 1.3.11 | Update `AppConfig` in `core/app_config.py` for plugin section | FR-236 | [x] |
| 1.3.12 | Create `plugins/__init__.py` with public exports | FR-230 | [x] |
| 1.3.13 | Write unit tests for `PluginManager` | - | [x] |
| 1.3.14 | Write unit tests for `discovery.py` | - | [x] |
| 1.3.15 | Write integration test for plugin lifecycle | - | [x] |

---

## Phase 2: User Story 5 - Package Core as pip-installable Library (P1)

**Blocks**: US3, US4, US6

### 2.1 pyproject.toml Configuration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 2.1.1 | Update `[project]` section with name `databricks-deep-research` | FR-270 | [x] |
| 2.1.2 | Configure proper version and description | FR-270 | [x] |
| 2.1.3 | Add all dependencies with version constraints | FR-270 | [x] |
| 2.1.4 | Configure `[project.scripts]` for CLI commands | FR-246 | [x] |
| 2.1.5 | Configure `[project.entry-points."deep_research.tools"]` for core tools | FR-245 | [x] |
| 2.1.6 | Configure `[project.entry-points."deep_research.plugins"]` group | FR-234 | [x] |

### 2.2 Public API Exports

| Task | Description | FR | Status |
|------|-------------|----|----|
| 2.2.1 | Export `run_research`, `stream_research` from orchestrator | FR-271 | [x] |
| 2.2.2 | Export `ResearchPlugin`, `ToolProvider`, `PromptProvider` from plugins | FR-271 | [x] |
| 2.2.3 | Export `ResearchTool`, `ToolDefinition`, `ToolResult`, `ResearchContext` | FR-271 | [x] |
| 2.2.4 | Export `LLMClient` from services | FR-271 | [x] |
| 2.2.5 | Export all agent nodes (`Coordinator`, `Planner`, etc.) | FR-271 | [x] |
| 2.2.6 | Export `ResearchState`, `StreamEvent` types | FR-271 | [x] |
| 2.2.7 | Document all public exports in `__init__.py` docstrings | FR-277 | [x] |

### 2.3 Installation Verification

| Task | Description | FR | Status |
|------|-------------|----|----|
| 2.3.1 | Test `pip install -e .` in clean virtual environment | FR-243 | [x] |
| 2.3.2 | Verify `from deep_research.agent.orchestrator import run_research` works | FR-271 | [x] |
| 2.3.3 | Verify `from deep_research.plugins import ResearchPlugin` works | FR-271 | [x] |
| 2.3.4 | Verify entry points are discoverable (`pip show databricks-deep-research`) | FR-245 | [x] |
| 2.3.5 | Create SC-004 acceptance test: pip install in clean env | SC-004 | [x] |

---

## Phase 3: User Story 3 - Build Child Research Agent from Parent Package (P1)

**Depends on**: US5

### 3.1 Plugin Discovery Enhancement

| Task | Description | FR | Status |
|------|-------------|----|----|
| 3.1.1 | Ensure plugins from installed packages are discovered | FR-234 | [x] |
| 3.1.2 | Add logging for discovered plugins on startup | FR-235 | [x] |
| 3.1.3 | Implement plugin name conflict detection and warning | FR-235 | [x] |

### 3.2 Configuration Override System

| Task | Description | FR | Status |
|------|-------------|----|----|
| 3.2.1 | Implement config file path override via `APP_CONFIG_PATH` env var | FR-272 | [x] |
| 3.2.2 | Support child project providing own `app.yaml` | FR-272 | [x] |
| 3.2.3 | Implement config merging (child overrides parent defaults) | FR-275 | [x] |
| 3.2.4 | Document configuration override mechanism | FR-275 | [x] |

### 3.3 Prompt Override System

| Task | Description | FR | Status |
|------|-------------|----|----|
| 3.3.1 | Update agent nodes to accept prompt overrides | FR-273 | [x] |
| 3.3.2 | Implement `PromptProvider` integration in orchestrator | FR-232 | [x] |
| 3.3.3 | Support per-agent prompt customization keys | FR-232 | [x] |
| 3.3.4 | Document prompt override format and available keys | FR-232 | [x] |

### 3.4 Tool Integration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 3.4.1 | Update researcher agent to use `ToolRegistry` | FR-201 | [x] |
| 3.4.2 | Pass `ResearchContext` to all tool executions | FR-203 | [x] |
| 3.4.3 | Integrate plugin-provided tools alongside core tools | FR-231 | [x] |
| 3.4.4 | Implement tool name prefixing for conflicts | FR-201 | [x] |

### 3.5 Example Child Project Structure

| Task | Description | FR | Status |
|------|-------------|----|----|
| 3.5.1 | Create `examples/child-project/` directory structure | - | [x] |
| 3.5.2 | Create example `pyproject.toml` for child project | FR-273 | [x] |
| 3.5.3 | Create example plugin with custom tool | FR-273 | [x] |
| 3.5.4 | Create example `app.yaml` with overrides | FR-272 | [x] |
| 3.5.5 | Document child project quickstart | FR-251 | [x] |

### 3.6 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 3.6.1 | Write test: child project installs parent as dependency | SC-003 | [x] |
| 3.6.2 | Write test: plugin is discovered on startup | SC-003 | [x] |
| 3.6.3 | Write test: custom tools available in researcher | SC-003 | [x] |
| 3.6.4 | Write test: app starts despite plugin failure | SC-006 | [x] |

---

## Phase 4: User Story 1 - Configure Built-in Vector Search Tool (P1)

**Depends on**: Phase 1 (Tool Infrastructure)

### 4.1 Vector Search Tool Implementation

| Task | Description | FR | Status |
|------|-------------|----|----|
| 4.1.1 | Create `src/deep_research/agent/tools/vector_search.py` | FR-210 | [x] |
| 4.1.2 | Implement `VectorSearchTool` class implementing `ResearchTool` | FR-210 | [x] |
| 4.1.3 | Implement `definition` property with JSON Schema parameters | FR-210 | [x] |
| 4.1.4 | Implement `execute()` using Databricks SDK `VectorSearchClient` | FR-210 | [x] |
| 4.1.5 | Implement `validate_arguments()` method | FR-210 | [x] |
| 4.1.6 | Return `ToolResult` with `source_type: vector_search` | FR-214 | [x] |
| 4.1.7 | Include source metadata (index name, relevance score) | FR-214 | [x] |

### 4.2 Configuration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 4.2.1 | Create `VectorSearchEndpointConfig` Pydantic model | FR-213 | [x] |
| 4.2.2 | Create `VectorSearchConfig` Pydantic model | FR-211 | [x] |
| 4.2.3 | Add `vector_search` section to `AppConfig` | FR-211 | [x] |
| 4.2.4 | Add example config to `config/app.yaml` (disabled by default) | FR-212 | [x] |
| 4.2.5 | Implement config validation on startup | FR-213 | [x] |

### 4.3 Authentication

| Task | Description | FR | Status |
|------|-------------|----|----|
| 4.3.1 | Reuse WorkspaceClient OAuth for Vector Search auth | FR-215 | [x] |
| 4.3.2 | Handle authentication errors gracefully | FR-215 | [x] |

### 4.4 Tool Registration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 4.4.1 | Create tool factory for Vector Search endpoints | FR-211 | [x] |
| 4.4.2 | Generate tool names as `search_<endpoint_name>` | FR-211 | [x] |
| 4.4.3 | Register VS tools in `ToolRegistry` on startup | FR-245 | [x] |
| 4.4.4 | Add to `deep_research.tools` entry point | FR-245 | [x] |

### 4.5 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 4.5.1 | Write unit tests for `VectorSearchTool` | - | [x] |
| 4.5.2 | Write unit tests for config validation | - | [x] |
| 4.5.3 | Write integration test with real VS endpoint (if available) | - | [x] |
| 4.5.4 | Create SC-001 acceptance test: enable VS via config only | SC-001 | [x] |

---

## Phase 5: User Story 2 - Configure Built-in Knowledge Assistant Tool (P1)

**Depends on**: Phase 1 (Tool Infrastructure)

### 5.1 Knowledge Assistant Tool Implementation

| Task | Description | FR | Status |
|------|-------------|----|----|
| 5.1.1 | Create `src/deep_research/agent/tools/knowledge_assistant.py` | FR-220 | [x] |
| 5.1.2 | Implement `KnowledgeAssistantTool` class implementing `ResearchTool` | FR-220 | [x] |
| 5.1.3 | Implement `definition` property with JSON Schema parameters | FR-220 | [x] |
| 5.1.4 | Implement `execute()` using Databricks serving endpoint API | FR-220 | [x] |
| 5.1.5 | Implement `validate_arguments()` method | FR-220 | [x] |
| 5.1.6 | Preserve source citations from KA responses | FR-224 | [x] |
| 5.1.7 | Return `ToolResult` with `source_type: knowledge_assistant` | FR-227 | [x] |

### 5.2 Configuration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 5.2.1 | Create `KnowledgeAssistantEndpointConfig` Pydantic model | FR-223 | [x] |
| 5.2.2 | Create `KnowledgeAssistantsConfig` Pydantic model | FR-221 | [x] |
| 5.2.3 | Add `knowledge_assistants` section to `AppConfig` | FR-221 | [x] |
| 5.2.4 | Add example config to `config/app.yaml` (disabled by default) | FR-222 | [x] |
| 5.2.5 | Implement config validation on startup | FR-223 | [x] |

### 5.3 Authentication

| Task | Description | FR | Status |
|------|-------------|----|----|
| 5.3.1 | Reuse WorkspaceClient OAuth for KA endpoint auth | FR-225 | [x] |
| 5.3.2 | Handle unavailable endpoint gracefully (continue with other tools) | FR-226 | [x] |

### 5.4 Tool Registration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 5.4.1 | Create tool factory for KA endpoints | FR-221 | [x] |
| 5.4.2 | Generate tool names as `retrieve_<endpoint_name>` | FR-221 | [x] |
| 5.4.3 | Register KA tools in `ToolRegistry` on startup | FR-245 | [x] |
| 5.4.4 | Add to `deep_research.tools` entry point | FR-245 | [x] |

### 5.5 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 5.5.1 | Write unit tests for `KnowledgeAssistantTool` | - | [x] |
| 5.5.2 | Write unit tests for citation preservation | - | [x] |
| 5.5.3 | Write integration test with real KA endpoint (if available) | - | [x] |
| 5.5.4 | Create SC-002 acceptance test: enable KA via config only | SC-002 | [x] |

---

## Phase 6: User Story 8 - Customize Agent Pipeline in Child Project (P1)

**Depends on**: Phase 1 (Plugin System)

### 6.1 Pipeline Configuration Models - Week 4

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.1.1 | Create `src/deep_research/agent/pipeline/` directory | FR-290 | [x] |
| 6.1.2 | Implement `AgentType` enum in `config.py` | FR-291 | [x] |
| 6.1.3 | Implement `AgentConfig` dataclass in `config.py` | FR-291 | [x] |
| 6.1.4 | Implement `PipelineConfig` dataclass in `config.py` | FR-290 | [x] |
| 6.1.5 | Add `get_agent_config()` method to `PipelineConfig` | FR-290 | [x] |
| 6.1.6 | Add `validate()` method to `PipelineConfig` | FR-298 | [x] |

### 6.2 Default Pipelines

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.2.1 | Create `defaults.py` module | FR-297 | [x] |
| 6.2.2 | Implement `DEFAULT_DEEP_RESEARCH_PIPELINE` (5-agent) | FR-297 | [x] |
| 6.2.3 | Implement `SIMPLE_RESEARCH_PIPELINE` (3-agent) | FR-297 | [x] |
| 6.2.4 | Implement `REACT_LOOP_PIPELINE` (sapresalesbot pattern) | FR-297 | [x] |
| 6.2.5 | Add pipeline selection from config | FR-297 | [x] |

### 6.3 Pipeline Executor

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.3.1 | Create `executor.py` module | FR-292 | [x] |
| 6.3.2 | Implement `PipelineExecutor` class | FR-292 | [x] |
| 6.3.3 | Implement agent execution loop | FR-292 | [x] |
| 6.3.4 | Implement `next_on_success` transition handling | FR-291 | [x] |
| 6.3.5 | Implement `next_on_failure` transition handling | FR-291 | [x] |
| 6.3.6 | Implement `loop_condition` evaluation | FR-291 | [x] |
| 6.3.7 | Implement `loop_back_to` navigation | FR-291 | [x] |
| 6.3.8 | Implement iteration limit enforcement | FR-290 | [x] |
| 6.3.9 | Implement timeout enforcement | FR-290 | [x] |

### 6.4 Pipeline Customization Protocols

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.4.1 | Create `src/deep_research/plugins/pipeline.py` | FR-293 | [x] |
| 6.4.2 | Implement `PhaseInsertion` dataclass | FR-294 | [x] |
| 6.4.3 | Implement `PipelineCustomization` dataclass | FR-295 | [x] |
| 6.4.4 | Implement `CustomPhase` protocol | FR-299 | [x] |
| 6.4.5 | Implement `PipelineCustomizer` protocol | FR-293 | [x] |
| 6.4.6 | Implement `PhaseProvider` protocol | FR-294 | [x] |

### 6.5 Phase Insertion Support

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.5.1 | Create `phase.py` module | FR-294 | [x] |
| 6.5.2 | Implement phase insertion before/after agents | FR-294 | [x] |
| 6.5.3 | Update `PipelineExecutor` to call custom phases | FR-299 | [x] |
| 6.5.4 | Pass `ResearchContext` and `ResearchState` to phases | FR-299 | [x] |

### 6.6 Orchestrator Integration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.6.1 | Refactor `orchestrator.py` to use `PipelineExecutor` | FR-292 | [x] |
| 6.6.2 | Integrate `PluginManager` for pipeline customization | FR-293 | [x] |
| 6.6.3 | Apply `agent_overrides` from customization | FR-296 | [x] |
| 6.6.4 | Apply `disabled_agents` from customization | FR-295 | [x] |
| 6.6.5 | Validate pipeline at startup, fail with clear error | FR-298 | [x] |

### 6.7 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 6.7.1 | Write unit tests for `PipelineConfig` validation | - | [x] |
| 6.7.2 | Write unit tests for `PipelineExecutor` transitions | - | [x] |
| 6.7.3 | Write unit tests for `PipelineExecutor` loop conditions | - | [x] |
| 6.7.4 | Write unit tests for phase insertion | - | [x] |
| 6.7.5 | Write integration test: custom 3-agent pipeline | SC-012 | [x] |
| 6.7.6 | Write integration test: disabled agents | - | [x] |

---

## Phase 7: User Story 9 - Define Custom Output Types in Child Project (P1)

**Depends on**: Phase 1 (Plugin System)

### 7.1 Output Type Infrastructure - Week 5

| Task | Description | FR | Status |
|------|-------------|----|----|
| 7.1.1 | Create `src/deep_research/output/` directory | FR-300 | [x] |
| 7.1.2 | Implement `SynthesisReport` default output type in `base.py` | FR-305 | [x] |
| 7.1.3 | Implement `SynthesizerConfig` dataclass | FR-300 | [x] |
| 7.1.4 | Implement `OutputTypeRegistry` class in `registry.py` | FR-300 | [x] |
| 7.1.5 | Add `register()`, `get_schema()`, `get_provider()` methods | FR-300 | [x] |

### 7.2 Output Type Provider Protocol

| Task | Description | FR | Status |
|------|-------------|----|----|
| 7.2.1 | Create `src/deep_research/plugins/output.py` | FR-300 | [x] |
| 7.2.2 | Implement `OutputTypeProvider` protocol | FR-300 | [x] |
| 7.2.3 | Add `get_output_schema()` method | FR-301 | [x] |
| 7.2.4 | Add `get_synthesizer_config()` method | FR-300 | [x] |
| 7.2.5 | Add `get_synthesizer_prompt()` method | FR-303 | [x] |

### 7.3 Synthesizer Integration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 7.3.1 | Update synthesizer agent to accept `OutputTypeProvider` | FR-302 | [x] |
| 7.3.2 | Pass custom output schema to LLM for structured generation | FR-302 | [x] |
| 7.3.3 | Apply custom synthesizer prompt from provider | FR-303 | [x] |
| 7.3.4 | Validate output against custom schema | FR-301 | [x] |
| 7.3.5 | Ensure custom outputs include sources for citation tracking | FR-304 | [x] |

### 7.4 Plugin Manager Integration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 7.4.1 | Collect `OutputTypeProvider` from loaded plugins | FR-300 | [x] |
| 7.4.2 | Register custom output types in `OutputTypeRegistry` | FR-300 | [x] |
| 7.4.3 | Pass output type to synthesizer based on plugin config | FR-302 | [x] |

### 7.5 Example Output Type

| Task | Description | FR | Status |
|------|-------------|----|----|
| 7.5.1 | Document `MeetingPrepOutput` example in quickstart | - | [x] |
| 7.5.2 | Create example `OutputTypeProvider` implementation | - | [x] |

### 7.6 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 7.6.1 | Write unit tests for `OutputTypeRegistry` | - | [x] |
| 7.6.2 | Write unit tests for custom schema validation | - | [x] |
| 7.6.3 | Write integration test: custom output type generation | SC-013 | [x] |

---

## Phase 8: User Story 6 - Deploy Child Project Using Parent Utilities (P1)

**Depends on**: US5

### 8.1 Lakebase Utilities - Week 7

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.1.1 | Create `src/deep_research/deployment/` directory | FR-310 | [x] |
| 8.1.2 | Create `lakebase.py` module | FR-311 | [x] |
| 8.1.3 | Implement `wait_for_lakebase()` async function | FR-311 | [x] |
| 8.1.4 | Implement health check using `generate_database_credential()` | FR-311 | [x] |
| 8.1.5 | Add CLI entry point: `python -m deep_research.deployment.lakebase wait` | FR-317 | [x] |

### 8.2 Database Utilities

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.2.1 | Create `database.py` module | FR-312 | [x] |
| 8.2.2 | Implement `create_database()` function | FR-312 | [x] |
| 8.2.3 | Implement `ensure_exists()` function | FR-312 | [x] |
| 8.2.4 | Add CLI entry point: `python -m deep_research.deployment.database create` | FR-317 | [x] |

### 8.3 Migration Utilities

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.3.1 | Create `migrations.py` module | FR-313 | [x] |
| 8.3.2 | Implement `run_migrations()` with multi-path `version_locations` support | FR-313 | [x] |
| 8.3.3 | Support running parent + child migrations in sequence | FR-252 | [x] |
| 8.3.4 | Add CLI entry point: `python -m deep_research.deployment.migrations run` | FR-317 | [x] |
| 8.3.5 | Register `deep-research-migrate` script | FR-316 | [x] |

### 8.4 Permissions Utilities

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.4.1 | Create `permissions.py` module | FR-314 | [x] |
| 8.4.2 | Implement `grant_to_app()` function | FR-314 | [x] |
| 8.4.3 | Support table and sequence permissions | FR-314 | [x] |
| 8.4.4 | Add CLI entry point: `python -m deep_research.deployment.permissions grant` | FR-317 | [x] |

### 8.5 App Runner

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.5.1 | Create `app_runner.py` module | FR-315 | [x] |
| 8.5.2 | Implement `run()` function with graceful shutdown | FR-315 | [x] |
| 8.5.3 | Add optional auto-migration flag | FR-315 | [x] |
| 8.5.4 | Register `deep-research-run` script in pyproject.toml | FR-315 | [x] |

### 8.6 App Factory

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.6.1 | Create `src/deep_research/core/factory.py` | FR-318 | [x] |
| 8.6.2 | Implement `create_app()` factory function | FR-318 | [x] |
| 8.6.3 | Support `config_path` parameter | FR-318 | [x] |
| 8.6.4 | Support `plugins` parameter for explicit registration | FR-318 | [x] |
| 8.6.5 | Support `pipeline` parameter for custom pipeline | FR-318 | [x] |
| 8.6.6 | Configure FastAPI lifecycle hooks for plugin init/shutdown | FR-318 | [x] |

### 8.7 CLI Module

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.7.1 | Create `cli.py` with argument parsing | FR-317 | [x] |
| 8.7.2 | Add `--instance`, `--database`, `--timeout` common args | FR-254 | [x] |
| 8.7.3 | Add help text for all commands | FR-254 | [x] |

### 8.8 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 8.8.1 | Write unit tests for `wait_for_lakebase()` | - | [x] |
| 8.8.2 | Write unit tests for `run_migrations()` multi-path | - | [x] |
| 8.8.3 | Write unit tests for `create_app()` factory | - | [x] |
| 8.8.4 | Create SC-008 acceptance test: deploy using Python utilities | SC-008 | [x] |

---

## Phase 9: User Story 7 - Multi-Source Citation Attribution (P2)

**Depends on**: US1, US2

### 9.1 Source Model Extension - Week 9

| Task | Description | FR | Status |
|------|-------------|----|----|
| 9.1.1 | Create `SourceType` enum in `models/source.py` | FR-260 | [x] |
| 9.1.2 | Add `source_type` field to `Source` model | FR-260 | [x] |
| 9.1.3 | Add `source_metadata` JSONB field to `Source` model | FR-263 | [x] |
| 9.1.4 | Update Pydantic schema for `Source` | FR-260 | [x] |

### 9.2 Database Migration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 9.2.1 | Create migration `010_source_type_field.py` | FR-260 | [x] |
| 9.2.2 | Add `source_type` column with default `'web'` | FR-260 | [x] |
| 9.2.3 | Add `source_metadata` JSONB column | FR-263 | [x] |
| 9.2.4 | Test migration up/down | FR-260 | [x] |

### 9.3 Evidence Selector Update

| Task | Description | FR | Status |
|------|-------------|----|----|
| 9.3.1 | Update evidence selector to handle mixed source types | FR-261 | [x] |
| 9.3.2 | Include `source_type` in evidence selection decisions | FR-261 | [x] |
| 9.3.3 | Support custom source types via `custom` value | FR-264 | [x] |

### 9.4 Citation Pipeline Update

| Task | Description | FR | Status |
|------|-------------|----|----|
| 9.4.1 | Update citation generator to include source type | FR-262 | [x] |
| 9.4.2 | Include source-specific metadata in citations | FR-263 | [x] |
| 9.4.3 | Update tool results to populate source_type and metadata | FR-214, FR-227 | [x] |

### 9.5 Frontend Update

| Task | Description | FR | Status |
|------|-------------|----|----|
| 9.5.1 | Update `Source` TypeScript interface with `sourceType` | FR-262 | [x] |
| 9.5.2 | Update `Source` interface with `sourceMetadata` | FR-263 | [x] |
| 9.5.3 | Create `SourceBadge` component for visual indicator | FR-262 | [x] |
| 9.5.4 | Display source type in citation cards | FR-262 | [x] |

### 9.6 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 9.6.1 | Write unit tests for source type handling | - | [x] |
| 9.6.2 | Write integration test for multi-source citations | - | [x] |
| 9.6.3 | Create SC-005 acceptance test: citations show source type badges | SC-005 | [x] |

---

## Phase 10: User Story 4 - Extend Parent UI in Child Project (P1)

**Depends on**: US5

### 10.1 Frontend Package Restructure - Week 11

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.1.1 | Create `frontend/src/core/` directory | FR-280 | [x] |
| 10.1.2 | Move exportable components to `core/components/` | FR-281 | [x] |
| 10.1.3 | Move exportable hooks to `core/hooks/` | FR-281 | [x] |
| 10.1.4 | Create re-exports in `src/components/` and `src/hooks/` | FR-281 | [x] |
| 10.1.5 | Create `core/index.ts` with public exports | FR-280 | [x] |

### 10.2 Component Registry

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.2.1 | Create `frontend/src/core/plugins/` directory | FR-282 | [x] |
| 10.2.2 | Implement `ComponentRegistry` class in `registry.ts` | FR-282 | [x] |
| 10.2.3 | Implement `registerRenderer()` for output types | FR-282 | [x] |
| 10.2.4 | Implement `registerPanel()` for custom panels | FR-283 | [x] |
| 10.2.5 | Implement `getRenderer()` and `getPanel()` lookups | FR-287 | [x] |
| 10.2.6 | Create `types.ts` with registry interfaces | FR-282 | [x] |

### 10.3 Output Type Renderers

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.3.1 | Create `outputTypes.ts` for output renderer registry | FR-306 | [x] |
| 10.3.2 | Register default `SynthesisReport` renderer | FR-306 | [x] |
| 10.3.3 | Implement fallback to generic JSON/markdown renderer | FR-306 | [x] |
| 10.3.4 | Auto-select custom renderer for matching output types | FR-287 | [x] |

### 10.4 Panel Registry

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.4.1 | Create panel slot system for extensible UI | FR-283 | [x] |
| 10.4.2 | Support child project adding panels without parent modification | FR-285 | [x] |

### 10.5 Event Label Registry

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.5.1 | Create event label registry for activity feed customization | FR-286 | [x] |
| 10.5.2 | Support child project customizing event labels | FR-286 | [x] |

### 10.6 Package Exports

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.6.1 | Update `package.json` with `exports` field | FR-280 | [x] |
| 10.6.2 | Configure `@deep-research/core` export path | FR-280 | [x] |
| 10.6.3 | Configure `@deep-research/core/plugins` export path | FR-282 | [x] |
| 10.6.4 | Configure `@deep-research/core/components` export path | FR-281 | [x] |

### 10.7 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 10.7.1 | Write unit tests for `ComponentRegistry` | - | [x] |
| 10.7.2 | Write test: child imports parent components | FR-281 | [x] |
| 10.7.3 | Write test: child registers custom renderer | SC-010 | [x] |
| 10.7.4 | Write test: child frontend builds successfully | FR-284 | [x] |

---

## Phase 11: User Story 10 - Handle Follow-up Conversations in Child Project (P2)

**Depends on**: US8 (Pipeline)

### 11.1 Conversation Infrastructure - Week 8

| Task | Description | FR | Status |
|------|-------------|----|----|
| 11.1.1 | Create `src/deep_research/conversation/` directory | FR-320 | [x] |
| 11.1.2 | Implement `FollowUpIntent` enum in `intent.py` | FR-321 | [x] |
| 11.1.3 | Implement `IntentClassification` dataclass in `intent.py` | FR-326 | [x] |
| 11.1.4 | Implement `AnswerResponse`, `UpdateResponse`, `ClarificationResponse` | FR-320 | [x] |

### 11.2 Conversation Protocols

| Task | Description | FR | Status |
|------|-------------|----|----|
| 11.2.1 | Create `src/deep_research/plugins/conversation.py` | FR-320 | [x] |
| 11.2.2 | Implement `IntentClassifier` protocol | FR-321 | [x] |
| 11.2.3 | Implement `QAHandler` protocol | FR-323 | [x] |
| 11.2.4 | Implement `UpdateHandler` protocol with `MAX_TOOL_CALLS` | FR-324 | [x] |
| 11.2.5 | Implement `ConversationProvider` protocol | FR-320 | [x] |

### 11.3 Default Handlers

| Task | Description | FR | Status |
|------|-------------|----|----|
| 11.3.1 | Create `default.py` module | FR-322 | [x] |
| 11.3.2 | Implement `DefaultIntentClassifier` using LLM | FR-321 | [x] |
| 11.3.3 | Implement `DefaultQAHandler` for read-only answers | FR-322 | [x] |
| 11.3.4 | Implement `DefaultUpdateHandler` with limited tool calls | FR-322 | [x] |

### 11.4 Conversation Handler Integration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 11.4.1 | Create `handler.py` with main conversation routing | FR-320 | [x] |
| 11.4.2 | Pass current state, sources, and context to handlers | FR-325 | [x] |
| 11.4.3 | Route based on intent classification | FR-321 | [x] |
| 11.4.4 | Integrate custom handlers from `ConversationProvider` | FR-320 | [x] |

### 11.5 Plugin Manager Integration

| Task | Description | FR | Status |
|------|-------------|----|----|
| 11.5.1 | Collect `ConversationProvider` from loaded plugins | FR-320 | [x] |
| 11.5.2 | Use custom classifiers/handlers when provided | FR-320 | [x] |
| 11.5.3 | Fall back to default handlers when not provided | FR-322 | [x] |

### 11.6 Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 11.6.1 | Write unit tests for `DefaultIntentClassifier` | - | [x] |
| 11.6.2 | Write unit tests for `DefaultQAHandler` | - | [x] |
| 11.6.3 | Write unit tests for `DefaultUpdateHandler` | - | [x] |
| 11.6.4 | Write integration test: custom conversation provider | SC-014 | [x] |

---

## Phase 12: Polish and Validation - Week 12

### 12.1 sapresalesbot Validation

| Task | Description | FR | Status |
|------|-------------|----|----|
| 12.1.1 | Create `examples/sales-research-agent/` structure | SC-015 | [x] |
| 12.1.2 | Implement custom pipeline (ReAct loop) | SC-015 | [x] |
| 12.1.3 | Implement `MeetingPrepOutput` custom output type | SC-015 | [x] |
| 12.1.4 | Implement custom conversation handlers | SC-015 | [x] |
| 12.1.5 | Verify deployment using parent utilities | SC-015 | [x] |
| 12.1.6 | Document sales-research-agent as reference implementation | SC-015 | [x] |

### 12.2 Integration Tests

| Task | Description | FR | Status |
|------|-------------|----|----|
| 12.2.1 | Write full E2E test: child project deployment | SC-009 | [x] |
| 12.2.2 | Write test: plugin lifecycle with real plugins | - | [x] |
| 12.2.3 | Write test: custom pipeline execution | - | [x] |
| 12.2.4 | Write test: custom output type rendering | - | [x] |

### 12.3 Performance Validation

| Task | Description | FR | Status |
|------|-------------|----|----|
| 12.3.1 | Benchmark tool registry lookup time (<1ms) | - | [x] |
| 12.3.2 | Benchmark pipeline execution overhead | - | [x] |
| 12.3.3 | Verify no regression in research pipeline performance | SC-007 | [x] |

### 12.4 Documentation

| Task | Description | FR | Status |
|------|-------------|----|----|
| 12.4.1 | Update `CLAUDE.md` with plugin architecture section | - | [x] |
| 12.4.2 | Update `quickstart.md` with all examples | - | [x] |
| 12.4.3 | Create migration guide for existing deployments | - | [x] |
| 12.4.4 | Document all configuration options | - | [x] |
| 12.4.5 | Review and update all docstrings | - | [x] |

### 12.5 Final Validation

| Task | Description | FR | Status |
|------|-------------|----|----|
| 12.5.1 | Run full test suite (unit, integration, E2E) | - | [x] |
| 12.5.2 | Verify all Success Criteria (SC-001 through SC-015) | - | [x] |
| 12.5.3 | Code review and merge preparation | - | [x] |

---

## Summary

### Task Counts by Phase

| Phase | Description | Tasks |
|-------|-------------|-------|
| 0 | Setup | 6 |
| 1 | Foundational | 41 |
| 2 | US5: Package Core | 16 |
| 3 | US3: Child Project | 18 |
| 4 | US1: Vector Search | 18 |
| 5 | US2: Knowledge Assistant | 17 |
| 6 | US8: Pipeline Customization | 28 |
| 7 | US9: Output Types | 17 |
| 8 | US6: Deployment | 26 |
| 9 | US7: Multi-Source Citations | 17 |
| 10 | US4: Frontend Extension | 22 |
| 11 | US10: Conversation | 18 |
| 12 | Polish | 16 |
| **Total** | | **~260** |

### Parallel Opportunities

Tasks that can run in parallel within the same phase:

- **Phase 1**: Tool infrastructure (1.2) and Plugin system (1.3) can start after package restructure (1.1)
- **Phase 4 & 5**: Vector Search and Knowledge Assistant can proceed in parallel
- **Phase 6 & 7**: Pipeline and Output Types can proceed in parallel
- **Phase 9 & 10**: Citations (backend) and Frontend Extension can proceed in parallel
- **Phase 10 & 11**: Frontend work and Conversation handling are independent

### Success Criteria Coverage

| Criteria | Tasks |
|----------|-------|
| SC-001 | 4.5.4 |
| SC-002 | 5.5.4 |
| SC-003 | 3.6.1, 3.6.2, 3.6.3 |
| SC-004 | 2.3.5 |
| SC-005 | 9.6.3 |
| SC-006 | 3.6.4 |
| SC-007 | 12.3.3 |
| SC-008 | 8.8.4 |
| SC-009 | 12.2.1 |
| SC-010 | 10.7.3 |
| SC-011 | 3.2.3 (implicit) |
| SC-012 | 6.7.5 |
| SC-013 | 7.6.3 |
| SC-014 | 11.6.4 |
| SC-015 | 12.1.1-12.1.6 |
