# Feature Specification: Plugin Architecture

**Feature Branch**: `005-plugin-architecture`
**Created**: 2026-01-17
**Status**: Draft
**Input**: User description: "Plugin architecture for extensible tools and data sources enabling external plugins to extend the core research agent with Vector Search endpoints, Knowledge Assistant endpoints, declarative pipeline configuration, custom output types, and full child application support"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Built-in Vector Search Tool (Priority: P1)

An administrator wants to enable Vector Search as a data source for research queries without writing any plugin code. They configure the endpoint details in `app.yaml` and the research agent automatically includes this as a searchable knowledge source.

**Why this priority**: Enables immediate value from internal knowledge bases (product docs, case studies) without any code changes. Most users will start here before building custom plugins.

**Independent Test**: Can be fully tested by adding a Vector Search configuration to `app.yaml`, restarting the application, and verifying the researcher agent can query the configured index.

**Acceptance Scenarios**:

1. **Given** an administrator has a Vector Search endpoint configured in `app.yaml` with `vector_search.enabled: true`, **When** a user submits a research query, **Then** the researcher agent has access to a `search_<name>` tool that queries the configured Vector Search index.

2. **Given** Vector Search is enabled but the endpoint configuration is invalid, **When** the application starts, **Then** startup fails with a clear error message identifying the misconfiguration.

3. **Given** a Vector Search query returns results, **When** the researcher includes this evidence in the synthesis, **Then** the source is attributed with `source_type: vector_search` and includes metadata (index name, relevance score).

---

### User Story 2 - Configure Built-in Knowledge Assistant Retrieval Tool (Priority: P1)

An administrator wants to enable a Databricks Knowledge Assistant endpoint as an optimized retrieval tool during research plan execution. Knowledge Assistants are sophisticated RAG-like retrieval systems with a chat interface that can retrieve and synthesize information from curated knowledge bases. The researcher agent uses KA as a retrieval tool alongside Vector Search and web search when executing research steps.

**Why this priority**: Knowledge Assistants provide optimized, enterprise-grade retrieval with built-in RAG-like capabilities. They serve as pre-configured retrieval endpoints that return high-quality, contextual information from curated sources. Equal priority to Vector Search as both are core retrieval tool integrations used during research execution.

**Independent Test**: Can be fully tested by adding a Knowledge Assistant configuration to `app.yaml`, restarting the application, and verifying the researcher agent can retrieve information from the configured KA endpoint during research plan execution.

**Acceptance Scenarios**:

1. **Given** an administrator has a Knowledge Assistant endpoint configured in `app.yaml` with `knowledge_assistants.enabled: true`, **When** the researcher executes a research step, **Then** the researcher can use a `retrieve_<name>` tool to query the Knowledge Assistant for relevant information.

2. **Given** a research plan includes steps that benefit from internal knowledge retrieval, **When** the researcher executes those steps, **Then** the researcher can choose between web_search, Vector Search, and Knowledge Assistant retrieval tools based on the step's information needs.

3. **Given** Knowledge Assistant is enabled but the endpoint is unavailable, **When** the researcher attempts to use the retrieval tool, **Then** the tool returns a graceful error and research continues with other available retrieval sources (web search, Vector Search).

4. **Given** a Knowledge Assistant retrieval returns information with source citations from its knowledge base, **When** the researcher includes this evidence in the synthesis, **Then** the original source citations are preserved and attributed to the KA retrieval source.

5. **Given** the researcher is executing a research step, **When** internal knowledge is more relevant than web sources, **Then** the researcher can use KA retrieval to get curated, verified information instead of or in addition to web search.

---

### User Story 3 - Build a Child Research Agent from Parent Package (Priority: P1)

A developer wants to create a new, specialized deep research agent (the "child") by reusing the core functionality of this project (the "parent"). The child installs the parent as a pip dependency, adds custom tools and prompts via the plugin system, and deploys as an independent Databricks App with its own branding and configuration.

**Why this priority**: This is the primary extensibility use case. The plugin architecture exists specifically to enable building derivative research agents that inherit core capabilities while adding domain-specific functionality. Equal priority to Vector Search/KA as it defines the core value proposition.

**Independent Test**: Can be fully tested by creating a minimal child project that: (1) declares `databricks-deep-research` as a pip dependency, (2) registers a custom tool via the plugin entry point, (3) runs the parent's orchestrator with the custom tool available, and (4) deploys to Databricks Apps.

**Acceptance Scenarios**:

1. **Given** a child project declares `databricks-deep-research` as a dependency in its `pyproject.toml`, **When** the child project installs dependencies, **Then** all parent functionality (orchestrator, agents, tools, services) is available for import.

2. **Given** a child project implements a plugin that provides custom tools and prompts, **When** the child application starts, **Then** the plugin is discovered and the custom tools are available alongside core tools (web_search, web_crawl, vector_search, knowledge_assistant).

3. **Given** a child project provides its own `app.yaml` configuration, **When** the child application starts, **Then** the child's configuration overrides parent defaults for models, endpoints, and agent settings.

4. **Given** a child project creates its own Databricks Apps bundle, **When** the child is deployed, **Then** it runs as an independent application with its own URL, service principal, and database.

5. **Given** a child project's plugin fails during initialization, **When** the child application starts, **Then** the plugin error is logged but the application continues with core functionality intact.

6. **Given** a child project wants to customize agent prompts for its domain, **When** the child's plugin provides prompt overrides, **Then** the customized prompts are used by the corresponding agents during research execution.

---

### User Story 4 - Extend Parent UI in Child Project (Priority: P1)

A developer building a child research agent wants to reuse and extend the parent's frontend UI. The child project can import parent UI components, add custom panels or renderers, and build a customized frontend that deploys alongside the child backend.

**Why this priority**: Complete child applications need both backend and frontend customization. Depends on P1 backend packaging being in place. P2 because frontend extension is secondary to core backend extensibility.

**Independent Test**: Can be fully tested by creating a child frontend that imports parent components, adds a custom panel, builds successfully, and renders the custom UI when deployed.

**Acceptance Scenarios**:

1. **Given** the parent frontend is published as an importable package, **When** a child project imports parent components, **Then** the imports succeed and components render correctly.

2. **Given** a child project wants to add custom UI panels (e.g., domain-specific visualizations), **When** the child registers panel components, **Then** the custom panels appear in the child's UI alongside parent panels.

3. **Given** a child project wants to customize how certain outputs are rendered, **When** the child registers custom output renderers, **Then** the custom renderers are used for matching output types.

4. **Given** a child project builds its frontend, **When** the build completes, **Then** the output includes both parent components and child customizations in a single deployable bundle.

5. **Given** a child application is deployed to Databricks Apps, **When** a user accesses the child's URL, **Then** they see the child's customized UI (not the parent's default UI).

---

### User Story 5 - Package Core as pip-installable Library (Priority: P1)

A developer wants to install `databricks-deep-research` as a pip package to use as the foundation for building child research agents or developing plugins. This is the foundational capability that enables all parent-child extensibility.

**Why this priority**: Foundation requirement for User Stories 3 and 4. Without proper pip packaging, child projects cannot depend on the parent. Must be completed first.

**Independent Test**: Can be fully tested by running `pip install databricks-deep-research` in a clean virtual environment and importing core modules.

**Acceptance Scenarios**:

1. **Given** the package is published (or installed in editable mode), **When** a developer runs `pip install databricks-deep-research`, **Then** the package installs successfully with all dependencies.

2. **Given** the package is installed, **When** a developer imports `from deep_research.agent.orchestrator import run_research`, **Then** the import succeeds and the orchestrator is callable.

3. **Given** the package is installed, **When** a developer imports `from deep_research.plugins import ResearchPlugin, ToolProvider`, **Then** the plugin protocols are available for implementing custom plugins.

4. **Given** the package is installed, **When** a child project lists entry points for `deep_research.tools`, **Then** the core tools (web_search, web_crawl, vector_search, knowledge_assistant) are discoverable.

5. **Given** the package is installed, **When** a developer imports `from deep_research.services.llm import LLMClient`, **Then** the LLM client is available for direct use in custom components.

---

### User Story 6 - Deploy Child Project Using Parent Utilities (Priority: P1)

A developer building a child research agent wants to deploy their child project to Databricks Apps using deployment utilities provided by the parent package. The parent provides Python modules for common deployment tasks (wait for Lakebase, create database, run migrations, grant permissions) that child projects can use directly instead of copying shell scripts.

**Why this priority**: Essential for child projects to be deployable. Without deployment utilities, child projects would need to duplicate complex deployment logic. Elevated from P2 to P1 because reliable deployment is critical for child project success.

**Independent Test**: Can be fully tested by creating a minimal child project, calling the parent's deployment utilities, and successfully deploying the child to Databricks Apps with its own database and service principal.

**Acceptance Scenarios**:

1. **Given** a child project has been created with parent as a dependency, **When** the developer uses `python -m deep_research.deployment.lakebase wait`, **Then** the command waits for the Lakebase instance to be ready.

2. **Given** a child project needs to create its database, **When** the developer uses `python -m deep_research.deployment.database create`, **Then** the database is created on the Lakebase instance.

3. **Given** a child project needs to run database migrations, **When** the developer uses `python -m deep_research.deployment.migrations run`, **Then** both parent migrations (from the installed package) and child-specific migrations run in the correct order.

4. **Given** a child project is deployed, **When** the developer uses `python -m deep_research.deployment.permissions grant`, **Then** the app's service principal receives appropriate table permissions.

5. **Given** a child project wants to run in production, **When** the developer uses `deep-research-run --app myapp:app`, **Then** the app starts with graceful shutdown handling and optional auto-migration.

6. **Given** the parent updates its deployment utilities in a new version, **When** the child updates its parent dependency, **Then** the child automatically benefits from deployment improvements without copying scripts.

---

### User Story 7 - Multi-Source Citation Attribution (Priority: P2)

A user wants to see clear attribution for where evidence came from in a research report, distinguishing between web sources, Vector Search results, and Knowledge Assistant responses.

**Why this priority**: Critical for trust and transparency. Users need to understand the provenance of information in research outputs.

**Independent Test**: Can be fully tested by running a research query that uses both web search and Vector Search, then verifying the final report shows distinct source type indicators.

**Acceptance Scenarios**:

1. **Given** a research report includes evidence from multiple source types, **When** the report is displayed, **Then** each citation shows a source type indicator (web, vector_search, knowledge_assistant).

2. **Given** a source comes from Vector Search, **When** the citation is displayed, **Then** it includes the index name and relevance score as metadata.

3. **Given** a source comes from a Knowledge Assistant, **When** the citation is displayed, **Then** it includes the assistant name and any citations the assistant provided.

---

### User Story 8 - Customize Agent Pipeline in Child Project (Priority: P1)

A developer building a child research agent needs a completely different agent architecture than the parent's 5-agent pipeline. The child wants to use a simpler ReAct loop, add custom phases, skip certain agents, or define entirely new pipelines. The plugin architecture must support full pipeline customization, not just tool/prompt additions.

**Why this priority**: Many child projects (like sapresalesbot) need fundamentally different agent architectures. Without pipeline customization, they cannot be built as child projects and must fork the parent. Critical for the "parent-child" value proposition.

**Independent Test**: Can be fully tested by creating a child plugin that defines a custom pipeline with 3 agents (instead of 5) and verifying the child application uses the custom pipeline.

**Acceptance Scenarios**:

1. **Given** a child project implements `PipelineCustomizer` protocol, **When** the child application runs a research query, **Then** the custom pipeline configuration is used instead of the default 5-agent pipeline.

2. **Given** a child plugin provides `PhaseProvider` with custom phases, **When** the pipeline executes, **Then** the custom phases run at the specified points (e.g., after background, before synthesis).

3. **Given** a child project wants to skip certain agents (e.g., background investigator), **When** the child configures `disabled_agents`, **Then** those agents are skipped during pipeline execution.

4. **Given** a child project wants to modify agent configurations (e.g., researcher max_tool_calls), **When** the child provides `agent_overrides`, **Then** the specified agents use the overridden configuration.

5. **Given** a child project defines a simpler pipeline (e.g., Coordinator → Researcher → Synthesizer), **When** the child application runs, **Then** only the defined agents execute in the specified order.

6. **Given** a child project needs conditional transitions between agents, **When** the child configures `next_on_success` and `loop_condition`, **Then** the pipeline executor follows the conditional logic.

---

### User Story 9 - Define Custom Output Types in Child Project (Priority: P1)

A developer building a child research agent needs to produce domain-specific structured outputs (e.g., `MeetingPrepOutput` with MEDDPICC questions, attendee briefs, case studies) instead of the parent's generic synthesis report. The plugin architecture must support custom output type definitions.

**Why this priority**: Different research domains require different output structures. A sales meeting prep bot produces different outputs than a technical research bot. Without custom output types, child projects cannot deliver domain-appropriate value.

**Independent Test**: Can be fully tested by creating a child plugin that defines `MeetingPrepOutput` schema and verifying the synthesizer produces output matching that schema.

**Acceptance Scenarios**:

1. **Given** a child plugin implements `OutputTypeProvider` protocol, **When** the synthesizer agent runs, **Then** it uses the custom output schema instead of the default synthesis format.

2. **Given** a child plugin provides a Pydantic model for custom output, **When** the synthesizer generates output, **Then** the output validates against the custom schema.

3. **Given** a child plugin provides synthesizer prompt customization for the output type, **When** synthesis runs, **Then** the custom prompt guides the LLM to produce correctly structured output.

4. **Given** a child project defines output-specific UI renderers, **When** the output is displayed, **Then** the custom renderer is used for the custom output type.

5. **Given** a custom output type includes source references, **When** the output is generated, **Then** the citation pipeline correctly processes and attributes sources.

---

### User Story 10 - Handle Follow-up Conversations in Child Project (Priority: P2)

A developer building a child research agent wants to handle follow-up messages after initial research, using a custom conversation architecture (e.g., two-agent QA + Research/Update system with intent classification). The plugin architecture must support conversation handler customization.

**Why this priority**: Sophisticated child applications need conversation handling beyond single-shot research. Intent classification, read-only Q&A, and plan modification are common patterns.

**Independent Test**: Can be fully tested by creating a child plugin that provides a custom conversation handler and verifying follow-up messages are routed correctly.

**Acceptance Scenarios**:

1. **Given** a child plugin implements `ConversationProvider` protocol, **When** a follow-up message arrives, **Then** the custom conversation handler processes it instead of the default.

2. **Given** a child implements custom intent classification, **When** a follow-up message arrives, **Then** it is routed to the appropriate handler (QA vs Research/Update).

3. **Given** a child implements a QA agent for read-only questions, **When** a user asks about the existing plan, **Then** the QA agent answers from context without modifying the plan.

4. **Given** a child implements a Research/Update agent, **When** a user requests changes or additional research, **Then** the agent can modify the plan with limited additional tool calls.

---

### Edge Cases

- What happens when all configured Vector Search endpoints are unavailable during a research query?
  - The researcher gracefully degrades to using only available tools (web_search, web_crawl).

- How does the system handle a plugin that provides tools with the same name as core tools?
  - Core tools take precedence; plugin tools with conflicting names are renamed with a plugin prefix (e.g., `myplugin_web_search`).

- What happens when Vector Search returns results without required metadata columns?
  - The tool logs a warning and includes available columns; missing metadata is marked as "unavailable".

- How does the system handle circular plugin dependencies?
  - Plugins are loaded in registration order without dependency resolution; circular references cause a warning but not failure.

- What happens when a plugin's configuration schema validation fails?
  - The plugin is disabled with an error log; application continues with remaining plugins.

- What happens when a child project uses an incompatible version of the parent package?
  - The child's `pyproject.toml` should specify version constraints; pip installation fails with dependency conflict if incompatible.

- How does a child project handle database migrations from both parent and child?
  - Child includes parent's migration history and can add child-specific migrations; Alembic runs both in sequence.

- What happens when a child project overrides a parent configuration value incorrectly?
  - Configuration validation at startup catches invalid values and fails with a clear error message before the app runs.

- How does the child UI handle updates to parent UI components?
  - Child rebuilds when parent package is updated; breaking changes in parent should follow semver and be documented.

- What happens when a custom pipeline has invalid transitions (e.g., referencing non-existent agent)?
  - Pipeline validation at startup fails with a clear error message identifying the misconfiguration.

- How does the system handle a custom output type that the frontend doesn't have a renderer for?
  - Falls back to a generic JSON/markdown renderer with a warning log.

- What happens when a custom phase throws an exception?
  - The phase error is logged, and the pipeline continues to the next phase (graceful degradation).

## Requirements *(mandatory)*

### Functional Requirements

#### Tool Infrastructure (FR-200 series)

- **FR-200**: System MUST provide a `ResearchTool` protocol with `definition`, `execute()`, and `validate_arguments()` methods that all tools implement.

- **FR-201**: System MUST provide a `ToolRegistry` class that allows dynamic registration and discovery of tools at runtime.

- **FR-202**: System MUST refactor existing core tools (`web_search`, `web_crawl`) to implement the `ResearchTool` protocol.

- **FR-203**: System MUST provide a `ResearchContext` dataclass containing chat_id, user_id, research_type, url_registry, evidence_registry, and plugin_data that is passed to all tool executions.

#### Vector Search Integration (FR-210 series)

- **FR-210**: System MUST provide a `VectorSearchTool` class that queries Databricks Vector Search endpoints using the Databricks SDK.

- **FR-211**: System MUST support configuring multiple Vector Search endpoints via `app.yaml` under the `vector_search` section.

- **FR-212**: Vector Search MUST be disabled by default and require explicit `vector_search.enabled: true` to activate.

- **FR-213**: Each Vector Search endpoint configuration MUST include: name, endpoint, index, description, num_results, and columns_to_return.

- **FR-214**: Vector Search results MUST include source metadata (index name, relevance score) for citation pipeline integration.

- **FR-215**: System MUST authenticate to Vector Search using the existing WorkspaceClient OAuth mechanism.

#### Knowledge Assistant Retrieval Integration (FR-220 series)

- **FR-220**: System MUST provide a `KnowledgeAssistantTool` class that queries Databricks Knowledge Assistant endpoints as an optimized retrieval tool during research plan execution.

- **FR-221**: System MUST support configuring multiple Knowledge Assistant endpoints via `app.yaml` under the `knowledge_assistants` section.

- **FR-222**: Knowledge Assistants MUST be disabled by default and require explicit `knowledge_assistants.enabled: true` to activate.

- **FR-223**: Each Knowledge Assistant endpoint configuration MUST include: name, endpoint_name, description, max_tokens, and temperature.

- **FR-224**: Knowledge Assistant retrieval responses MUST preserve any source citations from the assistant's underlying knowledge base.

- **FR-225**: System MUST authenticate to Knowledge Assistant endpoints using the existing WorkspaceClient OAuth mechanism.

- **FR-226**: The researcher agent MUST be able to select Knowledge Assistant retrieval as a tool option alongside web_search, web_crawl, and Vector Search when executing research steps.

- **FR-227**: Knowledge Assistant retrieval tool MUST return structured results that integrate with the existing evidence registry for citation tracking.

#### Plugin System (FR-230 series)

- **FR-230**: System MUST define a `ResearchPlugin` protocol with `name`, `version`, `initialize()`, and `shutdown()` methods.

- **FR-231**: System MUST define a `ToolProvider` protocol with a `get_tools(context: ResearchContext)` method that returns a list of `ResearchTool` instances.

- **FR-232**: System MUST define a `PromptProvider` protocol with a `get_prompt_overrides(context: ResearchContext)` method that returns prompt customizations.

- **FR-233**: System MUST provide a `PluginManager` class that discovers, loads, initializes, and shuts down plugins.

- **FR-234**: System MUST discover plugins via the `deep_research.plugins` Python entry point group.

- **FR-235**: Plugin initialization failures MUST be logged but MUST NOT prevent application startup.

- **FR-236**: System MUST support plugin-specific configuration via `plugins.<plugin_name>` sections in `app.yaml`.

#### Pipeline Configuration (FR-290 series) - NEW

- **FR-290**: System MUST define a `PipelineConfig` dataclass specifying agent sequence, transitions, and loop conditions.

- **FR-291**: System MUST define an `AgentConfig` dataclass specifying agent_type, enabled, model_tier, next_on_success, next_on_failure, loop_condition, and agent-specific config.

- **FR-292**: System MUST provide a `PipelineExecutor` class that executes agents according to the pipeline configuration with conditional branching.

- **FR-293**: System MUST define a `PipelineCustomizer` protocol that allows plugins to provide custom pipeline configurations.

- **FR-294**: System MUST define a `PhaseProvider` protocol that allows plugins to insert custom phases at specified points in the pipeline.

- **FR-295**: System MUST support disabling specific agents via `disabled_agents` configuration in pipeline customization.

- **FR-296**: System MUST support overriding agent-specific configurations via `agent_overrides` in pipeline customization.

- **FR-297**: System MUST provide default pipeline configurations: `DEFAULT_DEEP_RESEARCH_PIPELINE` (5-agent) and `SIMPLE_RESEARCH_PIPELINE` (3-agent).

- **FR-298**: System MUST validate pipeline configurations at startup and fail with clear error if invalid.

- **FR-299**: Custom phases MUST receive the same `ResearchContext` and `ResearchState` as built-in agents.

#### Output Type Customization (FR-300 series) - NEW

- **FR-300**: System MUST define an `OutputTypeProvider` protocol with `get_output_schema()` and `get_synthesizer_config()` methods.

- **FR-301**: Custom output types MUST be defined as Pydantic models for validation.

- **FR-302**: System MUST pass the custom output schema to the synthesizer agent for structured output generation.

- **FR-303**: System MUST support output-specific prompt customization via the output type provider.

- **FR-304**: Custom output types MUST integrate with the existing citation pipeline for source attribution.

- **FR-305**: System MUST provide a default `SynthesisReport` output type for standard research reports.

- **FR-306**: Frontend MUST support custom output renderers registered via the component registry for custom output types.

#### Deployment Utilities (FR-310 series) - NEW

- **FR-310**: Parent package MUST provide a `deep_research.deployment` Python package with reusable deployment utilities.

- **FR-311**: System MUST provide `deep_research.deployment.lakebase` module with `wait_for_lakebase()` and health check functions.

- **FR-312**: System MUST provide `deep_research.deployment.database` module with `create_database()` and `ensure_exists()` functions.

- **FR-313**: System MUST provide `deep_research.deployment.migrations` module with `run_migrations()` supporting multi-path version_locations.

- **FR-314**: System MUST provide `deep_research.deployment.permissions` module with `grant_to_app()` for service principal permissions.

- **FR-315**: System MUST provide `deep-research-run` CLI command for production app execution with graceful shutdown.

- **FR-316**: System MUST provide `deep-research-migrate` CLI command for running migrations.

- **FR-317**: All deployment utilities MUST be callable via `python -m deep_research.deployment.<module>` with CLI arguments.

- **FR-318**: System MUST provide a `create_app()` factory function for programmatic application instantiation.

#### Conversation Handling (FR-320 series) - NEW

- **FR-320**: System MUST define a `ConversationProvider` protocol for plugins to customize follow-up message handling.

- **FR-321**: System MUST define an `IntentClassifier` protocol for routing follow-up messages to appropriate handlers.

- **FR-322**: Default conversation handling MUST support the existing research flow (no custom handlers required).

- **FR-323**: Child projects MUST be able to provide custom QA agents for read-only context queries.

- **FR-324**: Child projects MUST be able to provide custom Research/Update agents for plan modifications.

- **FR-325**: Custom conversation handlers MUST receive the current state, sources, and recent context for processing.

- **FR-326**: Intent classification results MUST include intent type, confidence score, and extracted parameters.

#### Child Project Extensibility (FR-270 series)

- **FR-270**: Parent package MUST be installable as a pip dependency by child projects via `pip install databricks-deep-research`.

- **FR-271**: Child projects MUST be able to import and use all parent modules (orchestrator, agents, tools, services, models).

- **FR-272**: Child projects MUST be able to provide their own `app.yaml` configuration that overrides parent defaults.

- **FR-273**: Child projects MUST be able to register custom plugins that extend parent functionality with domain-specific tools and prompts.

- **FR-274**: Child projects MUST be able to deploy as independent Databricks Apps with their own URL, service principal, and database.

- **FR-275**: Parent configuration MUST support being overridden by child configuration at multiple levels (endpoints, models, agents, plugins).

- **FR-276**: Child projects MUST be able to reuse parent's database migration infrastructure and add child-specific migrations.

- **FR-277**: Parent MUST export all public APIs needed for child projects to build complete research agent applications.

- **FR-278**: Child projects MUST be able to replace the entire agent pipeline with a custom architecture.

- **FR-279**: Child projects MUST be able to define custom output types for domain-specific structured outputs.

#### Package Structure (FR-240 series)

- **FR-240**: System MUST restructure the package from `src/` to `src/deep_research/` for proper pip packaging.

- **FR-241**: System MUST update all imports to use `from deep_research.xxx import yyy` format.

- **FR-242**: System MUST create a comprehensive `pyproject.toml` with dependencies, entry points, and CLI scripts.

- **FR-243**: System MUST support `pip install -e .` for development installation.

- **FR-244**: System MUST export public APIs via `src/deep_research/__init__.py`.

- **FR-245**: System MUST register core tools via `deep_research.tools` entry point group.

- **FR-246**: System MUST register CLI commands via `[project.scripts]` in pyproject.toml.

#### Child Project Deployment (FR-250 series)

- **FR-250**: Parent MUST provide Python deployment utilities (not just copyable scripts) that child projects can use directly.

- **FR-251**: Parent MUST document the deployment workflow with examples of using deployment utilities.

- **FR-252**: Parent's database migration infrastructure MUST support running both parent migrations (from the installed package) and child-specific migrations.

- **FR-253**: Child projects MUST be able to deploy to Databricks Apps with their own application name, service principal, and database.

- **FR-254**: Deployment utilities MUST accept parameters for app name, database name, instance name, and target workspace.

- **FR-255**: Parent MUST provide `deep-research-run` command that child projects can use in their `app.yaml` command section.

#### Multi-Source Citations (FR-260 series)

- **FR-260**: System MUST add a `source_type` field to the Source model with values: `web`, `vector_search`, `knowledge_assistant`, `custom`.

- **FR-261**: Evidence selector MUST handle mixed source types and include source_type in selection decisions.

- **FR-262**: Citation display MUST show a visual indicator of source type (icon or label).

- **FR-263**: Citation metadata MUST include source-type-specific information (index for Vector Search, assistant name for Knowledge Assistant).

- **FR-264**: Custom source types from plugins MUST be supported via the `custom` source_type value with plugin-defined metadata.

#### Frontend Extensibility (FR-280 series)

- **FR-280**: Parent frontend MUST be publishable as an importable package that child projects can depend on.

- **FR-281**: Child projects MUST be able to import and render parent UI components in their own frontend.

- **FR-282**: System MUST provide a component registry that allows child projects to register custom output renderers.

- **FR-283**: System MUST provide a panel registry that allows child projects to add custom UI panels.

- **FR-284**: Child frontend builds MUST include both parent components and child customizations in a single deployable bundle.

- **FR-285**: Child UI customizations MUST be configurable without modifying parent component source code.

- **FR-286**: System MUST provide an event label registry for child projects to customize activity feed labels.

- **FR-287**: Custom output type renderers MUST be automatically selected when rendering matching output types.

### Key Entities

- **ResearchTool**: An executable capability with a definition (name, description, parameters), execution logic, and argument validation. Tools are provided by core or plugins.

- **ResearchContext**: Contextual information for tool execution including user identity, session state, registries for URLs and evidence, and plugin-provided data.

- **ResearchPlugin**: An external extension that provides tools, prompt customizations, pipeline customizations, or output types. Discovered via Python entry points and configured via `app.yaml`.

- **ToolRegistry**: A runtime collection of available tools, combining core tools with plugin-provided tools. Used by the researcher agent to select and execute tools.

- **PluginManager**: The lifecycle manager for plugins, handling discovery, initialization, tool collection, and shutdown.

- **PipelineConfig**: A declarative specification of the agent pipeline including agent sequence, transitions, loop conditions, and global settings.

- **PipelineExecutor**: The runtime engine that executes agents according to a PipelineConfig, handling conditional branching and phase insertion.

- **PhaseInsertion**: A specification for inserting a custom phase at a specific point in the pipeline (e.g., "after background").

- **OutputTypeProvider**: A plugin protocol that defines custom structured output schemas and synthesizer configuration.

- **ConversationProvider**: A plugin protocol that defines custom follow-up message handling including intent classification and response generation.

- **Source**: Extended to include source_type (web, vector_search, knowledge_assistant, custom) and type-specific metadata for citation attribution.

- **Child Project**: A separate codebase that depends on the parent `databricks-deep-research` package, extends it with custom plugins and UI, and deploys as an independent Databricks App.

- **Parent Package**: The core `databricks-deep-research` pip package that provides orchestrator, agents, tools, services, deployment utilities, and UI components for child projects to build upon.

- **ComponentRegistry**: A frontend registry that maps output types and panel IDs to custom renderers, enabling child projects to extend the UI without modifying parent code.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Administrators can enable Vector Search as a data source with configuration changes only (no code changes required).

- **SC-002**: Administrators can enable Knowledge Assistant as a retrieval tool with configuration changes only (no code changes required).

- **SC-003**: External plugins can be installed via `pip install` and discovered automatically on application startup.

- **SC-004**: Developers can install the core package via `pip install -e .` in a clean environment.

- **SC-005**: Research reports clearly distinguish between evidence from web sources, Vector Search, and Knowledge Assistants.

- **SC-006**: Application startup succeeds even if one or more plugins fail to initialize.

- **SC-007**: All existing functionality continues to work after package restructure (no regressions).

- **SC-008**: Child projects can deploy to Databricks Apps using parent's deployment utilities (`python -m deep_research.deployment.*`).

- **SC-009**: A child project can be created, configured, and deployed as an independent Databricks App within a reasonable development timeframe.

- **SC-010**: Child projects can extend the parent UI with custom panels and renderers without forking the parent codebase.

- **SC-011**: Child projects can override parent configuration defaults without modifying parent package files.

- **SC-012**: Child projects can define custom agent pipelines that replace the default 5-agent architecture.

- **SC-013**: Child projects can define custom output types with domain-specific structured schemas.

- **SC-014**: Child projects can implement custom conversation handlers for follow-up message processing.

- **SC-015**: sapresalesbot can be implemented as a child project using this plugin architecture.

## Scope & Boundaries

### In Scope (v1)

- Tool protocol and registry infrastructure
- Vector Search tool implementation
- Knowledge Assistant retrieval tool implementation
- Plugin discovery and lifecycle management
- Package restructure for pip installation
- **Declarative pipeline configuration (PipelineConfig, PipelineExecutor)**
- **Pipeline customization protocols (PipelineCustomizer, PhaseProvider)**
- **Custom output type support (OutputTypeProvider)**
- **Conversation handling extensibility (ConversationProvider, IntentClassifier)**
- **Deployment utilities as Python modules (deep_research.deployment.*)**
- **CLI commands (deep-research-run, deep-research-migrate)**
- **App factory function (create_app())**
- Multi-source citation attribution
- Plugin configuration via `app.yaml`
- Child project backend extensibility (import parent, add plugins, deploy independently)
- Child project UI extensibility (import parent components, add custom panels/renderers)
- Configuration override mechanism for child projects
- **Support for building sapresalesbot as a child project**

### Out of Scope (Deferred to v2)

- Plugin dependency resolution and version management
- Plugin marketplace or registry
- Hot-reloading of plugins without application restart
- Child project scaffolding CLI (e.g., `deep-research init my-child-project`)
- Automatic parent version compatibility checking
- Visual pipeline designer/editor
- Plugin sandboxing/security isolation

## Assumptions

- The Databricks SDK (`databricks-sdk>=0.30.0`) provides stable APIs for Vector Search and Serving endpoints.
- Plugin developers have Python packaging knowledge to create entry points.
- Existing WorkspaceClient authentication is sufficient for all new Databricks API integrations.
- The current evidence selector can be extended to handle additional source types without major refactoring.
- Plugin-specific configuration schemas will use Pydantic for validation.
- Child project developers have experience with Python packaging and Databricks Apps deployment.
- Child projects will use the same frontend framework (React) as the parent for UI extensibility.
- Child projects will manage their own Databricks Apps bundle configuration and deployment.
- Pipeline customizations are defined at startup time (not dynamically during execution).
- Custom output types are Pydantic models for validation and serialization.

## Dependencies

- **Databricks SDK** (>=0.30.0): Required for Vector Search and Knowledge Assistant API access.
- **Existing WorkspaceClient**: Reused for OAuth authentication to Databricks services.
- **Existing citation pipeline**: Extended for multi-source attribution.
- **Alembic**: Used for multi-path migration support.

## Open Questions (Resolved)

| Question                  | Decision                                                    |
|---------------------------|-------------------------------------------------------------|
| Package naming            | `databricks-deep-research`                                  |
| Tool defaults             | Disabled by default, require explicit config                |
| Plugin API scope          | Full scope: Tools, Prompts, Pipelines, Outputs, Conversation|
| Package restructure       | Yes - full restructure in this initiative                   |
| Vector Search auth        | Reuse existing WorkspaceClient                              |
| Plugin config location    | In `app.yaml` under `plugins.<plugin_name>` section         |
| Pipeline customization    | In scope for v1 (required for sapresalesbot)                |
| Output types              | In scope for v1 (required for sapresalesbot)                |
| Deployment utilities      | Python modules, not just copyable scripts                   |
| Conversation handling     | In scope for v1 (required for sapresalesbot)                |
