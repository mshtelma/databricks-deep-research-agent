# Feature Specification: Custom Agent Configuration & Selection

**Feature Branch**: `009-custom-agent-config`
**Created**: 2026-02-09
**Status**: Draft
**Input**: User description: "Feature extension for custom agents feature: Allow redefining models per agent for each category present in yaml. Endpoints should be populated live. Allow defining enterprise and web sources for the whole agent or step. Web source definition means white/black list of web sites as it's done in yaml. We just need a UI for almost existing functionality. The main screen must have a dropdown to pick a custom agent. Simple prompt templates should be easily selectable from some drop down. Each user should be able to create simple prompt templates. No need to support variables or things like that just now. Think ultra hard"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select a Custom Agent from the Main Chat Screen (Priority: P1)

A user opens the Deep Research chat interface and sees a dropdown selector near the message input area. The dropdown lists all available custom agents: the user's own private agents and any workspace-shared agents. By default, no agent is selected (standard behavior). When the user picks an agent, the system applies that agent's configuration — model overrides, source restrictions, prompt templates, preset workflow — to all subsequent research queries in the current chat session. The user can switch agents or return to "Default" at any time.

**Why this priority**: This is the entry point for all other sub-features. Without the ability to select an agent, none of the per-agent configurations (models, sources, templates) can be applied. It delivers immediate value by letting users switch between purpose-built research profiles.

**Independent Test**: Can be fully tested by creating a custom agent (via existing UI), selecting it from the main screen dropdown, submitting a query, and verifying the agent name appears in the chat context. Delivers the value of switching between different research configurations without manual re-configuration each time.

**Acceptance Scenarios**:

1. **Given** the user has created two custom agents ("Legal Research" and "Market Analysis"), **When** they open the chat and click the agent selector, **Then** they see both agents listed along with a "Default" option.
2. **Given** a workspace agent "Company Docs" is shared by another user, **When** the current user opens the agent selector, **Then** "Company Docs" appears under a "Workspace" group.
3. **Given** the user has selected "Legal Research" agent, **When** they submit a query, **Then** the backend receives the agent's ID, resolves the full configuration from the database, and applies model overrides, source scope, and prompt template to the research pipeline.
4. **Given** the user has selected an agent, **When** they switch to "Default", **Then** the system reverts to standard behavior with no agent-specific overrides.
5. **Given** the user selects an agent, **When** they navigate to a different chat or reload the page, **Then** the agent selection persists (remembered for new chats).

---

### User Story 2 - Configure Per-Agent Model Overrides (Priority: P2)

A user opens the custom agent editor (create or edit flow) and sees a "Model Configuration" section. This section shows each model category currently defined in the system (e.g., "simple", "analytical", "complex"). For each category, the user can optionally select an endpoint from a live-populated dropdown. The available endpoints are fetched dynamically from the system's active endpoint configuration rather than being hardcoded. If the user leaves a category blank, the system defaults are used. These overrides only apply when this agent is selected for a chat session.

**Why this priority**: Model selection is the most impactful configuration — it determines cost, speed, and quality of research output. Users running cost-sensitive workloads want to force cheaper models; users needing high-quality synthesis want premium models. This directly affects every query.

**Independent Test**: Can be tested by creating an agent, overriding the "complex" model category to use a different endpoint, selecting that agent, submitting a research query, and verifying in the activity log or backend traces that the overridden endpoint was used for synthesis.

**Acceptance Scenarios**:

1. **Given** the system has endpoints "haiku", "sonnet", "opus" configured in app.yaml, **When** the user opens the agent editor's model section, **Then** each model category ("simple", "analytical", "complex") shows a dropdown populated with these three endpoints plus an "Use Default" option.
2. **Given** the user sets "analytical" to "opus" for their agent, **When** they save and select this agent for a chat, **Then** all analytical-tier LLM calls use the "opus" endpoint.
3. **Given** the user leaves "simple" as "Use Default", **When** the agent is used, **Then** the simple tier uses whatever is configured in the system-wide YAML.
4. **Given** an endpoint is removed from the YAML configuration, **When** the user opens the agent editor, **Then** the removed endpoint no longer appears in the dropdowns and any agents that referenced it show a warning indicator.
5. **Given** the user saves model overrides, **When** they re-open the agent editor, **Then** their saved selections are correctly restored.

---

### User Story 3 - Define Enterprise and Web Source Scope per Agent (Priority: P3)

A user edits a custom agent and configures which data sources the agent can use. This includes selecting an overall source scope (enterprise only, web only, all) and toggling individual enterprise sources (Vector Search indexes, Genie spaces, Knowledge Assistants). This is the same functionality as the existing per-query source scope selector, but saved as a persistent agent configuration so the user doesn't need to re-select sources every time.

**Why this priority**: Source scoping is the second most impactful configuration. An agent for "Internal Compliance" should never query the web, while a "Competitive Analysis" agent should only use web sources. Persisting this per-agent avoids repetitive manual toggling.

**Independent Test**: Can be tested by creating an agent with "enterprise_only" scope and only "Compliance Docs" VS index enabled, selecting the agent, running a query, and verifying that no web searches occur and only the enabled enterprise source is queried.

**Acceptance Scenarios**:

1. **Given** the user is editing an agent, **When** they open the "Source Configuration" section, **Then** they see scope options (Enterprise Only, Web Only, All) and a list of available enterprise sources with toggles.
2. **Given** the user sets scope to "enterprise_only" and enables only "Product Knowledge" KA, **When** research runs under this agent, **Then** the researcher only queries the "Product Knowledge" endpoint and skips web search.
3. **Given** the user sets scope to "all" with no source restrictions, **When** research runs, **Then** both web and enterprise sources are available as normal.
4. **Given** a new enterprise source is added to the system after the agent was created, **When** the user opens the agent editor, **Then** the new source appears in the available list (with an "off" default for agents with explicit source lists).

---

### User Story 4 - Define Web Domain Whitelist/Blacklist per Agent (Priority: P4)

A user edits a custom agent and configures web domain filtering. They can choose a filtering mode (whitelist only, blacklist only, or both) and add domain patterns to include or exclude. This mirrors the existing `search.domain_filter` YAML configuration but provides a per-agent UI. Domain patterns support wildcards (e.g., `*.gov`, `*.edu`, `news.*`). When the agent is selected, these domain filters are applied to all web searches during research.

**Why this priority**: Domain filtering is a natural extension of source scoping. While source scope controls *whether* web search happens, domain filtering controls *where* on the web the agent searches. This is critical for compliance, quality control, and focused research.

**Independent Test**: Can be tested by creating an agent with a whitelist of `*.gov` and `*.edu`, selecting the agent, running a web search query, and verifying that search results are filtered to only government and educational domains.

**Acceptance Scenarios**:

1. **Given** the user is editing an agent, **When** they open the "Web Domain Filtering" section, **Then** they see a mode selector (Include, Exclude, Both) and text areas for adding domain patterns.
2. **Given** the user adds `*.gov` and `reuters.com` to the include list with mode set to "include", **When** research runs under this agent, **Then** only results from `.gov` domains and `reuters.com` are included.
3. **Given** the user adds `*.ru` to the exclude list in "exclude" mode, **When** research runs, **Then** results from Russian domains are blocked.
4. **Given** the user enters an invalid pattern (e.g., empty string), **When** they try to save, **Then** the system shows a validation error.
5. **Given** no domain filter is configured on the agent, **When** research runs, **Then** the system-wide domain filter from YAML applies as usual.

---

### User Story 5 - Per-Step Source Configuration (Priority: P5)

A user editing a custom agent's preset steps can optionally assign source preferences to individual steps. For example, step 1 "Gather internal data" might be restricted to enterprise sources, while step 2 "Survey external literature" might be web-only. Each step's source configuration overrides the agent-level scope for that step only.

**Why this priority**: This provides fine-grained control over multi-step workflows. While agent-level source scope (US3) covers most use cases, some workflows benefit from mixing source types across steps. This builds on existing preset step infrastructure.

**Independent Test**: Can be tested by creating a 2-step agent where step 1 uses enterprise-only and step 2 uses web-only, running a research query, and verifying that each step uses the correct source type in the activity log.

**Acceptance Scenarios**:

1. **Given** the user is editing a preset step, **When** they open the step's source config, **Then** they see source scope options (Enterprise Only, Web Only, All, "Use Agent Default") and source toggles.
2. **Given** step 1 has "enterprise_only" and step 2 has "web_only", **When** the planner executes step 1, **Then** it only queries enterprise sources; when executing step 2, it only does web search.
3. **Given** a step has "Use Agent Default" selected, **When** research runs, **Then** the step inherits the agent-level source scope.

---

### User Story 6 - Select and Create Simple Prompt Templates (Priority: P6)

A user can create simple prompt templates consisting of a name and plain text content — no variables, no rendering logic. These templates appear in a dropdown in the agent editor where the user can select them as the system prompt or synthesis prompt for their agent. Additionally, a lightweight template creation flow is available directly from the dropdown ("Create New...") so the user doesn't need to navigate away. Each user can create their own private templates and optionally share them with the workspace.

**Why this priority**: Prompt templates complete the agent configuration story. While models control *how well* the LLM performs and sources control *what data* it sees, prompts control *what it does* with that data. Simplified templates (no variables) make this accessible to non-technical users.

**Independent Test**: Can be tested by creating a simple template "Be concise and formal", selecting it as the system prompt in an agent, running a query, and verifying the output style reflects the prompt guidance.

**Acceptance Scenarios**:

1. **Given** the user has created templates "Formal Report" and "Bullet Points", **When** they open the agent editor and click the system prompt dropdown, **Then** both templates appear along with "None" and "Create New..." options.
2. **Given** the user selects "Create New...", **When** they enter a name and plain text content and save, **Then** the new template is created and automatically selected in the dropdown.
3. **Given** the user creates a template with visibility "workspace", **When** another user opens the template dropdown, **Then** they see the shared template.
4. **Given** a template is already selected on an agent, **When** the template is deleted, **Then** the agent's prompt field reverts to empty (no error, graceful handling).
5. **Given** the user types in the plain text area, **When** they include special characters or markdown, **Then** the content is stored and displayed verbatim (no variable parsing, no rendering).

---

### Edge Cases

- What happens when the user selects an agent whose referenced model endpoint no longer exists? The system falls back to system defaults for that model category and shows a warning badge on the agent selector.
- What happens when all enterprise sources referenced by an agent are deleted? The agent's source config effectively becomes unrestricted; the system logs a warning and falls back to system defaults.
- What happens when two users create agents with the same name? Names are scoped per-user, so collisions only occur if the same user tries duplicate names (rejected with validation error).
- What happens if the agent's prompt template contains text that exceeds the LLM's context window? The system enforces a configurable maximum length in the template editor and warns the user on save.
- What happens when the endpoints API is temporarily unavailable? The model category dropdowns show a loading/error state and disable saving until endpoints are loaded.
- What happens if a user switches agents mid-conversation with an active research job? The switch applies only to new queries; in-progress jobs continue with their original agent configuration.
- What happens if an agent with domain filters is used in "simple" mode (no web search)? Domain filters are silently ignored since no web search occurs — no error, no warning.
- What happens when a user selects a workspace agent referencing resources they lack OBO access to? The system returns a clear error identifying the inaccessible resource (e.g., "You do not have access to endpoint 'opus'. Contact the agent owner or your workspace admin.").

## Requirements *(mandatory)*

### Functional Requirements

#### Agent Selection (US1)

- **FR-001**: System MUST display an agent selector control on the main chat screen near the message input area.
- **FR-002**: The agent selector MUST list the current user's private agents and all workspace-visible agents, grouped by ownership.
- **FR-003**: The agent selector MUST include a "Default" option that uses standard system configuration with no overrides.
- **FR-004**: When an agent is selected, the frontend MUST send the agent's ID with the query submission. The backend MUST resolve the full agent configuration (model overrides, source scope, prompt templates, domain filters) from the database and validate it against the current system configuration at query time.
- **FR-005**: The selected agent MUST persist across new chat sessions and page reloads (stored in user preferences).
- **FR-006**: Switching agents MUST only affect new queries; in-progress research jobs are not interrupted.

#### Model Overrides (US2)

- **FR-007**: The agent editor MUST display a "Model Configuration" section listing all model categories defined in the system configuration.
- **FR-008**: Each model category MUST show a dropdown populated with currently active endpoints, fetched live from a backend API.
- **FR-009**: Each model category dropdown MUST include an "Use Default" option, which is the pre-selected state for new agents.
- **FR-010**: The backend MUST expose an API endpoint that returns the list of currently configured model categories and their available endpoints.
- **FR-011**: When a selected endpoint is no longer available (removed from config), the system MUST show a warning and fall back to the system default for that category.
- **FR-012**: Model overrides MUST be stored as part of the custom agent configuration (persisted to database).
- **FR-012a**: When a user selects a workspace agent that references an endpoint or enterprise source they lack OBO access to, the system MUST return a clear error message identifying the inaccessible resource, rather than silently falling back to defaults.

#### Source Configuration (US3, US5)

- **FR-013**: The agent editor MUST include a "Source Configuration" section with scope selection (Enterprise Only, Web Only, All) and individual source toggles.
- **FR-014**: The source toggle list MUST be populated dynamically from the user's available data sources (same data as the existing per-query source scope selector).
- **FR-015**: The agent-level source scope MUST be applied as the authoritative source configuration for all research queries when the agent is selected and defines sources.
- **FR-015a**: When the selected agent defines a source configuration, the per-query source scope selector MUST be hidden. When no agent is selected or the agent has no source config, the per-query selector MUST remain visible.
- **FR-016**: Preset steps MUST support an optional source scope override that takes precedence over the agent-level scope during that step's execution.
- **FR-017**: Steps with no source scope override MUST inherit the agent-level configuration.

#### Web Domain Filtering (US4)

- **FR-018**: The agent editor MUST include a "Web Domain Filtering" section with mode selection (Include, Exclude, Both) and pattern input areas.
- **FR-019**: Domain patterns MUST support wildcard syntax consistent with the existing YAML configuration (e.g., `*.gov`, `*.edu`, `news.*`).
- **FR-020**: Domain filter configuration MUST be validated on save: patterns must be non-empty and contain valid domain characters.
- **FR-021**: When an agent with domain filters is selected, the filters MUST override the system-wide domain filter during web search execution.
- **FR-022**: When no domain filter is configured on the agent, the system-wide YAML domain filter MUST apply unchanged.

#### Prompt Templates (US6)

- **FR-023**: The agent editor MUST provide dropdown selectors for system prompt template and synthesis prompt template.
- **FR-024**: Template dropdowns MUST list the user's private templates and workspace-visible templates, plus "None" and "Create New..." options.
- **FR-025**: Selecting "Create New..." MUST open an inline creation form requiring only name and plain text content (no variables, no type selection beyond the implied context).
- **FR-026**: New templates created inline MUST be persisted and immediately available for selection.
- **FR-027**: Templates MUST support plain text content only — no variable substitution, no rendering logic.
- **FR-028**: If a selected template is deleted, the agent MUST gracefully clear its reference (no errors, the field reverts to "None").

### Key Entities

- **CustomAgent** (extended): Existing entity, extended with model overrides (mapping of category name to endpoint identifier), domain filter mode, include domain patterns list, and exclude domain patterns list. Existing fields for source scope, enabled/disabled sources, and prompt template references already cover source config and templates.
- **AgentPresetStep** (extended): Existing entity, extended with an optional per-step source scope override. The existing `source_hints` field partially supports this; formal source scope needs to be added.
- **PromptTemplate** (existing): Already supports name, content, type, visibility. No changes needed to the entity itself — only the creation UI is simplified for this iteration.
- **Endpoint Catalog** (virtual/read-only): Not a persisted entity. Derived live from the YAML configuration at runtime. Exposed via a read-only API for frontend consumption. Contains model categories, their assigned endpoint lists, and individual endpoint metadata (name, context window, capabilities).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can select a custom agent from the main chat screen in under 2 seconds (dropdown opens, agent list loads, selection applies).
- **SC-002**: Model category dropdowns in the agent editor populate with live endpoints within 3 seconds of opening the section.
- **SC-003**: 100% of queries submitted with an agent selected correctly apply that agent's model overrides, source scope, domain filters, and prompt templates (no configuration leakage between agents).
- **SC-004**: Users can create a simple prompt template (name + text) in under 30 seconds without leaving the agent editor.
- **SC-005**: Agent selection persists reliably across browser sessions and new chats — users never need to re-select their preferred agent after a page reload.
- **SC-006**: Domain filter patterns configured on an agent produce the same filtering behavior as equivalent patterns in the YAML configuration.
- **SC-007**: Per-step source overrides correctly scope research to the configured sources, with no source leakage across step boundaries.
- **SC-008**: When a referenced endpoint or template is unavailable, the system degrades gracefully with a visible warning rather than failing silently or erroring.

## Clarifications

### Session 2026-02-09

- Q: When a custom agent with source config is selected, what happens to the per-query source scope selector? → A: The per-query source scope selector is hidden when the selected agent defines sources. If the agent has no source config or "Default" agent is selected, the per-query selector remains visible.
- Q: How does the backend receive agent configuration on query submission? → A: Frontend sends only the agent_id. Backend looks up the full config from DB and validates against current YAML at query time. No config snapshot is sent from the frontend.
- Q: Can a user use a workspace agent that references endpoints or sources they lack access to? → A: No. OBO authentication means queries run with the user's own permissions. If a resource is inaccessible, the system MUST return a clear error explaining which resource the user lacks access to, rather than silently falling back.

## Assumptions

- The existing custom agent CRUD, preset step management, and prompt template CRUD are fully functional and stable — this feature extends them, not replaces them.
- The backend endpoints API (FR-010) can be derived from the existing app config loaded from YAML without additional infrastructure.
- Domain filtering at the agent level follows override semantics: agent-level filters replace (not merge with) the system-level domain filter when present.
- Template simplification (no variables) is intentional for this iteration. Variable support can be added later without breaking changes.
- The agent selector dropdown is visible on all chat screen modes (simple, web_search, deep_research), though some agent features (source scope, domain filters) only take effect in modes that use those capabilities.
- When an agent with source configuration is selected, the per-query source scope selector is hidden entirely — the agent owns source decisions. When no agent is selected or the agent has no source config, the existing per-query selector is shown.
- The "selected agent" preference is per-user, not per-chat. All new chats start with the user's most recently selected agent.
- The application is not yet live. Database schema changes (new columns, JSONB shape changes) can be applied freely via migration recreation — no backward-compatible migration constraints.
