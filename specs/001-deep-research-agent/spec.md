# Feature Specification: Deep Research Agent

**Feature Branch**: `001-deep-research-agent`
**Created**: 2025-12-21
**Status**: Draft
**Input**: User description: "Deep research agent with ReAct/Reflexion, web search via Brave, web page fetching, multi-user chat UI with persistence, robust model routing with rate limiting and failover"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Conduct Deep Research via Chat (Priority: P1)

A user opens the chat interface, starts a new conversation, and asks a complex research question. The agent iteratively searches the web using Brave Search, fetches relevant pages, analyzes the content, reflects on gaps in its knowledge, and synthesizes a comprehensive answer with citations.

**Why this priority**: This is the core value proposition—users need the agent to perform deep, iterative research. Without this, the system provides no value.

**Independent Test**: Can be tested by sending a research query and verifying the agent searches the web, fetches pages, and returns a synthesized answer with source citations.

**Acceptance Scenarios**:

1. **Given** a user is logged into the chat interface, **When** they submit a research question like "What are the latest developments in quantum computing error correction?", **Then** the agent performs multiple search iterations, displays its reasoning process, and returns a comprehensive answer with cited sources.

2. **Given** a research query is in progress, **When** the agent identifies knowledge gaps during reflection, **Then** it automatically conducts additional searches to fill those gaps before providing a final answer.

3. **Given** an agent is researching, **When** a search or fetch operation fails, **Then** the agent gracefully handles the error, attempts alternative searches, and communicates any limitations in its response.

---

### User Story 2 - Intelligent Query Understanding & Adaptive Research (Priority: P1.5)

The agent intelligently analyzes each user query to determine the appropriate response strategy. For initial questions, it estimates complexity and automatically selects the appropriate research depth. For follow-up questions, it distinguishes between simple clarifications (answerable from existing context), complex follow-ups (requiring additional research), and new topics (requiring fresh deep research). When queries are ambiguous or underspecified, the agent proactively asks clarifying questions before proceeding.

**Why this priority**: This intelligence is essential to the core research experience. Without it, users either over-research simple questions (wasting time/resources) or under-research complex ones (poor quality). Proactive clarification prevents wasted research on misunderstood queries. This directly enhances User Story 1.

**Independent Test**: Can be tested by submitting queries of varying complexity, follow-up questions of different types, and ambiguous queries to verify the agent correctly classifies and responds appropriately.

**Acceptance Scenarios**:

1. **Given** a user submits an initial question without specifying research depth, **When** the query is simple (e.g., "What is the capital of France?"), **Then** the agent provides a direct answer without triggering deep web research.

2. **Given** a user submits an initial question without specifying research depth, **When** the query is complex (e.g., "Compare the economic policies of the EU and US regarding AI regulation"), **Then** the agent automatically selects extended research depth and performs comprehensive web research.

3. **Given** a user asks a follow-up question after receiving a research answer, **When** the follow-up is a simple clarification (e.g., "Can you explain that third point in more detail?"), **Then** the agent responds using the existing context without new web searches.

4. **Given** a user asks a follow-up question, **When** the follow-up requires new information not in the current context (e.g., "How does this compare to what happened in Japan?"), **Then** the agent conducts targeted additional research to answer the question.

5. **Given** a user submits an ambiguous or underspecified query, **When** the agent cannot determine the user's intent with confidence, **Then** the agent asks 1-3 clarifying questions before proceeding with research.

6. **Given** the agent has asked clarifying questions, **When** the user provides answers, **Then** the agent incorporates those answers to refine its understanding and proceeds with appropriately scoped research.

7. **Given** a user's follow-up question introduces a substantially new topic, **When** the new topic is unrelated to prior research, **Then** the agent treats it as a fresh research query with appropriate depth estimation.

---

### User Story 3 - Multi-User Chat with Persistence (Priority: P2)

Multiple users can access the system simultaneously, each with their own isolated chat history. Users can create new chats, switch between existing conversations, and resume previous research sessions at any time.

**Why this priority**: Persistence enables users to build on previous research and multi-user support is essential for shared workspace environments. However, basic research functionality must work first.

**Independent Test**: Can be tested by creating multiple users, having each create chats, and verifying chat isolation and persistence across sessions.

**Acceptance Scenarios**:

1. **Given** a user has previous chat sessions, **When** they log in to the application, **Then** they see a list of their previous chats sorted by most recent activity.

2. **Given** a user creates a new chat, **When** they navigate away and return later, **Then** the full conversation history is preserved and visible.

3. **Given** two users are logged in simultaneously, **When** each creates and uses chats, **Then** their conversations are completely isolated from each other.

4. **Given** a user has many chats, **When** they view their chat list, **Then** they can identify each chat by its title or first message summary.

---

### User Story 4 - Message Control & Refinement (Priority: P2.5)

Users can control and refine their research interactions by stopping in-progress operations, editing previous messages, and regenerating responses. This gives users full control over the research process and allows iterative refinement of queries.

**Why this priority**: Essential UX for any modern chat interface. Users expect to be able to correct mistakes, try alternative phrasings, and stop runaway operations. Directly impacts user satisfaction and productivity.

**Independent Test**: Can be tested by initiating a research query, stopping it mid-execution, editing a previous message, and regenerating a response.

**Acceptance Scenarios**:

1. **Given** a research operation is in progress, **When** the user clicks the stop/cancel button, **Then** the operation stops within 2 seconds and any partial results are preserved.

2. **Given** a user has sent a message, **When** they click edit on that message, **Then** they can modify the text and resubmit, which invalidates all subsequent messages in the thread.

3. **Given** an agent response is displayed, **When** the user clicks regenerate, **Then** the system generates a new response to the same query using fresh search results.

4. **Given** a user edits a message in the middle of a conversation, **When** the edit is submitted, **Then** all messages after the edited message are removed and the agent processes the new query.

---

### User Story 5 - Resilient Model Access with Automatic Failover (Priority: P3)

The system intelligently routes requests to available model endpoints based on configured roles (e.g., "reasoning", "fast", "structured"). When the primary endpoint is overloaded or unavailable, the system automatically fails over to alternative endpoints, adjusting the request as needed (e.g., truncating context for smaller context windows). Users experience uninterrupted research even during high load or partial outages.

**Why this priority**: Model availability directly impacts research quality. Without resilient routing, a single endpoint failure would block all users. However, basic research and persistence must work first with at least one model.

**Independent Test**: Can be tested by simulating endpoint failures or rate limits and verifying the system automatically routes to alternative endpoints without user intervention.

**Acceptance Scenarios**:

1. **Given** the primary endpoint for a model role is rate-limited, **When** the agent sends a request, **Then** the system automatically routes to an alternative endpoint configured for that role.

2. **Given** an endpoint has a smaller context window than the request requires, **When** the system selects that endpoint as fallback, **Then** it automatically adjusts the request (truncates context) to fit within the endpoint's limits.

3. **Given** all endpoints for a model role are unavailable, **When** a request is made, **Then** the system provides a clear error message and queues the request for retry with exponential backoff.

4. **Given** multiple endpoint selection strategies are configured (round-robin, priority-based), **When** endpoints are available, **Then** the system follows the configured selection algorithm.

---

### User Story 6 - Chat Organization & Discovery (Priority: P3.5)

Users can effectively organize and find their research across many conversations. They can rename chats for clarity, search across all chat history, archive old conversations, and export research for sharing or documentation.

**Why this priority**: Enterprise users accumulate many conversations over time. Without organization and search, valuable research becomes lost. Export capability is essential for sharing findings and compliance.

**Independent Test**: Can be tested by creating multiple chats, renaming them, searching for specific content, archiving old chats, and exporting a conversation.

**Acceptance Scenarios**:

1. **Given** a user has a chat, **When** they click rename, **Then** they can provide a custom title that replaces the auto-generated one.

2. **Given** a user has multiple chats, **When** they enter a search query in the chat list, **Then** the system returns chats containing matching text within 1 second.

3. **Given** a user wants to hide but not delete a chat, **When** they archive it, **Then** the chat is hidden from the default list but remains searchable and recoverable.

4. **Given** a user wants to share research findings, **When** they click export, **Then** the system generates a Markdown or PDF file containing the full conversation with sources and citations.

5. **Given** search returns many results, **When** the user views the results, **Then** they can see which messages matched and navigate directly to them.

---

### User Story 7 - View Agent Reasoning Process (Priority: P4)

Users can observe the agent's step-by-step reasoning, including which searches it performed, what pages it fetched, and how it reflected on the information gathered. This transparency builds trust and helps users understand the research process.

**Why this priority**: Transparency is important for trust but the core research, persistence, and model resilience features must work first. Users can still get value without seeing internal reasoning.

**Independent Test**: Can be tested by submitting a research query and verifying that reasoning steps, search queries, and fetched URLs are displayed during or after the research process.

**Acceptance Scenarios**:

1. **Given** the agent is processing a research query, **When** it performs a search or fetch operation, **Then** the user can see which action was taken and why.

2. **Given** a research query has completed, **When** the user views the response, **Then** they can expand or access a log of all reasoning steps that led to the answer.

---

### User Story 8 - REST API Access for External Integration (Priority: P5)

External applications can interact with the research agent via REST API, enabling integration with custom UIs, automation workflows, or other Databricks applications.

**Why this priority**: API access enables the agent to be used programmatically and is required for the Databricks chat agent pattern. However, the core UI and agent functionality take precedence.

**Independent Test**: Can be tested by making API calls to submit research queries and receive responses without using the UI.

**Acceptance Scenarios**:

1. **Given** a valid API request with a research query, **When** the request is submitted to the agent endpoint, **Then** the agent processes the query and returns a structured response with the research findings.

2. **Given** an API client provides conversation context (previous messages), **When** a follow-up query is submitted, **Then** the agent uses the context to provide a relevant continuation of the research.

---

### User Story 9 - Automated End-to-End Testing (Priority: P2)

The development team has automated browser-based tests that simulate real user interactions with the chat interface. Tests cover complete user journeys including starting research conversations, asking follow-up questions, using message controls (stop, edit, regenerate), and verifying response quality.

**Why this priority**: Automated e2e tests are essential for maintaining quality and preventing regressions during development. They enable confident refactoring and feature additions while ensuring core user journeys remain functional.

**Independent Test**: Can be tested by running the e2e test suite against a running instance and verifying all tests pass.

**Acceptance Scenarios**:

1. **Given** the application is running, **When** the e2e test suite is executed, **Then** tests simulate a user opening the chat interface, submitting a research query, and receiving a response with citations.

2. **Given** a test conversation exists, **When** the e2e test submits a follow-up question, **Then** the test verifies the agent responds appropriately using context from the previous exchange.

3. **Given** a research operation is in progress during a test, **When** the test triggers a stop/cancel action, **Then** the test verifies the operation stops within the expected timeframe.

4. **Given** a test has received an agent response, **When** the test edits a previous message and resubmits, **Then** the test verifies subsequent messages are invalidated and a new response is generated.

5. **Given** a test has received an agent response, **When** the test triggers regenerate, **Then** the test verifies a new response is generated for the same query.

6. **Given** the e2e test suite completes, **When** test results are reviewed, **Then** detailed logs and screenshots are available for any failures to aid debugging.

7. **Given** a research query with multiple agent steps, **When** the e2e test observes the research progress panel, **Then** the test verifies plan steps transition through pending → in_progress → completed states.

8. **Given** a test needs to verify streaming behavior, **When** the agent begins synthesizing, **Then** the test observes incremental content appearing in the message area before the final response completes.

---

### User Story 10 - Comprehensive Python Unit Testing (Priority: P2)

The development team has a comprehensive suite of Python unit tests covering all backend components: services, API endpoints, agent nodes, tools, and utilities. Tests use mocking to isolate components and run quickly without external dependencies. Test coverage metrics are tracked and maintained above established thresholds.

**Why this priority**: Unit tests are essential for maintaining code quality, enabling confident refactoring, and catching regressions early. They complement E2E tests by providing fast feedback on individual component behavior and edge cases that are difficult to test at the integration level.

**Independent Test**: Can be tested by running the unit test suite and verifying tests pass with coverage above thresholds.

**Acceptance Scenarios**:

1. **Given** the unit test suite is executed, **When** tests run against service layer components (ChatService, MessageService), **Then** tests verify CRUD operations, error handling, and business logic using mocked database sessions.

2. **Given** the unit test suite is executed, **When** tests run against API endpoints, **Then** tests verify request validation, response schemas, authentication handling, and error responses using FastAPI TestClient with mocked services.

3. **Given** the unit test suite is executed, **When** tests run against agent nodes (Coordinator, Planner, Researcher, Reflector, Synthesizer), **Then** tests verify prompt construction, LLM response parsing, state transitions, and error handling using mocked LLM clients.

4. **Given** the unit test suite is executed, **When** tests run against tools (web_search, web_crawler), **Then** tests verify input validation, response parsing, error handling, and rate limiting using mocked HTTP clients.

5. **Given** the unit test suite is executed, **When** tests run against utility modules (JSON repair, context truncation, model routing), **Then** tests verify edge cases, malformed input handling, and configuration parsing.

6. **Given** the unit test suite completes, **When** coverage reports are generated, **Then** line coverage meets or exceeds the established threshold (target: 80% for services, 70% for API, 60% for agents).

7. **Given** a developer adds new functionality, **When** they submit changes for review, **Then** corresponding unit tests are included and overall coverage does not decrease.

8. **Given** tests need to verify async behavior, **When** tests run against async functions, **Then** tests properly await coroutines and verify async context managers using pytest-asyncio.

---

### User Story 11 - Frontend Unit Testing (Priority: P3)

The development team has unit tests for React components and hooks using Vitest and React Testing Library. Tests verify component rendering, user interactions, state management, and hook behavior with mocked API responses.

**Why this priority**: Frontend unit tests ensure UI components behave correctly in isolation and catch JavaScript/TypeScript errors before they reach production. Lower priority than Python tests because E2E tests already cover critical user journeys.

**Independent Test**: Can be tested by running the frontend test suite and verifying tests pass.

**Acceptance Scenarios**:

1. **Given** the frontend test suite is executed, **When** tests run against custom hooks (useStreamingQuery, useChats, useMessages), **Then** tests verify state transitions, error handling, and cleanup using mocked fetch/SSE.

2. **Given** the frontend test suite is executed, **When** tests run against chat components (MessageList, MessageInput, ChatSidebar), **Then** tests verify rendering, user interactions, and accessibility.

3. **Given** the frontend test suite is executed, **When** tests run against research components (PlanProgress, AgentStatusIndicator), **Then** tests verify state display and animations.

4. **Given** a component receives invalid props or error states, **When** the test renders the component, **Then** the component handles errors gracefully without crashing.

---

### Edge Cases

- What happens when a user submits an empty or whitespace-only query?
  - System returns HTTP 400 with error message: "Query cannot be empty. Please enter a research question."

- What happens when Brave Search API rate limits are exceeded?
  - System should queue requests and inform user of delays, or gracefully degrade with a clear message.

- What happens when a fetched web page is inaccessible (403, 404, timeout)?
  - Agent should skip the page, note it in reasoning, and continue with other sources.

- What happens when a user's session expires during a long research operation?
  - The research should complete, and results should be available when the user re-authenticates.

- What happens when the LLM returns an unexpectedly long response?
  - Response should be handled gracefully with potential truncation or pagination in the UI.

- What happens when a chat has no messages?
  - Empty chats should be handled gracefully in the UI (placeholder message or auto-cleanup).

- What happens when all model endpoints for a role are rate-limited simultaneously?
  - System should implement exponential backoff with jitter, queue the request, and inform the user of the delay.

- What happens when a model endpoint returns malformed structured output?
  - System should retry with the same or alternative endpoint, and fall back to unstructured output if retries fail.

- What happens when context must be truncated for a fallback endpoint?
  - System should preserve the most recent messages and system instructions, truncating older context first while maintaining conversation coherence.

- What happens when a user stops research mid-iteration?
  - System should halt all pending operations within 2 seconds, preserve any partial results gathered so far, and display them to the user with a clear indication that research was stopped early.

- What happens when a user edits a message in the middle of a conversation?
  - System should remove all messages after the edited message, then process the new query as if it were a fresh continuation from that point.

- What happens when search returns too many results?
  - System should paginate results with reasonable defaults (e.g., 20 per page), allow filtering/sorting, and show total count.

- What happens when export fails for large conversations?
  - System should attempt chunked generation, provide clear error messages if the conversation is too large, and suggest alternatives (e.g., export as Markdown instead of PDF, or export a date range).

- What happens when the system misclassifies query complexity?
  - System should allow users to override automatic depth selection at any time. If user manually selects a different depth after seeing classification reasoning, system should immediately adapt and restart research with the new depth if research hasn't completed.

- What happens when a user ignores or dismisses clarifying questions?
  - System should proceed with best-effort research based on its interpretation of the query, but note in its response that results may not fully address user intent due to ambiguity. User can always ask follow-ups.

- What happens when follow-up detection incorrectly treats a new topic as a clarification?
  - System displays its classification reasoning, allowing users to rephrase or explicitly request fresh research. User can also manually select research depth to override automatic behavior.

- What happens when a simple query is actually nuanced but appears straightforward?
  - System should err on the side of providing substantive responses. If "simple" classification produces a too-brief answer, user can regenerate with explicit depth selection or ask a follow-up prompting deeper research.

- What happens when the same user edits from multiple devices/tabs simultaneously?
  - System uses last-write-wins semantics. When a conflict is detected, the UI automatically refreshes to show the latest state. Users can recover by re-applying their intended changes.

- What happens when an endpoint's tokens_per_minute limit is reached mid-request?
  - System should wait until the next minute window resets OR failover to the next available endpoint if fallback_on_429 is enabled for that role. User is informed if significant delay is expected.

- What happens when endpoint configuration overrides conflict with role defaults?
  - Endpoint-specific values always take precedence over role-level defaults. The effective configuration is computed by merging role defaults with endpoint overrides, with endpoint values winning.

- What happens when the Reflector continuously decides ADJUST, creating a replan loop?
  - System enforces max_plan_iterations (default: 3). After reaching the limit, the system proceeds to Synthesizer with whatever observations have been collected, noting in the response that research may be incomplete.

- What happens when unit tests require database access?
  - Tests use mocked AsyncSession objects; no real database connection is established. Test fixtures provide pre-configured mock responses.

- What happens when E2E tests fail due to timing issues?
  - Tests use explicit waits with reasonable timeouts instead of arbitrary sleeps. Flaky tests are identified and fixed or quarantined.

- What happens when external services (Brave, LLM) are unavailable during testing?
  - Unit tests use mocks and are unaffected. E2E tests use recorded responses (fixtures) or skip tests that require live services.

- What happens when Planner creates an empty plan (no steps)?
  - Coordinator should have already handled simple queries. If Planner produces an empty plan, system treats it as "has_enough_context" and proceeds directly to Synthesizer with background investigation results.

- What happens when YAML configuration contains invalid values?
  - System validates configuration at startup and fails fast with clear error messages indicating which fields are invalid. Application does not start until configuration is fixed.

- What happens when YAML configuration references undefined endpoints in a role?
  - System validates endpoint references and fails fast with an error message listing the undefined endpoints and which roles reference them.

- What happens when environment variables referenced in YAML are not set?
  - If variable uses default syntax (`${ENV_VAR:-default}`), the default value is used. If no default is provided (`${ENV_VAR}`), system fails fast with an error message listing missing variables.

- What happens when Researcher fails to gather any information for a step?
  - Researcher stores an empty or failure observation. Reflector evaluates whether to CONTINUE (try next step), ADJUST (replan with different approach), or COMPLETE (enough info from other steps).

- What happens when background investigation times out or fails?
  - Planner proceeds without background context. The plan may be less informed, but research can still be conducted. A warning is logged for observability.

## Requirements *(mandatory)*

### Functional Requirements

**Agent Core:**
- **FR-001**: System MUST implement a ReAct/Reflexion reasoning loop that iteratively searches, fetches, reflects, and synthesizes information.
- **FR-002**: System MUST integrate with Brave Search API to perform web searches based on agent-generated queries.
- **FR-003**: System MUST fetch and parse web page content when the agent decides additional context is needed.
- **FR-004**: System MUST provide synthesized research answers with citations linking to source URLs.
- **FR-005**: System MUST expose the agent via a REST API endpoint compatible with Databricks chat agent patterns. API supports query submission with conversation context; message control features (stop, edit, regenerate) are UI-only for V1.
- **FR-024**: System MUST support configurable research depth levels (auto, light, medium, extended) that control the maximum number of search/reflect iterations: light (max 2 iterations), medium (max 5 iterations), extended (max 10 iterations). When set to "auto" (default), the system determines depth based on query complexity.

**Query Intelligence:**
- **FR-045**: System MUST analyze each query to estimate complexity and classify it as: simple (answerable without research), moderate (light research needed), or complex (extended research needed).
- **FR-046**: When research depth is set to "auto" (default), system MUST automatically select the appropriate depth level based on query complexity estimation.
- **FR-047**: System MUST distinguish between follow-up question types: simple clarification (uses existing context), complex follow-up (requires additional targeted research), or new topic (requires fresh research).
- **FR-048**: For simple clarifications, system MUST respond using existing conversation context and prior research results without initiating new web searches.
- **FR-049**: For complex follow-ups, system MUST conduct targeted additional research while leveraging existing context to avoid redundant searches.
- **FR-050**: System MUST detect ambiguous or underspecified queries where user intent cannot be determined with confidence.
- **FR-051**: When a query is ambiguous, system MUST ask 1-3 clarifying questions before proceeding with research, explaining why clarification is needed.
- **FR-052**: System MUST incorporate user responses to clarifying questions to refine query understanding and scope the research appropriately.
- **FR-053**: System MUST display its query classification reasoning to the user (e.g., "This appears to be a complex topic requiring extended research" or "I'll answer from our previous discussion").

**Multi-Agent Architecture:**
- **FR-055**: System MUST implement a 5-agent architecture: Coordinator (query classification, clarification), Planner (research plan creation), Researcher (step execution), Reflector (progress evaluation), and Synthesizer (final report generation).
- **FR-056**: Coordinator agent MUST classify incoming queries and run up to 3 clarification rounds for vague or underspecified queries before handoff to Planner.
- **FR-057**: System MUST perform background investigation (quick web search, ~5 seconds) before planning phase to provide context for research plan creation.
- **FR-092**: Background Investigator MUST generate focused search queries using LLM instead of passing raw user queries directly to search API, preventing HTTP 422 errors from overly long queries.
- **FR-058**: Planner agent MUST create structured research plans with typed steps (RESEARCH for web search/crawl, ANALYSIS for reasoning without tools). Each step includes title, description, step_type, and needs_search flag.
- **FR-059**: Researcher agent MUST execute ONE plan step at a time, storing the observation before reflection. Steps are executed sequentially, not in parallel.
- **FR-060**: Reflector agent MUST evaluate after EACH step and decide: CONTINUE (proceed to next step), ADJUST (return to Planner for replanning), or COMPLETE (skip remaining steps, proceed to Synthesizer).
- **FR-061**: When Reflector decides ADJUST, Planner MUST receive all observations collected so far and adjust remaining steps accordingly. Suggested changes from Reflector should inform the replanning.
- **FR-062**: Synthesizer agent MUST analyze all collected observations and generate a final report with proper citations and source attribution.
- **FR-063**: System MUST repair malformed JSON outputs from LLMs using balanced brace extraction and the json_repair library as fallback.
- **FR-064**: System MUST validate research plans and auto-repair plans with missing or invalid step_type fields by inferring from needs_search value.
- **FR-065**: System MUST track plan_iterations and enforce max_plan_iterations (default: 3) to prevent infinite replan loops.

**Model Routing & Resilience:**
- **FR-016**: System MUST support named model roles (e.g., "micro", "simple", "analytical", "complex") that define default parameters (temperature, max_tokens, reasoning_effort, rotation_strategy, fallback_on_429, tokens_per_minute) applying to all endpoints in that role.
- **FR-017**: Each model endpoint configuration MUST include endpoint identifier (Databricks serving endpoint name), max_context_window (input limit), and tokens_per_minute (rate limit). Endpoints MAY override role-level defaults for temperature, max_tokens, and reasoning parameters.
- **FR-018**: System MUST support configurable endpoint selection strategies per model role: round-robin (distribute load evenly) or priority-based (try endpoints in configured order until success). Default strategy is priority-based.
- **FR-019**: When fallback_on_429 is enabled for a role, system MUST automatically failover to alternative endpoints when the current endpoint returns rate limit errors (429) or is unavailable (5xx).
- **FR-020**: System MUST implement exponential backoff with jitter for retry attempts on failed requests, and track per-endpoint tokens_per_minute usage to avoid exceeding rate limits.
- **FR-021**: When a fallback endpoint has a smaller context window (max_context_window), system MUST automatically truncate the request to fit while preserving conversation coherence (prioritize system prompt, then recent messages).
- **FR-022**: System MUST support structured output generation (JSON mode or function calling). For reasoning parameters (reasoning_effort, reasoning_budget), see FR-054.
- **FR-023**: System MUST track endpoint health and availability in-memory (not persisted to database) to inform routing decisions. Health state includes consecutive error count, rate limit status, and last success timestamp.
- **FR-054**: System MUST support reasoning_effort (low/medium/high) and reasoning_budget parameters at the role level, with optional endpoint-level overrides, for models that support extended thinking (e.g., o-series models).

**Central Configuration (YAML):**
- **FR-081**: System MUST load all model and agent configuration from a central YAML configuration file (`config/app.yaml`).
- **FR-082**: The YAML configuration MUST define model endpoints with: endpoint identifier (Databricks serving endpoint name), max_context_window, tokens_per_minute, and optional overrides (temperature, max_tokens, reasoning_effort, reasoning_budget, supports_structured_output).
- **FR-083**: The YAML configuration MUST define model roles (micro, simple, analytical, complex) with: ordered list of endpoint references, default temperature, max_tokens, reasoning_effort, tokens_per_minute limit, rotation_strategy, and fallback_on_429 flag.
- **FR-084**: The YAML configuration MUST define a default role that is used when no specific role is requested.
- **FR-085**: The YAML configuration MUST define agent configuration including: max_search_queries, max_search_results, max_urls_to_crawl, content_preview_length, content_storage_length, max_previous_observations, page_contents_limit, max_plan_iterations, and max_clarification_rounds.
- **FR-086**: The YAML configuration MUST define search configuration including: brave_requests_per_second, default_result_count, and freshness settings.
- **FR-087**: System MUST validate the YAML configuration at startup and fail fast with clear error messages if configuration is invalid (missing required fields, invalid values, undefined endpoint references).
- **FR-088**: System MUST support environment variable interpolation in YAML configuration using `${ENV_VAR}` syntax with optional default values `${ENV_VAR:-default}`.
- **FR-089**: System MUST fall back to sensible defaults when the YAML configuration file is not present, allowing development without explicit configuration.
- **FR-090**: System MAY support hot-reload of configuration changes without application restart (file watch mechanism).

**Message Control:**
- **FR-029**: Users MUST be able to stop/cancel in-progress research operations.
- **FR-030**: Users MUST be able to edit their previous messages.
- **FR-031**: Editing a message MUST invalidate (remove) all subsequent messages in the conversation thread.
- **FR-032**: Users MUST be able to regenerate agent responses to get fresh research results.

**Chat Management:**
- **FR-006**: System MUST support multiple concurrent users with isolated data.
- **FR-007**: System MUST persist chat conversations across user sessions.
- **FR-008**: Users MUST be able to create new chat conversations.
- **FR-009**: Users MUST be able to switch between existing conversations.
- **FR-010**: Users MUST be able to view their conversation history.
- **FR-026**: Users MUST be able to delete their chat conversations (soft delete with 30-day recovery window before permanent purge).
- **FR-036**: Users MUST be able to rename chat conversations with custom titles.
- **FR-037**: Users MUST be able to search across all their chat history using full-text search.
- **FR-038**: Users MUST be able to export chat conversations as Markdown or PDF with sources and citations.
- **FR-039**: Users MUST be able to archive chats (hide from default list while remaining searchable and recoverable).
- **FR-093**: System MUST auto-select the most recent chat or create a new chat when user navigates to the home page without a specific chat selected.
- **FR-094**: System MUST NOT display "Untitled Chat" label; instead show "New chat..." with muted styling for chats without a title.

**Feedback:**
- **FR-033**: Users MUST be able to rate agent responses (thumbs up/down).
- **FR-034**: Users MUST be able to report factual errors or problematic content in responses.
- **FR-035**: User feedback MUST be logged via MLflow traces for quality analysis and model improvement.

**Personalization:**
- **FR-040**: Users MUST be able to set persistent system instructions (custom instructions) that apply to all their conversations.

**User Interface:**
- **FR-011**: System MUST provide a web-based chat interface for interacting with the agent.
- **FR-012**: Chat interface MUST display messages in chronological order with clear user/agent attribution.
- **FR-013**: Chat interface MUST show loading/processing indicators during research operations.
- **FR-014**: Chat interface MUST stream agent reasoning steps in real-time during research and consolidate them into an expandable section after completion.
- **FR-025**: Chat interface MUST provide a research depth selector (auto, light, medium, extended) defaulting to "auto", allowing users to override automatic depth selection when desired.
- **FR-041**: Messages MUST have copy-to-clipboard functionality for easy content extraction.
- **FR-042**: System MUST support keyboard shortcuts for common actions (new chat, send message, stop, navigate).
- **FR-091**: Agent messages MUST render with full markdown support including headings, lists, code blocks with syntax highlighting, tables, blockquotes, links (opening in new tabs), and GFM extensions (strikethrough, task lists).
- **FR-095**: Research plan step status indicators MUST transition smoothly (pending → in_progress → completed/skipped) without visual flickering or reverting states.

**Authentication & Authorization:**
- **FR-015**: System MUST authenticate users via Databricks workspace identity.

**Observability:**
- **FR-027**: System MUST implement comprehensive distributed tracing using MLflow traces for all research operations, model calls, and tool executions.
- **FR-028**: System MUST capture structured logs and metrics for latency, error rates, endpoint usage, and research session statistics.
- **FR-043**: System MUST maintain audit logs of user actions (create, read, delete, export, edit) for security and compliance.
- **FR-096**: System MUST enable async logging for MLflow in FastAPI context to ensure traces are properly flushed.
- **FR-097**: System MUST wrap LLM client calls in MLflow spans with tier, endpoint, token metrics, and latency attributes.
- **FR-098**: System MUST set research context attributes (session_id, query, config) on root MLflow span for trace correlation.
- **FR-099**: System MUST call `mlflow.update_current_trace()` with `mlflow.trace.user` (user_id) and `mlflow.trace.session` (chat_id) metadata to group all traces from the same chat conversation together.
- **FR-066**: System MUST expose GET /health endpoint returning service status (database connectivity, model endpoint availability) for monitoring and load balancer health checks.

**Testing Infrastructure:**
- **FR-067**: System MUST have pytest-based unit test suite with pytest-asyncio for async function testing.
- **FR-068**: Unit tests MUST use dependency injection and mocking to isolate components from external dependencies (database, LLM, HTTP clients).
- **FR-069**: Unit tests MUST be organized in a `tests/unit/` directory with subdirectories mirroring the `src/` structure (e.g., `tests/unit/services/`, `tests/unit/api/`, `tests/unit/agent/`).
- **FR-070**: System MUST have test fixtures providing mock database sessions, mock LLM clients, and mock HTTP responses.
- **FR-071**: Unit tests MUST cover service layer methods including happy path, error conditions, and edge cases.
- **FR-072**: Unit tests MUST cover API endpoints including request validation, response schemas, and authentication handling.
- **FR-073**: Unit tests MUST cover agent nodes including prompt construction, response parsing, and state transitions.
- **FR-074**: System MUST have Playwright-based E2E test suite with Page Object Model architecture.
- **FR-075**: E2E tests MUST be organized in `e2e/tests/` with corresponding page objects in `e2e/pages/`.
- **FR-076**: E2E tests MUST use `data-testid` attributes for element selection instead of CSS classes or text content.
- **FR-077**: System MUST generate test coverage reports in HTML and XML formats for CI integration.
- **FR-078**: Frontend MUST have Vitest-based unit test suite with React Testing Library for component testing.
- **FR-079**: Frontend unit tests MUST cover custom hooks with mocked fetch and SSE responses.
- **FR-080**: Frontend unit tests MUST verify component accessibility using testing-library accessibility queries.

**Accessibility:**
- **FR-044**: UI MUST comply with WCAG 2.1 AA accessibility standards including keyboard navigation, screen reader support, and sufficient color contrast.

### Key Entities

- **User**: A workspace user who can create and own chats. Identified by Databricks workspace identity. Each user has their own isolated set of chats.

- **Chat**: A conversation thread between a user and the agent. Contains a sequence of messages, has a title or summary for identification, and belongs to exactly one user. Supports soft deletion with 30-day recovery window.

- **Message**: A single exchange in a chat. Can be from the user (query) or the agent (response). Agent messages may include structured reasoning traces and source citations.

- **ResearchSession**: The execution context of a single research query. Tracks the search queries performed, pages fetched, reasoning steps taken, and final synthesis. Associated with a single agent message.

- **Source**: A web resource referenced in research. Contains URL, title, snippet, and relevance to the research query. Linked to research sessions and agent responses.

- **ModelRole**: A named abstraction representing a capability tier (e.g., "micro", "simple", "analytical", "complex"). Defines role-level defaults: temperature, max_tokens (output limit), reasoning_effort, reasoning_budget, tokens_per_minute, rotation_strategy, and fallback_on_429. Maps to an ordered list of endpoint IDs. Loaded from YAML configuration.

- **ModelEndpoint**: A configured connection to a specific model. Contains endpoint identifier (Databricks serving endpoint name), max_context_window (input context limit, required), and tokens_per_minute (rate limit per endpoint, required). May optionally override role-level defaults for temperature, max_tokens, reasoning_effort, reasoning_budget, and supports_structured_output. Loaded from YAML configuration.

- **AppConfig**: Central configuration loaded from `config/app.yaml`. Contains models section (endpoints and roles), agents section (research limits, plan iterations), and search section (rate limits, freshness). Supports environment variable interpolation and falls back to sensible defaults when file is absent.

- **EndpointHealth**: Tracks the availability and performance of a model endpoint at runtime. Maintained in-memory only (not persisted to database). Includes is_healthy flag, consecutive error count, rate_limited_until timestamp, tokens_used_this_minute counter, and last successful request time.

- **QueryClassification**: The result of analyzing a user query. Contains complexity level (simple/moderate/complex), follow-up type (new topic/clarification/complex follow-up), ambiguity flags, suggested clarifying questions if ambiguous, and recommended research depth. Used to drive adaptive research behavior.

- **Plan**: A structured research plan created by the Planner agent. Contains id, title, thought (planning rationale), ordered list of PlanStep items, has_enough_context flag, and iteration counter. Plans can be adjusted by the Planner when the Reflector decides ADJUST.

- **PlanStep**: A single step in a research plan. Contains id, title, description, step_type (RESEARCH or ANALYSIS), needs_search flag, status (PENDING, IN_PROGRESS, COMPLETED, SKIPPED), and observation (result from Researcher after execution). Steps are executed sequentially.

- **StepType**: Enumeration of plan step types. RESEARCH steps involve web search and content crawling (executed by Researcher agent). ANALYSIS steps involve pure reasoning without tools (executed by Synthesizer agent).

- **ReflectionResult**: The decision from the Reflector agent after evaluating a completed step. Contains decision (CONTINUE, ADJUST, COMPLETE), reasoning explaining the decision, and optional suggested_changes list (hints for Planner when decision is ADJUST).

- **UserPreferences**: Persistent user settings including system instructions (custom instructions), default research depth, and UI preferences. Applied automatically to all user conversations.

- **MessageFeedback**: User-provided ratings and reports on agent responses. Contains rating (positive/negative), optional error report text, timestamp, and link to the rated message. Logged via MLflow traces for analysis.

- **AuditLog**: Record of user actions for security and compliance. Tracks action type (create, read, delete, export, edit), target entity, user identity, timestamp, and relevant metadata.

## Clarifications

### Session 2025-12-21

- Q: What storage mechanism should be used for chat data persistence? → A: Databricks Lakebase (PostgreSQL within Databricks ecosystem)
- Q: What is the maximum number of research iterations before synthesizing? → A: Configurable via UI with t-shirt sizes (light, medium, extended)
- Q: How should agent reasoning steps be displayed to users? → A: Both streaming (real-time during research) and post-hoc (consolidated expandable section)
- Q: Can users delete their chat conversations? → A: Soft delete (recoverable for 30 days, then permanently purged)
- Q: What level of observability should the system provide? → A: Comprehensive logging/tracing using MLflow traces

### Gap Analysis Session 2025-12-21

Enterprise feature gap analysis identified 11 critical/high-priority capabilities missing from initial spec:

**Added (Critical):**
- Stop/cancel operations (FR-029)
- Message editing & regeneration (FR-030 to FR-032)
- User feedback with MLflow logging (FR-033 to FR-035)
- Chat rename (FR-036)
- Full-text search across chats (FR-037)
- Chat export (FR-038)
- Audit logging (FR-043)

**Added (High Priority):**
- Chat archive (FR-039)
- System instructions/custom preferences (FR-040)
- Copy-to-clipboard (FR-041)
- Keyboard shortcuts (FR-042)
- WCAG 2.1 AA accessibility (FR-044)

**Deferred to V2:**
- File uploads/document analysis
- Sharing/collaboration
- Chat folders
- Usage quotas visibility (removed from scope)
- Admin UI for model endpoint configuration
- REST API message control (stop, edit, regenerate via API)
- Customer-managed encryption keys (CMK)
- Real-time cross-device sync

### Contradiction & Gap Analysis Session 2025-12-21

- Q: How should model endpoint configuration be handled (admin role)? → A: External configuration only (config files, env vars, Databricks secrets); no built-in admin UI for V1
- Q: Should the REST API support message control features (stop, edit, regenerate)? → A: No, API is query-only with conversation context; message control features are UI-only for V1
- Q: What encryption requirements apply to stored data? → A: Databricks platform encryption (at-rest and in-transit handled by platform); no application-layer or CMK for V1
- Q: What is the target service availability? → A: 99% availability (~87 hours downtime/year); basic reliability suitable for team-scale usage
- Q: How should concurrent same-user conflicts be handled? → A: Last-write-wins with automatic UI refresh on conflict detection

### Enhanced Model Routing Session 2025-12-21

Enhanced model routing configuration with hierarchical defaults and per-endpoint rate limiting:

- **Role-Level Defaults**: Model roles (e.g., simple, analytical, complex) define default parameters: temperature, max_tokens, reasoning_effort, rotation_strategy, fallback_on_429
- **Endpoint-Level Overrides**: Endpoints can override role defaults; max_context_window and tokens_per_minute are endpoint-specific (required)
- **Per-Endpoint Rate Limiting**: Each endpoint has its own tokens_per_minute budget; rate limiting is tracked independently
- **In-Memory Health Tracking**: EndpointHealth is maintained in-memory only (not persisted to DB); each process instance tracks independently
- **Reasoning Parameters**: Added reasoning_effort (low/medium/high) and reasoning_budget for o-series models

New FRs: FR-054 (reasoning effort/budget)
New Edge Cases: 2 (tokens_per_minute limit, endpoint override conflicts)
New Assumptions: 3 (Endpoint Health Scope, Token Rate Tracking, Config Hot Reload)

### Intelligent Query Understanding Session 2025-12-21

Added adaptive research and query intelligence capabilities:

- **Auto Research Depth**: System now defaults to "auto" mode where it estimates query complexity (simple/moderate/complex) and selects appropriate research depth automatically
- **Follow-up Question Handling**: System distinguishes between simple clarifications (answered from existing context), complex follow-ups (targeted additional research), and new topics (fresh research)
- **Proactive Clarification**: System detects ambiguous or underspecified queries and asks 1-3 clarifying questions before proceeding
- **Classification Transparency**: System displays its query classification reasoning to users (e.g., "This appears to be a complex topic requiring extended research")

New FRs: FR-045 to FR-053 (Query Intelligence section)
New Entity: QueryClassification
New Edge Cases: 4 (misclassification, ignored clarifications, follow-up detection errors, nuanced simple queries)
New Success Criteria: SC-015 to SC-018
New Assumptions: 3 (Query Classification, Follow-up Detection, Clarification UX)

### Authentication Implementation Session 2025-12-22

Implemented profile-based OAuth authentication for both LLM access and Lakebase:

- **Profile-Based Auth**: `DATABRICKS_CONFIG_PROFILE` is the preferred authentication method, enabling OAuth token generation via WorkspaceClient
- **LLM Client Auth**: Uses `WorkspaceClient.config.authenticate()` then `oauth_token().access_token` to get OAuth tokens for Databricks serving endpoints
- **Lakebase OAuth**: Uses `WorkspaceClient.database.generate_database_credential()` with instance names to get database tokens (1-hour lifetime, 5-minute refresh buffer)
- **Host Derivation**: Lakebase host is derived from instance name (`{LAKEBASE_INSTANCE_NAME}.database.cloud.databricks.com`), not configured separately
- **Fallback Support**: Direct `DATABRICKS_TOKEN` still supported for LLM access; `DATABASE_URL` available for local PostgreSQL development

Updated Configuration:
- `DATABRICKS_CONFIG_PROFILE`: Required for Lakebase OAuth, preferred for LLM access
- `LAKEBASE_INSTANCE_NAME`: Instance name only (not full hostname)
- `DATABRICKS_TOKEN`: Optional fallback (does not work with Lakebase)
- `DATABASE_URL`: Optional fallback for local PostgreSQL

### Research Activity Panel Session 2025-12-24

Improved the Research Activity panel (FR-014) to display human-readable event labels:

- **Event Formatting**: Raw event types (e.g., "agent_started", "step_completed") are transformed into user-friendly labels with emojis (e.g., "🔍 Analyzing query...", "✓ Found 3 sources")
- **Color Coding**: Events are color-coded by status: green (completed), amber (in-progress), blue (decisions), red (errors)
- **Duration Display**: Completed agent events show execution time (e.g., "✓ Analyzed (1.2s)")
- **Implementation**: Formatting logic in `frontend/src/utils/activityLabels.ts` with `formatActivityLabel()` and `getActivityColor()` functions

Agent labels:
- Coordinator: "🔍 Analyzing query..."
- Background Investigator: "📚 Background search..."
- Planner: "📋 Creating plan..."
- Researcher: "🔬 Researching..."
- Reflector: "🤔 Evaluating..."
- Synthesizer: "✍️ Writing report..."

### Central YAML Configuration Session 2025-12-24

Added central YAML-based configuration to consolidate all model and agent settings:

- **Central Config File**: All model endpoints, roles, agent limits, and search settings are defined in `config/app.yaml`
- **Model Tiers**: Added "micro" tier for ultra-lightweight operations (pattern matching, entity extraction) alongside existing simple/analytical/complex tiers
- **Priority-Based Routing**: Default endpoint selection strategy is priority-based (config order = priority order) with fallback on 429 errors
- **Agent Configuration**: Research limits (max_search_queries, max_urls_to_crawl, etc.) and plan iterations centralized
- **Search Configuration**: Brave API rate limits and freshness settings centralized
- **Environment Variables**: YAML supports `${ENV_VAR:-default}` interpolation for secrets and environment-specific values
- **Validation**: Configuration is validated at startup with clear error messages for invalid values
- **Fallback Defaults**: System works without config file using sensible defaults for development

New FRs: FR-081 to FR-090 (Central Configuration section)
New Entity: AppConfig
Updated Entities: ModelRole, ModelEndpoint (now loaded from YAML)

### Multi-Agent Architecture Session 2025-12-21

Adopted 5-agent architecture inspired by deer-flow reference implementation, with key differentiation in reflection pattern:

- **5 Specialized Agents**: Coordinator (simple tier), Planner (analytical tier), Researcher (analytical tier), Reflector (simple tier), Synthesizer (complex tier)
- **Plan-then-Execute Pattern**: Planner creates structured research plan with typed steps before execution begins
- **Step-by-Step Reflection**: Unlike deer-flow's batch-then-replan, we reflect AFTER EACH step to adapt in real-time
- **Background Investigation**: Quick web search (~5s) before planning to provide context for better research plans
- **JSON Repair**: Adopted from deer-flow for handling malformed LLM outputs
- **Orchestration Framework**: OPEN DECISION - Plain async Python recommended over LangGraph for simplicity

New FRs: FR-055 to FR-065 (Multi-Agent Architecture section)
New Entities: Plan, PlanStep, StepType, ReflectionResult
New Edge Cases: 4 (replan loops, empty plans, researcher failures, background investigation timeout)
New Assumptions: 5 (Agent Tiers, Reflection Pattern, Plan Iterations, JSON Repair, Orchestration)

## Assumptions

- **Databricks Integration**: The application runs within a Databricks workspace environment and uses profile-based authentication via `DATABRICKS_CONFIG_PROFILE`. WorkspaceClient is initialized with a CLI profile for OAuth token generation. Direct token auth via `DATABRICKS_TOKEN` is supported as fallback but profile-based auth is preferred for Lakebase OAuth.
- **Brave Search API**: A valid Brave Search API key is available for configuration. Rate limits follow Brave's standard tier limits.
- **LLM Access**: Multiple model endpoints are accessible via Databricks serving endpoints. Authentication uses OAuth tokens obtained via `WorkspaceClient.config.authenticate()` and `oauth_token().access_token`. The OpenAI client is configured with the workspace host and OAuth token. Both profile-based auth and direct token auth are supported.
- **Data Storage**: Chat conversations and user data are persisted in Databricks Lakebase (PostgreSQL). Authentication uses OAuth tokens generated via `WorkspaceClient.database.generate_database_credential()` with automatic token refresh (1-hour lifetime, 5-minute refresh buffer). Host is derived from instance name: `{LAKEBASE_INSTANCE_NAME}.database.cloud.databricks.com`. Fallback to direct `DATABASE_URL` is available for local development.
- **Session Duration**: Research operations may take 30-60 seconds for complex queries; users will wait for comprehensive results.
- **Data Retention**: Chat history is retained indefinitely unless deleted by the user. Deleted chats enter a 30-day soft-delete recovery window before permanent purge.
- **Concurrency**: System is designed for team-scale usage (10-100 concurrent users), not enterprise-wide deployment.
- **Model Configuration**: Model roles and endpoints are configured via central YAML file (`config/app.yaml`). Secrets can be injected via environment variable interpolation (`${ENV_VAR}`). No built-in admin UI is provided; end users do not modify model routing. System falls back to sensible defaults when config file is absent.
- **Endpoint Health Scope**: EndpointHealth is maintained in-memory per process instance; not shared across replicas. Each application instance independently tracks endpoint availability.
- **Token Rate Tracking**: tokens_per_minute is tracked per-endpoint with approximate counting based on request/response token estimates. Each endpoint has an independent rate limit budget.
- **Config Hot Reload**: Model configuration can be hot-reloaded from configuration files without application restart (file watch mechanism).
- **Structured Output**: Models support JSON mode or function calling for structured generation; fallback to parsing unstructured output when not available.
- **Context Truncation**: When truncation is required, the system prioritizes: system prompt > recent messages > older messages.
- **Observability**: MLflow is available in the Databricks workspace for comprehensive tracing of research operations, model calls, and tool executions.
- **Keyboard Shortcuts**: Keyboard shortcuts follow platform conventions (Cmd on Mac, Ctrl on Windows) and are documented in-app.
- **Export Generation**: Export generates client-side (no server rendering for PDFs initially); very large conversations may require Markdown-only export.
- **Audit Retention**: Audit logs are retained for minimum 1 year per enterprise compliance requirements.
- **Query Classification**: The LLM can reliably estimate query complexity and detect ambiguity based on the query text and conversation history alone; no external knowledge graph or specialized classifier is required.
- **Follow-up Detection**: Conversation context (recent messages) is sufficient to distinguish between clarifications, complex follow-ups, and new topics; explicit user signals (e.g., "new question:") are not required but can be used as hints.
- **Clarification UX**: Users are willing to answer 1-3 clarifying questions before research proceeds; overly long interrogation sequences should be avoided.
- **Data Encryption**: Data encryption at rest and in transit is handled by the Databricks platform. No application-layer encryption or customer-managed keys (CMK) required for V1.
- **Conflict Resolution**: When the same user accesses the application from multiple devices/tabs, last-write-wins semantics apply. UI detects conflicts and refreshes automatically; no real-time sync for V1.
- **Agent Model Tiers**: Agents are mapped to model tiers based on their complexity requirements: Coordinator/Background/Reflector use simple tier (fast, low-latency), Planner/Researcher use analytical tier (medium complexity), Synthesizer uses complex tier (deep reasoning).
- **Step-by-Step Reflection**: The Reflector agent evaluates after each research step, not after all steps complete. This allows real-time adaptation to discoveries but incurs more LLM calls. The Reflector uses a simple tier model to minimize cost.
- **Plan Iterations**: Maximum plan iterations defaults to 3. Each time Reflector decides ADJUST, the system returns to Planner which counts as one iteration. This prevents infinite replan loops.
- **JSON Repair**: LLM outputs may be malformed JSON, especially from smaller models. The system attempts repair using balanced brace extraction and the json_repair library before failing.
- **Orchestration Framework**: The multi-agent workflow can be implemented with plain async Python or LangGraph. Plain Python is recommended for simplicity unless checkpointing or graph visualization features are specifically needed.
- **Testing Isolation**: Unit tests are fully isolated from external dependencies (database, LLM APIs, HTTP services). All external interactions are mocked.
- **Test Data**: Test fixtures use deterministic data generators; no production data is used in tests. Fixtures are version-controlled alongside test code.
- **CI Environment**: Tests run in a CI environment with access to no external services by default. E2E tests against live services are optional and require explicit configuration.
- **Test Parallelization**: Unit tests can run in parallel; E2E tests run sequentially to avoid resource contention and flakiness.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete a deep research query and receive a synthesized answer within 2 minutes for typical questions.
- **SC-002**: Research answers include at least 3 cited sources for comprehensive queries.
- **SC-003**: Users can access their complete chat history from any device after re-authentication.
- **SC-004**: 95% of research queries complete successfully without user-visible errors.
- **SC-005**: Multiple users can conduct simultaneous research sessions without data cross-contamination.
- **SC-006**: Chat interface loads and displays conversation history within 3 seconds.
- **SC-007**: API consumers can integrate with the agent using standard REST patterns without custom client libraries.
- **SC-008**: System maintains 99% request success rate through automatic failover when any single model endpoint is unavailable.
- **SC-009**: Failover to alternative endpoints occurs within 5 seconds of detecting a rate limit or availability issue.
- **SC-010**: Research quality remains consistent regardless of which endpoint handles the request within a model role.
- **SC-011**: Stop/cancel operations take effect within 2 seconds of user request.
- **SC-012**: Chat search returns results within 1 second for typical queries across user's full history.
- **SC-013**: UI passes WCAG 2.1 AA automated accessibility audit (e.g., axe-core) with zero critical violations.
- **SC-014**: 100% of user actions (create, read, delete, export, edit) are logged for audit within 1 second of occurrence.
- **SC-015**: Query complexity classification is displayed to users within 3 seconds of query submission.
- **SC-016**: Simple clarification follow-ups (using existing context) respond within 5 seconds without initiating web searches.
- **SC-017**: System correctly identifies and responds to clarifying questions (avoiding unnecessary research) in at least 80% of conversational follow-ups.
- **SC-018**: When clarifying questions are asked, users can provide answers and receive appropriately scoped research within the standard response time (SC-001).
- **SC-019**: Service maintains 99% availability measured monthly (excluding scheduled maintenance windows).
- **SC-020**: Python unit test suite achieves minimum 80% line coverage for service layer modules.
- **SC-021**: Python unit test suite achieves minimum 70% line coverage for API endpoint modules.
- **SC-022**: Python unit test suite achieves minimum 60% line coverage for agent node modules.
- **SC-023**: All unit tests complete execution in under 60 seconds on standard development hardware.
- **SC-024**: E2E test suite covers all critical user journeys defined in User Story 9 acceptance scenarios.
- **SC-025**: E2E tests can run in both headed (debugging) and headless (CI) modes.
- **SC-026**: Test failures produce actionable diagnostics including logs, screenshots, and stack traces.
- **SC-027**: Frontend unit test suite achieves minimum 70% line coverage for custom hooks.
- **SC-028**: CI pipeline runs all tests (unit + E2E) and blocks merge on failure.
- **SC-029**: System starts successfully with default configuration when `config/app.yaml` is not present.
- **SC-030**: Configuration validation errors are reported with clear, actionable messages within 5 seconds of application startup.
