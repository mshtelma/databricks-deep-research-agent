# Feature Specification: Tiered Query Modes

**Feature Branch**: `004-tiered-query-modes`
**Created**: 2026-01-04
**Status**: Draft
**Input**: User description: "Three query modes (Simple, Web Search, Deep Research) with progressive disclosure UI where Deep Research reveals depth options (light/medium/extended), aligned with ChatGPT/Gemini alternatives. Enhanced activity display with centered research panel, detailed event labels, collapsible accordion after completion, and comprehensive visited sources tracking."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Simple Direct Answer (Priority: P1)

A user wants a quick factual answer or explanation without any web research. They ask a straightforward question like "What is the Pythagorean theorem?" or "Explain the difference between TCP and UDP" and expect an immediate, conversational response from the LLM.

**Why this priority**: This is the most common use case for casual users and represents the lowest-friction interaction. It provides immediate value without any waiting time and serves users who don't need sourced information.

**Independent Test**: Can be fully tested by selecting "Simple" mode, asking a general knowledge question, and receiving an immediate LLM response without any web search activity or citations.

**Acceptance Scenarios**:

1. **Given** the user is on the chat interface with "Simple" mode selected, **When** they submit a question "What is photosynthesis?", **Then** they receive a direct LLM-generated answer within 3 seconds with no citations, no research progress indicators, and no sources panel.

2. **Given** the user has "Simple" mode selected, **When** they submit a question, **Then** no web search requests are made and no research session is created in the backend.

3. **Given** the user has "Simple" mode selected, **When** they view the response, **Then** the response is displayed as conversational text without citation markers, verification stages, or source references.

---

### User Story 2 - Quick Web Search Answer (Priority: P1)

A user wants a quick answer backed by a few recent web sources. They ask something like "What's the current price of Bitcoin?" or "Who won the latest Super Bowl?" and want a fast response with 2-3 sources for verification.

**Why this priority**: This mode serves users who need current information or want source verification but don't require comprehensive research. It balances speed with credibility and is the most common web-enhanced interaction pattern.

**Independent Test**: Can be fully tested by selecting "Web Search" mode, asking a current events question, and receiving a sourced answer within 10-15 seconds with visible web search activity and 2-5 source citations.

**Acceptance Scenarios**:

1. **Given** the user is on the chat interface with "Web Search" mode selected, **When** they submit a question about current events, **Then** they see a brief activity indicator showing search progress and receive an answer with 2-5 inline citations within 15 seconds.

2. **Given** the user has "Web Search" mode selected, **When** they submit a question, **Then** the system performs 1-3 web searches and displays the sources used in the response.

3. **Given** the user has "Web Search" mode selected, **When** they view the response, **Then** the response includes clickable source links and the sources panel shows the retrieved web pages.

4. **Given** the user has "Web Search" mode selected, **When** web search fails or returns no results, **Then** the system falls back to providing a direct LLM answer with a notice that no sources were found.

---

### User Story 3 - Deep Research with Depth Selection (Priority: P2)

A user wants to conduct comprehensive research on a complex topic. They click "Deep Research" which reveals the existing depth options (Light/Medium/Extended), select their preferred depth, and submit their research query.

**Why this priority**: This serves power users who need thorough, multi-source analysis. It's more complex than the first two modes but essential for the application's core value proposition as a research tool.

**Independent Test**: Can be fully tested by selecting "Deep Research" mode, choosing a depth level, submitting a research query, and observing the full research workflow with plan, steps, reflections, and comprehensive final report.

**Acceptance Scenarios**:

1. **Given** the user clicks the "Deep Research" button, **When** the button becomes active, **Then** the depth selector (Light/Medium/Extended) appears below or beside the mode buttons.

2. **Given** the user has "Deep Research" mode active with "Medium" depth selected, **When** they submit a complex query, **Then** the full research pipeline executes (classification, planning, research steps, reflection, synthesis) with all progress indicators visible.

3. **Given** the user is in "Deep Research" mode, **When** they switch to "Simple" or "Web Search" mode, **Then** the depth selector disappears as it's no longer relevant.

4. **Given** the user has "Deep Research" mode with "Extended" depth, **When** they submit a research query, **Then** the system creates a research plan with 5-10 steps as configured for extended research.

---

### User Story 4 - Mode Persistence and Memory (Priority: P3)

A user's selected query mode persists within their session and optionally across sessions via their preferences. If they switch to "Web Search" mode, subsequent messages use that mode until changed.

**Why this priority**: Improves user experience by remembering preferences, but users can work around it by manually selecting each time. Nice-to-have for productivity but not essential for core functionality.

**Independent Test**: Can be fully tested by selecting a mode, sending multiple messages, closing and reopening the chat, and verifying the mode is preserved.

**Acceptance Scenarios**:

1. **Given** the user selects "Web Search" mode, **When** they send three consecutive messages, **Then** all three messages are processed in "Web Search" mode without requiring reselection.

2. **Given** the user has mode persistence enabled in preferences, **When** they close the browser and return later, **Then** their last selected mode is restored.

3. **Given** the user is in a new chat session, **When** no mode has been explicitly selected, **Then** the default mode from user preferences is applied (default: Simple).

---

### User Story 5 - Visual Mode Distinction (Priority: P3)

Users can easily distinguish which mode is currently active and understand what each mode does through clear visual design and helpful tooltips.

**Why this priority**: Improves usability and reduces confusion, but the core functionality works without polished visuals. Important for production quality but not for MVP.

**Independent Test**: Can be fully tested by hovering over each mode button and verifying tooltips appear, and by observing distinct visual styling for the selected vs unselected modes.

**Acceptance Scenarios**:

1. **Given** the user views the mode selector, **When** they see the three mode buttons, **Then** the currently selected mode is visually distinct (highlighted/filled) from unselected modes.

2. **Given** the user hovers over a mode button, **When** the tooltip appears, **Then** it displays a brief description of what that mode does.

3. **Given** the user is on a narrow screen (mobile), **When** they view the mode selector, **Then** the buttons are appropriately sized and the depth selector (when shown) doesn't overflow.

---

### User Story 6 - Centered Activity Panel During Research (Priority: P2)

During active Deep Research, the research activity panel (showing events like "claim_verified", "numeric_claim_detected", "verification_summary") is displayed in a centered, prominent position so users can easily monitor progress without looking at a sidebar.

**Why this priority**: Enhances visibility of research progress during the waiting period. Users currently need to look at a small sidebar; centering draws attention to meaningful activity updates.

**Independent Test**: Can be fully tested by starting a Deep Research query and observing that the activity panel appears centered in the main content area (not in a sidebar) while research is in progress.

**Acceptance Scenarios**:

1. **Given** the user submits a Deep Research query, **When** the research pipeline starts executing, **Then** the activity panel appears centered in the main content area below the user's message.

2. **Given** the research is in progress with the centered activity panel visible, **When** new events occur (step_started, claim_verified, etc.), **Then** the events appear in real-time in the centered panel with smooth entry animations.

3. **Given** the research is in progress, **When** the user scrolls the chat, **Then** the activity panel remains visible (sticky positioning) so users don't lose sight of progress.

4. **Given** the research completes, **When** the final report renders, **Then** the centered activity panel smoothly transitions to a collapsed accordion within the message card.

---

### User Story 7 - Enhanced Event Labels with Details (Priority: P2)

Research activity events display descriptive labels with contextual information instead of raw event type names. For example, "claim_verified" becomes "Claim Verified: The 2024 revenue exceeded $50B" with the claim text included.

**Why this priority**: Raw event names like "claim_verified" are meaningless to users. Descriptive labels with context help users understand what the system is doing and build trust in the research process.

**Independent Test**: Can be fully tested by triggering a Deep Research query and verifying that each event type displays a human-readable label with relevant context from the event data.

**Acceptance Scenarios**:

1. **Given** a `claim_verified` event occurs during research, **When** it appears in the activity panel, **Then** it displays as "Claim Verified: [first 60 chars of claim text]" with a checkmark icon and the verdict (supported/partial/unsupported).

2. **Given** a `numeric_claim_detected` event occurs, **When** it appears in the activity panel, **Then** it displays as "Numeric Claim: [value] [unit]" (e.g., "Numeric Claim: $2.5 billion revenue") with a number badge icon.

3. **Given** a `verification_summary` event occurs, **When** it appears in the activity panel, **Then** it displays as "Verification Complete: X supported, Y partial, Z unsupported" with a summary icon.

4. **Given** a `step_started` event occurs, **When** it appears in the activity panel, **Then** it displays as "Researching: [step title]" with a progress indicator.

5. **Given** a `tool_call` event for web_search occurs, **When** it appears in the activity panel, **Then** it displays as "Searching: [query text]" with a search icon.

---

### User Story 8 - Collapsible Activity Accordion After Completion (Priority: P2)

When research completes and the final report is displayed, all activity events from the research session are hidden within a collapsible accordion. Users can expand it to review the research journey if desired, but it doesn't clutter the completed message.

**Why this priority**: The activity log is valuable during research but becomes noise after completion. An accordion preserves the information for users who want to audit the process while keeping the UI clean.

**Independent Test**: Can be fully tested by completing a Deep Research query and verifying that the activity panel collapses into an expandable accordion in the message card.

**Acceptance Scenarios**:

1. **Given** a Deep Research query completes successfully, **When** the final report is rendered, **Then** the activity events are moved into a collapsed "Research Activity" accordion section at the bottom of the message card.

2. **Given** the activity accordion is collapsed, **When** the user clicks to expand it, **Then** all events from the research session are displayed in chronological order with their enhanced labels.

3. **Given** the activity accordion is expanded, **When** the user clicks to collapse it, **Then** it smoothly animates closed without affecting the rest of the message content.

4. **Given** a Deep Research query fails or is cancelled, **When** viewing the partial message, **Then** the activity accordion still shows all events that occurred before the failure, with the error event highlighted.

---

### User Story 9 - Comprehensive Visited Sources Display (Priority: P2)

The system tracks and displays ALL sources visited during research, not just those directly cited in claims. Users can see every URL the system crawled, searched, or considered, organized by research step.

**Why this priority**: Currently only cited sources are displayed. Users want transparency into what information the system consulted, even if it wasn't cited. This builds trust and allows users to explore tangential sources.

**Independent Test**: Can be fully tested by completing a Deep Research query and verifying that a "All Visited Sources" section shows more URLs than the "Cited Sources" section.

**Acceptance Scenarios**:

1. **Given** a Deep Research completes, **When** the user views the message, **Then** they see separate sections for "Cited Sources" (sources with claims) and "All Visited Sources" (every URL crawled).

2. **Given** the "All Visited Sources" section is displayed, **When** the user views it, **Then** sources are organized by the research step in which they were visited (e.g., "Step 1: Background Context", "Step 2: Revenue Data").

3. **Given** a source was visited but not cited, **When** it appears in "All Visited Sources", **Then** it displays the URL, page title, and a brief snippet of what was extracted, with a visual distinction from cited sources.

4. **Given** a source was both visited AND cited, **When** viewing sources, **Then** it appears in both sections with the citation section showing claim details and the visited section showing the extraction metadata.

5. **Given** some crawled pages returned errors or empty content, **When** viewing "All Visited Sources", **Then** failed pages are shown with a warning indicator and the error reason (e.g., "Page blocked crawling", "Timeout").

---

### Edge Cases

- What happens when a user switches mode mid-response? The current request continues in its original mode; the new mode applies to the next message.
- How does the system handle "Simple" mode for queries that truly need web data (e.g., "What's the weather today?")? Simple mode always uses LLM-only response; users must select Web Search for current data.
- What happens if Deep Research is selected but no depth is chosen? Default to "Auto" depth which lets the system decide based on query complexity.
- How does the system behave when the web search service is unavailable? Web Search mode falls back to Simple mode behavior with a notification. Deep Research shows an error and offers retry.
- What happens if the activity accordion contains hundreds of events? The accordion content is virtualized to show the most recent 50 events with a "Load more" option for older events.
- How are duplicate visited sources handled? Sources are deduplicated by URL; if the same URL is visited in multiple steps, it appears once with a note showing all steps that accessed it.
- What happens when Simple or Web Search mode is used? No activity accordion is shown for Simple mode. Web Search mode shows a minimal activity log inline (not in accordion) with just the search queries performed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support three query modes: Simple (LLM-only), Web Search (quick sourced answer), and Deep Research (full pipeline).
- **FR-002**: System MUST display mode selector as a button group in the chat input area, replacing the current depth-only selector.
- **FR-003**: System MUST show depth sub-selector (Auto/Light/Medium/Extended) only when Deep Research mode is selected.
- **FR-004**: Simple mode MUST generate responses using only the LLM without any web search or verification.
- **FR-005**: Simple mode MUST NOT create research sessions, sources, or claims in the database.
- **FR-006**: Web Search mode MUST perform 1-3 focused web searches to answer the query.
- **FR-007**: Web Search mode MUST complete within 15 seconds for typical queries.
- **FR-008**: Web Search mode MUST display inline citations linking to sources used.
- **FR-009**: Deep Research mode MUST execute the full research pipeline (coordinator, planner, researcher, reflector, synthesizer).
- **FR-010**: Deep Research mode MUST respect the selected depth level (Light/Medium/Extended) or auto-detect if "Auto" is selected.
- **FR-011**: System MUST persist the selected mode within the chat session.
- **FR-012**: System MUST include query mode in the API request payload.
- **FR-013**: System MUST display appropriate progress indicators for each mode (none for Simple, brief for Web Search, full for Deep Research).
- **FR-014**: System MUST allow users to set a default query mode in their preferences.
- **FR-015**: Web Search mode MUST store sources in the database for reference, similar to Deep Research but without full research session metadata.
- **FR-016**: System MUST handle mode-specific errors gracefully (e.g., web search timeout falls back to Simple mode with notification).

#### Centered Activity Panel (User Story 6)

- **FR-017**: System MUST display the research activity panel in a centered position within the main content area during active Deep Research (not in a sidebar).
- **FR-018**: System MUST position the activity panel directly below the user's message that triggered the research, within the message stream.
- **FR-019**: System MUST keep the activity panel visible with sticky positioning when the user scrolls during active research.
- **FR-020**: System MUST animate new events into the activity panel with smooth entry transitions (fade-in or slide-in within 200ms).
- **FR-021**: System MUST transition the activity panel from centered position to collapsed accordion when research completes.

#### Enhanced Event Labels (User Story 7)

- **FR-022**: System MUST display human-readable labels for all research events instead of raw event type names.
- **FR-023**: System MUST include contextual data in event labels (e.g., claim text for claim_verified, query text for tool_call).
- **FR-024**: System MUST truncate long claim text in event labels to 60 characters with ellipsis.
- **FR-025**: System MUST display appropriate icons for each event type (checkmark for verified, search icon for queries, warning for errors).
- **FR-026**: System MUST show verdict status (supported/partial/unsupported/contradicted) for claim_verified events with color-coded badges.
- **FR-027**: System MUST display numeric_claim_detected events with the extracted value and unit (e.g., "Numeric Claim: $2.5 billion").
- **FR-028**: System MUST display verification_summary events with aggregate counts (e.g., "Verification Complete: 12 supported, 3 partial, 1 unsupported").

#### Collapsible Activity Accordion (User Story 8)

- **FR-029**: System MUST collapse all research activity events into an accordion section within the message card when research completes.
- **FR-030**: System MUST default the activity accordion to collapsed state after research completion.
- **FR-031**: System MUST display a summary label on the collapsed accordion (e.g., "Research Activity (24 events)").
- **FR-032**: System MUST display all events in chronological order when the accordion is expanded.
- **FR-033**: System MUST animate accordion expand/collapse transitions smoothly (within 300ms).
- **FR-034**: System MUST preserve activity events in the accordion even when the research fails or is cancelled.
- **FR-035**: System MUST highlight error events with distinct styling when displayed in the accordion.
- **FR-036**: System MUST virtualize the accordion content when events exceed 50, with "Load more" pagination.

#### Comprehensive Visited Sources (User Story 9)

- **FR-037**: System MUST track all URLs visited during research, including those not directly cited in claims.
- **FR-038**: System MUST distinguish between "Cited Sources" (sources with claims attached) and "All Visited Sources" (every URL crawled).
- **FR-039**: System MUST display visited sources organized by the research step in which they were accessed.
- **FR-040**: System MUST display for each visited source: URL, page title, extraction snippet (first 150 chars), and step reference.
- **FR-041**: System MUST visually distinguish uncited sources from cited sources (e.g., muted styling, different icon).
- **FR-042**: System MUST show sources that appear in both sections with a "cited" badge in the visited sources list.
- **FR-043**: System MUST display failed page crawls with a warning indicator and error reason (blocked, timeout, not found).
- **FR-044**: System MUST deduplicate sources by URL when the same page is visited in multiple steps, showing all step references.
- **FR-045**: System MUST store visited sources metadata in the backend for persistence across page reloads.

#### Verify Sources Toggle (Source Verification Control)

- **FR-046**: System MUST display a "Verify sources" checkbox when Web Search or Deep Research mode is selected.
- **FR-047**: "Verify sources" checkbox MUST default to `false` for Web Search mode (prioritizing speed) and `true` for Deep Research mode (prioritizing accuracy).
- **FR-048**: When "Verify sources" is enabled, system MUST run the full citation verification pipeline (claim extraction, evidence selection, verification).
- **FR-049**: When "Verify sources" is disabled, system MUST use classical synthesis with markdown-style citations `[Title](url)` without claim-level verification.
- **FR-050**: System MUST pass the `verify_sources` parameter through the API to control the backend verification pipeline.
- **FR-051**: The checkbox state MUST reset to the mode-appropriate default when the user switches query modes.

#### Snippet Fallback for Failed Web Fetches

- **FR-052**: System MUST use Brave Search snippets as fallback evidence when web page fetching fails (blocked, timeout, parsing error).
- **FR-053**: System MUST include snippets in the evidence pool even when full page content is available, allowing the evidence selector to use them.
- **FR-054**: Snippet-based evidence MUST be marked with `is_snippet_based: true` and assigned lower confidence scores (0.5) compared to full-content evidence.
- **FR-055**: Sources with only snippets (no fetched content) MUST bypass content quality filtering and be included in the evidence pool.

### Message Export

- **FR-056**: Agent messages MUST display a 3-dot menu button in the top-right corner for export options (Export Report, Verification Report, Copy to Clipboard).
- **FR-057**: "Export Report" MUST download the agent synthesis as a standalone markdown file with title, metadata, content, and sources list.
- **FR-058**: "Verification Report" MUST download claims with verification verdicts and evidence quotes as markdown. This option MUST only appear when claims exist.
- **FR-059**: "Copy to Clipboard" MUST copy the report markdown content to the system clipboard with a toast notification on success.
- **FR-060**: The export menu MUST NOT appear while the message is streaming or when the message ID is not a valid UUID (e.g., placeholder IDs).

### Key Entities *(include if feature involves data)*

- **QueryMode**: Enumeration representing the three query modes (simple, web_search, deep_research). Stored per message and in user preferences.
- **Message**: Extended to include query_mode field indicating which mode was used for that message.
- **UserPreferences**: Extended to include default_query_mode setting.
- **ResearchSession**: Created for both Deep Research and Web Search modes. Deep Research creates full sessions with plan, steps, claims, and sources. Web Search creates lightweight sessions containing only sources (no plan, steps, or claims). Simple mode does not create sessions.
- **ResearchEvent**: Individual activity event during research (step_started, claim_verified, tool_call, etc.). Contains event_type, timestamp, and event-specific payload data. Stored per research session for accordion display. Retention: deleted when parent chat is deleted.
- **Source (Extended)**: Existing Source model extended with visited source fields: `is_cited` (boolean to distinguish cited vs visited-only), `step_index` and `step_title` (research step reference), `crawl_status` (success/failed/timeout/blocked), and `error_reason` (if crawl failed). This approach reuses the existing Source table rather than creating a separate VisitedSource table. Retention: deleted when parent research session is deleted.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete a Simple mode query in under 3 seconds (perceived response time from submit to first content).
- **SC-002**: Users can complete a Web Search mode query in under 15 seconds with at least 2 visible source citations.
- **SC-003**: The mode selection UI is discoverable by 90% of new users on first visit. Measurement: Track `mode_selector_interaction` analytics event on first session; success = user changes mode OR hovers mode button within first 60 seconds. Target: 90% of new users trigger this event before submitting their first query.
- **SC-004**: Mode switching between Simple/Web Search and Deep Research is instantaneous (under 100ms) with depth selector appearing/disappearing smoothly.
- **SC-005**: At least 80% of queries are correctly served by the selected mode (users don't need to retry with a different mode).
- **SC-006**: System maintains sub-5-second time-to-first-byte for Simple mode responses.
- **SC-007**: Users report satisfaction with query result quality matching their mode selection expectations (qualitative feedback).
- **SC-008**: The centered activity panel is visible to users within 1 second of starting Deep Research, improving perceived responsiveness.
- **SC-009**: Users can understand what the system is doing at any point during research by reading activity labels (measured by user comprehension testing).
- **SC-010**: 100% of research events are preserved in the collapsed accordion after completion, accessible via expand.
- **SC-011**: Users discover the "All Visited Sources" section and find it useful for understanding research scope (measured by click-through rate on expand).
- **SC-012**: Activity accordion expand/collapse animations complete within 300ms with no jank or layout shift.
- **SC-013**: The visited sources section displays at least 50% more unique URLs than the cited sources section (demonstrating comprehensive tracking).

## Clarifications

### Session 2026-01-04

- Q: How long should ResearchEvent and VisitedSource records be retained? → A: Retain for chat session lifetime (delete when chat deleted)
- Q: Should Web Search mode create a session for source persistence? → A: Create lightweight session (sources only, no plan/steps/claims)

## Assumptions

### Architecture: Maximum Code Reuse

- **Simple Mode**: Reuses existing `is_simple_query` detection path in orchestrator (100% existing code).
- **Web Search Mode**: Reuses existing `researcher.py` with `mode=classic` and `synthesizer.py` with `generation_mode=natural`. No new agent class required - only ~50 lines of routing logic in orchestrator.
- **Deep Research Mode**: Full pipeline unchanged (coordinator → planner → researcher → reflector → synthesizer with citation verification).

### Existing Components Reused

- The existing `ResearchDepth` enum (auto, light, medium, extended) is reused for Deep Research sub-selection.
- The existing researcher's `mode=classic` provides single-pass execution with configurable limits (`max_search_queries`, `max_urls_to_crawl`).
- The existing synthesizer's `generation_mode=natural` outputs [1], [2] style numeric citations suitable for Web Search mode.
- The existing `is_simple_query` path handles Simple mode without creating research sessions.
- User preferences storage already supports adding new fields (default_query_mode).

### Infrastructure

- The Brave Search API is used for both Web Search mode and Deep Research mode web searches.
- The frontend uses the existing component architecture (React, TanStack Query, Tailwind).
- The existing SSE streaming infrastructure supports all event types needed for activity tracking (agent_started, claim_verified, tool_call, etc.).
- The existing `activityLabels.ts` utility can be extended to support enhanced label formatting with event-specific context.
- Backend already captures crawled URLs via tool_result events; visited sources tracking extends this existing data flow.
- The database schema can be extended to store research events and visited sources metadata without breaking existing functionality.
- Animation and transition effects use CSS transitions or React animation libraries already available in the frontend stack.
