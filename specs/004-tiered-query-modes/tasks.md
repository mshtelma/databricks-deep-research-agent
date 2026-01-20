# Tasks: Tiered Query Modes

**Input**: Design documents from `/specs/004-tiered-query-modes/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested in the feature specification. Test tasks are omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `src/` at repository root (Python 3.11+, FastAPI, Pydantic v2)
- **Frontend**: `frontend/src/` (React 18, TypeScript 5.x, TanStack Query, Tailwind)
- **Config**: `config/app.yaml` for centralized configuration

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and shared type definitions

- [X] T001 Add QueryMode enum to src/models/enums.py following ResearchDepth pattern
- [X] T002 [P] Add QueryMode TypeScript type to frontend/src/types/index.ts
- [X] T003 [P] Add query_modes configuration section to config/app.yaml with simple/web_search/deep_research settings
- [X] T004 Add QueryModeConfig Pydantic model to src/core/app_config.py
- [X] T005 Add get_query_mode_config() accessor to src/agent/config.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**Important**: No user story work can begin until this phase is complete

- [X] T006 Create database migration src/db/migrations/versions/007_tiered_query_modes.py with:
  - query_mode column on research_sessions table
  - default_query_mode column on user_preferences table
  - is_cited, step_index, step_title, crawl_status, error_reason columns on sources table
  - New research_events table with id, research_session_id, event_type, timestamp, payload, created_at
  - Indexes for research_events (session, timestamp)
  - Indexes for sources (is_cited, step_index)
- [X] T007 Update ResearchSession model in src/models/research_session.py:
  - Add query_mode field (VARCHAR(20), default='deep_research')
  - Add events relationship to ResearchEvent
- [X] T008 [P] Create ResearchEvent model in src/models/research_event.py with SQLAlchemy mapped columns
- [X] T009 [P] Update UserPreferences model in src/models/user_preferences.py:
  - Add default_query_mode field (VARCHAR(20), default='simple')
- [X] T010 Update Source model in src/models/source.py:
  - Add is_cited, step_index, step_title, crawl_status, error_reason fields per data-model.md
- [X] T011 Add query_mode field to ResearchState in src/agent/state.py
- [X] T012 Update OrchestrationConfig in src/agent/orchestrator.py to include query_mode parameter
- [X] T013 Add query_mode parameter to stream research API in src/api/v1/research.py
- [X] T014 [P] Update preferences API in src/api/v1/preferences.py to expose default_query_mode
- [X] T015 Create ResearchEventService in src/services/research_event_service.py for event persistence

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Simple Direct Answer (Priority: P1) - MVP

**Goal**: Users can get quick LLM-only responses without web search by selecting "Simple" mode

**Independent Test**: Select "Simple" mode, ask "What is photosynthesis?", receive immediate LLM response with no citations, no research progress, no sources panel within 3 seconds

### Implementation for User Story 1

- [X] T016 [US1] Add Simple mode routing in orchestrator src/agent/orchestrator.py:
  - If query_mode == SIMPLE, call existing is_simple_query path
  - Stream direct LLM response via synthesis_progress events only
  - Skip session creation, source storage, and all research events
- [X] T017 [US1] Create QueryModeSelector component in frontend/src/components/chat/QueryModeSelector.tsx:
  - Button group with Simple, Web Search, Deep Research options
  - Visual highlight for selected mode
  - Disable during streaming
- [X] T018 [US1] Create useQueryMode hook in frontend/src/hooks/useQueryMode.ts:
  - Initialize from localStorage or user preferences
  - Persist mode changes to localStorage
  - Return { mode, setQueryMode, isDeepResearch }
- [X] T019 [US1] Integrate QueryModeSelector into frontend/src/components/chat/MessageInput.tsx:
  - Position above or beside text input
  - Pass selected mode to sendQuery
- [X] T020 [US1] Update useStreamingQuery hook in frontend/src/hooks/useStreamingQuery.ts:
  - Add queryMode parameter to sendQuery signature
  - Include query_mode in API request
- [X] T021 [US1] Update AgentMessage rendering in frontend/src/components/chat/AgentMessage.tsx:
  - Hide sources panel when query_mode is 'simple'
  - Hide activity panel/accordion when query_mode is 'simple'

**Checkpoint**: At this point, User Story 1 (Simple mode) should be fully functional and testable independently

---

## Phase 4: User Story 2 - Quick Web Search Answer (Priority: P1)

**Goal**: Users can get quick sourced answers with 2-5 citations by selecting "Web Search" mode

**Independent Test**: Select "Web Search" mode, ask "What's the current price of Bitcoin?", receive answer with inline [1], [2] citations and sources panel within 15 seconds

### Implementation for User Story 2

- [X] T022 [US2] Add Web Search mode routing in orchestrator src/agent/orchestrator.py (~50 lines):
  - Create minimal 1-step plan programmatically
  - Configure researcher with mode=classic, max_search_queries=2, max_urls_to_crawl=3
  - Skip reflector (always COMPLETE after 1 step)
  - Call synthesizer with generation_mode=natural for [1], [2] citations
  - Create lightweight session with sources only
- [X] T023 [US2] Add mode-specific event filtering in src/schemas/streaming.py:
  - Define WEB_SEARCH_EVENTS set (agent_started, tool_call, tool_result, synthesis_started, synthesis_progress, research_completed)
  - Create should_emit_event(event, mode) filter function
- [X] T024 [US2] Update frontend to show brief activity for Web Search mode in frontend/src/components/chat/AgentMessage.tsx:
  - Show inline activity log (not accordion) with search queries performed
  - Display sources panel with [1], [2] style source references
- [X] T025 [US2] Add Web Search mode timeout handling in src/agent/orchestrator.py:
  - Implement 15-second timeout
  - Fallback to Simple mode response with notification if timeout exceeded

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Deep Research with Depth Selection (Priority: P2)

**Goal**: Users can conduct comprehensive research with depth selection (Light/Medium/Extended) via "Deep Research" mode

**Independent Test**: Select "Deep Research" mode, choose "Medium" depth, submit complex query, observe full pipeline execution with plan, steps, reflections, and comprehensive report

### Implementation for User Story 3

- [X] T026 [US3] Add progressive disclosure for depth selector in frontend/src/components/chat/QueryModeSelector.tsx:
  - Show ResearchDepthSelector only when Deep Research is selected
  - Animate depth selector appearance/disappearance
- [X] T027 [US3] Integrate depth selector with mode selector in frontend/src/components/chat/MessageInput.tsx:
  - Conditionally render depth selector: {queryMode === 'deep_research' && <ResearchDepthSelector />}
  - Pass both queryMode and researchDepth to sendQuery
- [X] T028 [US3] Update API to validate research_depth only applies to deep_research mode in src/api/v1/research.py:
  - Return 400 if research_depth provided but query_mode is not 'deep_research'
  - Default to 'auto' depth if not specified for deep_research mode
- [X] T029 [US3] Ensure Deep Research mode uses full pipeline in src/agent/orchestrator.py:
  - Route to existing _stream_full_research when query_mode == DEEP_RESEARCH
  - Respect selected depth level from config

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently

---

## Phase 6: User Story 4 - Mode Persistence and Memory (Priority: P3)

**Goal**: User's selected mode persists within session and optionally across sessions via preferences

**Independent Test**: Select "Web Search" mode, send 3 messages (all use Web Search), close browser, return and verify mode is restored

### Implementation for User Story 4

- [X] T030 [US4] Update useQueryMode hook to sync with user preferences in frontend/src/hooks/useQueryMode.ts:
  - Load default from UserPreferences on first mount
  - Provide updateDefaultMode function that calls preferences API
- [X] T031 [US4] Add "Set as default" option to QueryModeSelector in frontend/src/components/chat/QueryModeSelector.tsx:
  - Show option on long-press or via dropdown (hook provides setModeAsDefault)
  - Call updateDefaultMode when selected
- [X] T032 [US4] Persist mode to user preferences API in src/api/v1/preferences.py:
  - Accept default_query_mode in PUT request body
  - Validate against QueryMode enum values
- [ ] T033 [US4] Add mode to Message model response in src/api/v1/messages.py:
  - Include query_mode from research_session in message response
  - Show historical mode used for each message

**Checkpoint**: Mode persistence works within and across sessions

---

## Phase 7: User Story 5 - Visual Mode Distinction (Priority: P3)

**Goal**: Users can easily distinguish active mode and understand each mode via visual design and tooltips

**Independent Test**: Hover over each mode button, verify tooltips appear with mode descriptions, observe distinct visual styling for selected vs unselected modes

### Implementation for User Story 5

- [X] T034 [US5] Add tooltips to QueryModeSelector in frontend/src/components/chat/QueryModeSelector.tsx:
  - Simple: "Quick LLM response without web search"
  - Web Search: "Fast answer with 2-5 web sources"
  - Deep Research: "Comprehensive multi-step research"
- [X] T035 [US5] Enhance visual styling in QueryModeSelector:
  - Selected mode: filled/highlighted background
  - Unselected modes: outline/muted styling
  - Hover states with subtle background change
  - Added icons for each mode (lightning, magnifying glass, microscope)
- [X] T036 [US5] Add responsive styling for mobile screens:
  - Ensure buttons don't overflow on narrow screens
  - Adjust depth selector layout for mobile
  - Test at 320px width minimum

**Checkpoint**: Mode selector is visually polished with clear affordances

---

## Phase 8: User Story 6 - Centered Activity Panel During Research (Priority: P2)

**Goal**: During Deep Research, activity panel displays centered in main content area with sticky positioning

**Independent Test**: Start Deep Research query, observe activity panel appears centered below user message, stays visible while scrolling, transitions smoothly to accordion on completion

### Implementation for User Story 6

- [X] T037 [US6] Create CenteredActivityPanel component in frontend/src/components/research/CenteredActivityPanel.tsx:
  - Position centered below user message during research
  - Use sticky positioning for scroll behavior
  - Accept events array and isLive prop
- [ ] T038 [US6] Create useResearchEvents hook in frontend/src/hooks/useResearchEvents.ts:
  - Accumulate events during streaming
  - Track isStreaming state
  - Provide events array for both centered panel and accordion
- [ ] T039 [US6] Integrate CenteredActivityPanel into ChatPage in frontend/src/pages/ChatPage.tsx:
  - Show CenteredActivityPanel when isStreaming && queryMode === 'deep_research'
  - Pass accumulated events from useResearchEvents
- [X] T040 [US6] Add smooth entry animations for new events (200ms fade-in):
  - Use CSS transitions or Framer Motion
  - Animate new events sliding in from bottom

**Checkpoint**: Centered activity panel works during Deep Research

---

## Phase 9: User Story 7 - Enhanced Event Labels with Details (Priority: P2)

**Goal**: Research events display descriptive human-readable labels with contextual information

**Independent Test**: Run Deep Research, observe "claim_verified" shows as "Claim Verified: [truncated claim text]" with verdict badge, "tool_call" shows as "Searching: [query text]"

### Implementation for User Story 7

- [X] T041 [US7] Create EnhancedEventLabel component in frontend/src/components/research/EnhancedEventLabel.tsx:
  - Accept StreamEvent and render with icon, text, optional badge
  - Support all event types per research.md section 8
- [X] T042 [US7] Extend activityLabels utility in frontend/src/utils/activityLabels.ts:
  - Add formatEnhancedEventLabel function
  - Add getVerdictIcon, getVerdictColor helpers
  - Implement truncate utility (60 chars for claims)
- [X] T043 [US7] Add icons for each event type:
  - claim_verified: checkmark (green/amber/red based on verdict)
  - numeric_claim_detected: hash badge
  - verification_summary: summary icon
  - step_started: progress indicator
  - tool_call/web_search: search icon
  - tool_call/web_crawl: globe icon
- [ ] T044 [US7] Update backend to include enhanced payload data in SSE events:
  - Add claim_text_truncated (60 chars) to claim_verified events
  - Add display_value to numeric_claim_detected events
  - Add display_text to verification_summary events

**Checkpoint**: Enhanced event labels display contextual information

---

## Phase 10: User Story 8 - Collapsible Activity Accordion After Completion (Priority: P2)

**Goal**: When research completes, activity events collapse into expandable accordion in message card

**Independent Test**: Complete Deep Research, verify centered panel transitions to collapsed accordion, expand to see all events in chronological order, collapse smoothly

### Implementation for User Story 8

- [X] T045 [US8] Create ActivityAccordion component in frontend/src/components/research/ActivityAccordion.tsx:
  - Use CSS-based accordion (native, no external deps)
  - Show summary label "Research Activity (N events)"
  - Default to collapsed state
  - Display events in chronological order when expanded
- [ ] T046 [US8] Add transition from centered panel to accordion:
  - On research_completed event, animate centered panel exit
  - Render ActivityAccordion in AgentMessage with same events
- [X] T047 [US8] Implement accordion animations (300ms expand/collapse):
  - Use CSS transitions for height animation
  - Ensure no layout shift during animation
- [X] T048 [US8] Add error event highlighting in accordion:
  - Distinct red styling for error events
  - Error events remain visible even when research fails
- [ ] T049 [US8] Persist events to database via ResearchEventService:
  - Call save_events after each SSE event
  - Load events from database when loading completed messages
- [ ] T050 [US8] Add include_events query param to messages API in src/api/v1/messages.py:
  - When include_events=true, join ResearchEvent records
  - Return events array in research_session response

**Checkpoint**: Activity accordion works for completed research

---

## Phase 11: User Story 8 (continued) - Virtualization for Large Event Sets

**Goal**: Accordion handles hundreds of events efficiently with virtualization

**Independent Test**: Complete research with >100 events, verify accordion virtualizes content with "Load more" pagination

### Implementation for User Story 8 (Virtualization)

- [ ] T051 [US8] Create VirtualizedList component in frontend/src/components/common/VirtualizedList.tsx:
  - Use react-window FixedSizeList
  - Threshold: virtualize when events > 50
- [ ] T052 [US8] Add "Load more" pagination to VirtualizedList:
  - Initially show 50 events
  - Button to load 50 more at a time
  - Show remaining count
- [ ] T053 [US8] Integrate VirtualizedList into ActivityAccordion:
  - Conditionally render VirtualizedList when events.length > 50
  - Use SimpleEventList for <= 50 events

**Checkpoint**: Accordion scales to large event counts

---

## Phase 12: User Story 9 - Comprehensive Visited Sources Display (Priority: P2)

**Goal**: System tracks and displays ALL sources visited during research, not just cited ones

**Independent Test**: Complete Deep Research, verify "All Visited Sources" section shows more URLs than "Cited Sources", sources organized by research step, failed crawls show warnings

### Implementation for User Story 9

- [ ] T054 [US9] Update source tracking in researcher to mark visited vs cited:
  - Set is_cited=false initially when source is crawled
  - Update is_cited=true when source is used in synthesis
  - Store step_index and step_title for each source
- [ ] T055 [US9] Track crawl failures in Source model:
  - Set crawl_status='failed'/'timeout'/'blocked' on error
  - Store error_reason with failure message
- [X] T056 [US9] Create VisitedSourcesPanel component in frontend/src/components/research/VisitedSourcesPanel.tsx:
  - Separate sections: "Cited Sources" and "All Visited Sources"
  - Group visited sources by step_index with step_title headers
  - Show URL, title, snippet (150 chars), step reference
- [X] T057 [US9] Add visual distinction for uncited vs cited sources:
  - Muted styling for uncited sources
  - "Cited" badge for sources in both sections
  - Warning icon for failed crawls with error reason tooltip
- [ ] T058 [US9] Handle source deduplication:
  - When same URL visited in multiple steps, show once with all step references
  - Backend dedupes by URL, tracks all step_indices
- [ ] T059 [US9] Integrate VisitedSourcesPanel into AgentMessage:
  - Show for both Web Search and Deep Research modes
  - Replace existing sources display with new component

**Checkpoint**: All visited sources are tracked and displayed comprehensively

---

## Phase 13: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T060 [P] Performance optimization: Ensure Simple mode <3s TTFB by precomputing common paths
- [ ] T061 [P] Add CSS will-change hints for activity panel animations to prevent jank
- [ ] T062 Implement mode switching edge case: current request continues in original mode, new mode applies to next message
- [ ] T063 Add fallback behavior when web search service unavailable: Web Search falls back to Simple with notification
- [ ] T064 Add mode_selected SSE event at start of stream indicating query_mode and research_depth
- [X] T065 Run mypy strict mode on all new Python files
- [X] T066 Run TypeScript strict mode typecheck on all new frontend files
- [ ] T067 Test animations on low-end devices (200ms events, 300ms accordion)
- [ ] T068 Run database migration on staging Lakebase and verify data migration
- [ ] T069 Run quickstart.md validation to ensure developer setup works

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-12)**: All depend on Foundational phase completion
  - US1 (Simple) and US2 (Web Search) are both P1 priority - can proceed in parallel
  - US3-US9 can proceed after P1 stories or in parallel with sufficient staffing
- **Polish (Phase 13)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (Simple Mode, P1)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 2 (Web Search, P1)**: Can start after Phase 2 - No dependencies on other stories
- **User Story 3 (Deep Research, P2)**: Can start after Phase 2 - Benefits from US1/US2 UI components
- **User Story 4 (Persistence, P3)**: Depends on US1-US3 for mode selector to exist
- **User Story 5 (Visual Polish, P3)**: Depends on US1-US3 for mode selector to exist
- **User Story 6 (Centered Panel, P2)**: Can start after Phase 2 - Benefits from US3 deep research flow
- **User Story 7 (Enhanced Labels, P2)**: Can start after US6 centered panel exists
- **User Story 8 (Accordion, P2)**: Depends on US6 centered panel and US7 enhanced labels
- **User Story 9 (Visited Sources, P2)**: Can start after Phase 2 - Independent frontend work

### Within Each User Story

- Models before services
- Services before API endpoints
- Backend before frontend (for API contracts)
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Setup Phase:**
- T002 (frontend types) and T003 (config) can run in parallel
- T004 (config model) depends on T003

**Foundational Phase:**
- T008 (ResearchEvent model) and T009 (UserPreferences update) can run in parallel
- Migration T006 should run first, then model updates

**User Stories:**
- US1 and US2 are both P1 and can be implemented in parallel
- Frontend tasks within a story marked [P] can run in parallel
- US6 (Centered Panel), US7 (Labels), US8 (Accordion) form a dependency chain
- US9 (Visited Sources) is largely independent and can run in parallel with US6-US8

---

## Parallel Example: User Story 1 + User Story 2

```bash
# After Foundational phase completes, launch both P1 stories:

# Developer A: User Story 1 (Simple Mode)
Task: "T016 [US1] Add Simple mode routing in orchestrator"
Task: "T017 [US1] Create QueryModeSelector component"
Task: "T018 [US1] Create useQueryMode hook"
# ...continues with US1 tasks

# Developer B: User Story 2 (Web Search Mode)
Task: "T022 [US2] Add Web Search mode routing in orchestrator"
Task: "T023 [US2] Add mode-specific event filtering"
# ...continues with US2 tasks
```

---

## Parallel Example: Activity Display Stories

```bash
# After US3 (Deep Research) is complete:

# Developer A: User Story 6 + 7 + 8 (Activity Panel chain)
Task: "T037 [US6] Create CenteredActivityPanel component"
Task: "T041 [US7] Create EnhancedEventLabel component"
Task: "T045 [US8] Create ActivityAccordion component"

# Developer B: User Story 9 (Visited Sources) - independent
Task: "T054 [US9] Update source tracking in researcher"
Task: "T056 [US9] Create VisitedSourcesPanel component"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Simple Mode)
4. Complete Phase 4: User Story 2 (Web Search Mode)
5. **STOP and VALIDATE**: Test both modes independently
6. Deploy/demo if ready - users can select Simple or Web Search

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 (Simple) -> Test independently -> Deploy/Demo (MVP!)
3. Add User Story 2 (Web Search) -> Test independently -> Deploy/Demo
4. Add User Story 3 (Deep Research) -> Test independently -> Deploy/Demo
5. Add User Stories 6-8 (Activity Display) -> Test independently -> Deploy/Demo
6. Add User Story 9 (Visited Sources) -> Test independently -> Deploy/Demo
7. Add User Stories 4-5 (Persistence & Polish) -> Test independently -> Deploy/Demo
8. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Simple)
   - Developer B: User Story 2 (Web Search)
3. After P1 stories complete:
   - Developer A: User Story 3 (Deep Research depth selector)
   - Developer B: User Story 6 (Centered Panel)
4. Continue with remaining stories based on priority and dependencies

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- **Reuse Strategy**: Web Search mode reuses existing researcher (mode=classic) and synthesizer (generation_mode=natural) - no new agent class needed (~50 lines routing vs ~500 lines new agent)
