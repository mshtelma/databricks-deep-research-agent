# Feature Specification: Claim-Level Citation Granularity

**Feature Branch**: `003-claim-level-citations`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "Citation granularity - Every user-visible claim traced to specific evidence with evidence cards, numeric claim handling, and verification pipeline"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - View Evidence Card for Any Claim (Priority: P1)

A researcher reading a research report wants to verify the accuracy of a specific claim. When they click or hover on any inline citation marker, they see a rich "evidence card" that shows the exact source passage supporting that claim, allowing them to quickly verify without leaving the current view.

**Why this priority**: This is the core value proposition - users need to trace claims to evidence. Without evidence cards, claims remain opaque and trust is diminished.

**Independent Test**: Can be fully tested by clicking any citation in a research response and verifying the evidence card displays complete source metadata and supporting quote.

**Acceptance Scenarios**:

1. **Given** a research response contains a claim with a citation marker, **When** the user clicks on the citation, **Then** an evidence card appears showing source title, author/publisher (if available), publication date, URL, the exact supporting quote (highlighted), and location within source (page/section).

2. **Given** an evidence card is displayed, **When** the user views the support assessment, **Then** they see one of: "Supported", "Partially supported", or "Not supported" based on verification analysis.

3. **Given** a research response with multiple claims in one sentence, **When** the user clicks different citation markers, **Then** each shows a distinct evidence card pointing to its specific supporting passage.

4. **Given** a user is viewing an evidence card, **When** they click outside the card or press Escape, **Then** the card closes and they return to reading the report.

---

### User Story 2 - Inspect Numeric Claims with Full Provenance (Priority: P1)

A financial analyst reviewing a research report sees numeric data (revenue figures, percentages, dates). They need absolute certainty about the source and any transformations applied to these numbers. Numeric claims appear as interactive chips that reveal complete provenance.

**Why this priority**: Numeric claims are highest-stakes for verification - users cite exact figures in their own work and need bulletproof provenance.

**Independent Test**: Can be fully tested by finding a numeric claim in a response and expanding it to verify quote, normalization, and derivation information display correctly.

**Acceptance Scenarios**:

1. **Given** a research response contains a numeric claim (e.g., "$3.2B revenue"), **When** the system generates the response, **Then** numeric claims are automatically detected and displayed as interactive chips with distinct visual styling.

2. **Given** a numeric claim chip is displayed, **When** the user clicks to expand it, **Then** they see: the exact quote from the source containing the number, the normalized value (e.g., "3,200,000,000 USD"), and any assumptions applied (currency year, exchange rate, rounding).

3. **Given** a numeric value was computed from multiple sources rather than directly quoted, **When** the user expands the claim, **Then** it displays a "Derived" label, lists all input values with their citations, and shows the calculation steps.

4. **Given** a numeric claim with unit conversion (e.g., "approximately 1.5 million users" from "1,487,230 users"), **When** the user expands the chip, **Then** both the original value and the converted/rounded value are shown with the transformation noted.

---

### User Story 3 - Atomic Claim Decomposition (Priority: P2)

A fact-checker reviewing a complex research response notices that some sentences contain multiple distinct factual assertions. The system automatically decomposes multi-fact sentences into atomic claims, each with its own citation, so the fact-checker can verify each assertion independently.

**Why this priority**: Multi-fact sentences with single document-level citations create verification ambiguity. Claim-level granularity enables precise fact-checking.

**Independent Test**: Can be fully tested by submitting a query that produces multi-fact responses and verifying each atomic claim has a distinct, clickable citation.

**Acceptance Scenarios**:

1. **Given** a sentence contains multiple factual assertions (e.g., "Company X was founded in 2010 and is headquartered in New York"), **When** the response is generated, **Then** each assertion has its own inline citation marker that links to specific evidence.

2. **Given** a complex paragraph with 5+ claims, **When** rendered in the UI, **Then** citation markers are visually distinct and do not clutter the reading experience.

3. **Given** a claim that spans multiple source passages, **When** the user clicks the citation, **Then** the evidence card shows all relevant supporting passages, not just one.

---

### User Story 4 - Verification Status Overview (Priority: P2)

A research manager wants a quick summary of citation quality for a response. They can view an overview showing how many claims are "Supported", "Partially supported", or "Not supported", helping them assess overall research reliability at a glance.

**Why this priority**: Aggregate verification status provides meta-level quality assurance without requiring manual review of every claim.

**Independent Test**: Can be fully tested by generating a research response and viewing the verification summary panel to confirm claim counts and percentages.

**Acceptance Scenarios**:

1. **Given** a research response with multiple cited claims, **When** the user views the verification summary, **Then** they see total claim count and breakdown by support level (Supported/Partially/Not supported).

2. **Given** a verification summary is displayed, **When** the user clicks on a support category, **Then** all claims in that category are highlighted in the response.

3. **Given** a response with low verification scores (>20% not supported), **When** displayed, **Then** a warning indicator appears to alert the user to potential reliability issues.

---

### User Story 5 - Structured Provenance Export (Priority: P3)

A compliance officer needs to document the evidence chain for regulatory purposes. They can export the full structured provenance data (all claims, citations, evidence spans, verification results) in a machine-readable format for audit trails.

**Why this priority**: Enterprise and regulated industries require documentary evidence of research provenance.

**Independent Test**: Can be fully tested by exporting provenance data and verifying the export contains complete claim-to-evidence mappings.

**Acceptance Scenarios**:

1. **Given** a research response with citations, **When** the user selects "Export Provenance", **Then** a structured file is downloaded containing all claims, their citations, evidence spans, and verification verdicts.

2. **Given** an exported provenance file, **When** processed by external tools, **Then** the format is parseable and contains unique identifiers linking claims to evidence.

---

### User Story 6 - Draft Chat with Deferred Persistence (Priority: P1)

A user wants to start a new research conversation immediately without waiting for database operations. When they click "New Chat", they see an empty chat instantly and can start typing. The chat is only persisted to the database after the first message is successfully processed.

**Why this priority**: This eliminates the slow synchronous database call before streaming, prevents orphaned database records on failure, and provides instant feedback for a better user experience.

**Independent Test**: Can be fully tested by creating a new chat, verifying the URL contains `?draft=1`, and confirming that after sending a message and receiving a response, the draft parameter is removed from the URL.

**Acceptance Scenarios**:

1. **Given** user clicks "New Chat", **When** the action completes, **Then** a new chat entry appears in sidebar immediately without any loading state or API call, and the URL contains `?draft=1`.

2. **Given** user has a draft chat, **When** they send a message and research completes successfully, **Then** the chat and messages are persisted atomically, and the `?draft=1` parameter is removed from the URL.

3. **Given** user has a draft chat with content being streamed, **When** they try to close the browser, **Then** they see a warning about unsaved content.

4. **Given** persistence fails, **When** the error occurs, **Then** user sees a retry option and their content is preserved.

5. **Given** user has an old draft (>24 hours), **When** the page loads, **Then** the stale draft is automatically cleaned up from localStorage.

6. **Given** user is viewing a draft chat, **When** they hover over the chat in the sidebar, **Then** the context menu (rename, archive, delete) is not available.

---

### Edge Cases

- What happens when a source is no longer accessible (URL returns 404)?
  - Display cached metadata and quote; indicate "Source unavailable - showing cached data"
- How does the system handle claims that cannot be verified against available evidence?
  - Mark as "Unverified" with explanation; do not generate claims without at least one supporting source
- What happens when evidence spans are very long (>500 characters)?
  - Show truncated span with "Show more" option to expand
- How does the system handle conflicting evidence from multiple sources?
  - Display all sources; mark as "Conflicting evidence" with both supporting and contradicting quotes
- What happens when numeric claim detection produces false positives (e.g., page numbers, dates)?
  - Apply contextual filtering; treat page references and structural numbers differently from factual numeric claims

## Requirements *(mandatory)*

### Functional Requirements

#### Claim Extraction & Attribution

- **FR-001**: System MUST decompose generated text into atomic claims, where each claim represents a single factual assertion
- **FR-002**: System MUST assign at least one evidence source to every generated claim; claims without evidence MUST NOT be generated
- **FR-003**: System MUST extract and store the minimal evidence span (supporting quote) from source documents for each claim
- **FR-004**: System MUST preserve source metadata including: title, author/publisher, publication date, URL/identifier, and location within source (page number, section heading, or character offsets)

#### Evidence Cards

- **FR-005**: System MUST display inline citation markers for every claim in research responses
- **FR-006**: System MUST show an evidence card when users interact with (click/hover) a citation marker
- **FR-007**: Evidence cards MUST display: source title, author/publisher (when available), date, URL/file ID, exact supporting quote with highlighting, source location (page/section), and verification verdict
- **FR-008**: Evidence cards MUST close when users click outside or press Escape

#### Numeric Claim Handling

- **FR-009**: System MUST automatically detect numeric claims containing: value, unit, comparator, and entity reference
- **FR-010**: System MUST render detected numeric claims as interactive chips with distinct visual styling
- **FR-011**: Expanded numeric chips MUST display: exact source quote, normalized value (e.g., standardized units, expanded abbreviations), and any assumptions (currency year, exchange rate, rounding method)
- **FR-012**: For derived/computed numeric values, the system MUST display: "Derived" label, all input values with their citations, and calculation steps
- **FR-013**: System MUST differentiate between directly quoted numbers and computed/transformed numbers

#### Verification Pipeline

- **FR-014**: System MUST verify each claim against its cited evidence using attribution assessment
- **FR-015**: System MUST assign verification verdict to each claim: "Supported", "Partially supported", or "Not supported"
- **FR-016**: System MUST provide verification summary showing claim counts by support level
- **FR-017**: System MUST flag responses with high unsupported claim rates (>20%) with a warning indicator
- **FR-021**: System MUST perform verification in real-time during response synthesis (as claims are generated)
- **FR-022**: System MUST cache verification results to avoid re-verification on subsequent views of the same response
- **FR-025**: System MUST track and expose unsupported claim rate per response as an observable metric for monitoring citation quality

#### Provenance Data

- **FR-018**: System MUST maintain structured provenance records linking claims to evidence spans
- **FR-019**: System MUST support export of provenance data in a structured, machine-readable format
- **FR-020**: Exported provenance MUST include unique identifiers for claims, sources, and claim-source relationships
- **FR-023**: System MUST retain cached evidence spans (source quotes and metadata) for the lifetime of the associated chat session
- **FR-024**: When a chat session is deleted, associated cached evidence spans MUST be deleted

#### Configuration Inheritance

- **FR-026**: Per-depth `citation_verification` config MUST merge with global config, not replace it
- **FR-026.1**: Fields explicitly set in per-depth config MUST override global values
- **FR-026.2**: Fields not set in per-depth config MUST inherit from global config
- **FR-026.3**: Detection of "explicitly set" MUST compare against Pydantic defaults, not global values

#### Hybrid ReClaim Pattern (ReAct Synthesis)

- **FR-027**: ReAct synthesis MUST use XML tags to separate grounded content from LLM reasoning
- **FR-027.1**: `<cite key="Key">claim</cite>` tags MUST be used for all factual claims; the key MUST match the citation key from read_snippet
- **FR-027.2**: `<free>text</free>` tags MUST be used for structural content (headers, transitions, analytical comparisons)
- **FR-027.3**: `<unverified>claim</unverified>` tags MAY be used for uncertain claims that couldn't find direct evidence
- **FR-027.4**: Text outside XML tags MUST be treated as scratchpad (excluded from final report)
- **FR-027.5**: System MUST parse tagged content after ReAct loop completion and assemble only tagged blocks into final report

### Key Entities

- **Claim**: A single atomic factual assertion extracted from generated text; includes claim text, claim type (general/numeric), position in response, verification verdict
- **Evidence Span**: A minimal text passage from a source document that supports a claim; includes quote text, start/end positions in source, source identifier
- **Source**: A document retrieved during research; includes title, author, publisher, date, URL, content type (web/PDF/other), retrieval timestamp
- **Citation**: A link between a claim and one or more evidence spans; includes confidence score, verification verdict, verification reasoning
- **NumericClaim**: A specialized claim containing numeric data; includes raw value, normalized value, unit, entity reference, derivation type (direct/computed), computation details if derived

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of factual claims in research responses have at least one clickable citation linking to source evidence
- **SC-002**: Users can access evidence details for any claim within 1 click/interaction
- **SC-003**: Evidence cards display supporting quote within 500ms of user interaction
- **SC-004**: 95% of numeric claims are correctly detected and displayed as interactive chips
- **SC-005**: Users can verify a numeric claim (view source quote + normalization) in under 5 seconds
- **SC-006**: Verification summary accuracy: system-assessed verdicts match human judgment 90% of the time
- **SC-007**: Responses with >20% unsupported claims trigger visible warning in 100% of cases
- **SC-008**: Provenance export contains complete claim-to-evidence mappings for 100% of cited claims
- **SC-009**: User satisfaction: 80% of users report increased confidence in research accuracy after viewing evidence cards
- **SC-010**: Fact-checking time reduction: users can verify individual claims 60% faster compared to manual source lookup

## Assumptions

- Web search results provide sufficient source metadata (title, URL, date) for evidence card display
- Source content is cached during research to ensure evidence spans remain accessible even if source URL changes; cache is retained for the lifetime of the chat session
- Users are familiar with academic-style inline citations and expect similar interaction patterns
- The existing research agent pipeline can be extended to output structured claim-level data without fundamental architectural changes
- Verification assessment can be performed during synthesis without significantly impacting response latency (target: <2s additional latency)

## Dependencies

- Existing web search and content extraction capabilities (Brave Search, Trafilatura)
- Existing research agent pipeline (Coordinator, Planner, Researcher, Reflector, Synthesizer)
- Chat message storage and retrieval system
- Frontend rendering capabilities for interactive components

## Clarifications

### Session 2025-12-25

- Q: When should claim verification occur? → A: Real-time during synthesis (verify as response streams, cache results)
- Q: How long to retain cached evidence spans? → A: Lifetime of chat session (retain as long as the chat exists)
- Q: What key metric to track for citation quality observability? → A: Unsupported claim rate per response

## Testing Infrastructure

The project uses a 3-tier test architecture to ensure comprehensive coverage:

### Test Tiers

| Tier | Directory | Config | Credentials | Purpose |
|------|-----------|--------|-------------|---------|
| Unit | `tests/unit/` | N/A (mocked) | None | Fast tests with mocked dependencies |
| Integration | `tests/integration/` | `config/app.test.yaml` | Required | Real LLM/Brave tests with minimal settings |
| Complex | `tests/complex/` | `config/app.yaml` | Required | Long-running research with production config |

### Running Tests

```bash
make test                    # Unit tests only (fast, no credentials)
make test-integration        # Integration tests (real LLM/Brave, test config)
make test-complex            # Complex long-running tests (production config)
make test-all                # All Python + Frontend tests
```

### Test Configuration

Integration tests use `config/app.test.yaml` with minimal settings for speed:
- `max_plan_iterations: 1`
- `max_search_queries: 1`
- `enable_clarification: false`
- Fastest model tiers

Complex tests use production `config/app.yaml` for full-fidelity testing.

### Citation-Specific Tests

- `tests/unit/services/citation/test_pipeline.py` - Unit tests with mocked LLM
- `tests/integration/test_citation_pipeline.py` - Real LLM citation tests (claim generation, numeric detection, evidence selection, verification)
- `tests/complex/test_complex_research.py` - Multi-entity comparative research with comprehensive citation verification

## Known Issues & Fixes

### Issue #1: Citation Position Mismatch (Fixed 2025-12-30)

**Symptom**: Citations appeared grey/unresolved in the frontend even though claims were generated and stored in the database.

**Root Cause**: The claim extraction pipeline had a data transformation mismatch:
1. LLM generates content with numeric markers: `"Claim one [0]. Claim two [1]."`
2. Markers are replaced with human-readable keys: `"Claim one [Arxiv]. Claim two [Zhipu]."`
3. **BUG**: `_parse_interleaved_content()` parsed claims from the ORIGINAL content (with `[0]`)
4. But `final_report` stored in DB had human-readable keys (with `[Arxiv]`)
5. Position calculation was based on `[0]` (2 chars) not `[Arxiv]` (7 chars)
6. Result: `position_start`/`position_end` misaligned with actual text

**Fix**: Parse claims from `content_with_keys` (after replacement) instead of original `content`. Added `reverse_key_map` to lookup evidence_index from citation keys.

**Files Modified**: `src/services/citation/claim_generator.py` (lines 346-477)

**Pattern to Avoid**: When transforming data (e.g., replacing markers), always ensure downstream processing uses the transformed version if positions/offsets are calculated.

---

### Issue #2: SSE Event Property Case Mismatch (Fixed 2025-12-30)

**Symptom**: Activity labels showed "NaN" durations, raw event types, and research steps showed 0/N completed.

**Root Cause**: TypeScript interfaces defined snake_case properties (`duration_ms`, `step_index`) but runtime SSE events had camelCase (`durationMs`, `stepIndex`).

**Fix**: Updated all event handlers to check camelCase properties first with fallback to snake_case:
```typescript
const durationMs = (event as any).durationMs ?? event.duration_ms;
```

**Files Modified**:
- `frontend/src/hooks/useStreamingQuery.ts`
- `frontend/src/utils/activityLabels.ts`

**Pattern to Avoid**: When serializing Python snake_case to JSON, be aware that JavaScript conventions differ. Either normalize on one side or handle both cases.

---

### Issue #3: Phantom Chats After DB Clean (Fixed 2025-12-30)

**Symptom**: After cleaning the database and restarting, old chats still appeared in the sidebar.

**Root Cause**: Two sources of stale data:
1. TanStack Query cache had 5-minute `staleTime` (too long)
2. localStorage draft chats survive for 24 hours, unaffected by DB operations

**Fix**:
- Reduced `staleTime` to 30 seconds
- Enabled `refetchOnWindowFocus` for fresh data on tab switch
- Added `clearStaleDrafts()` to remove drafts >60 seconds old that don't exist in API

**Files Modified**:
- `frontend/src/main.tsx`
- `frontend/src/hooks/useDraftChats.ts`
- `frontend/src/pages/ChatPage.tsx`

**Pattern to Avoid**: When using local storage for optimistic UI, always sync with backend state on load to prevent stale data.

## Out of Scope

- Training custom models for citation learning (may be considered in future iterations)
- Vector store integration for improved retrieval (separate feature)
- Real-time source monitoring for URL availability
- Multi-language numeric claim normalization (English only for initial release)
- PDF page-level location extraction (URL-only sources for initial release; page numbers when available in metadata)
