# Feature Specification: Data Source Selection Integration

**Feature Branch**: `008-data-source-selection`
**Created**: 2026-02-05
**Status**: Draft
**Input**: Integration gap fix - enable users to select data sources (Vector Search indexes, Genie spaces, Knowledge Assistants) when submitting research queries

## Overview

The Deep Research Agent has enterprise data source discovery and management capabilities, but users cannot currently select which data sources to use when submitting a research query. The data source selection UI components exist but are not connected to the research submission flow. This feature completes the integration so users can choose specific enterprise data sources for their research.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select Data Source Scope Before Research (Priority: P1)

A researcher wants to control whether their research query searches enterprise data sources, web sources, or both. Before submitting a query, they select a scope (Enterprise Only, Web Only, or All) to focus the research on the appropriate sources.

**Why this priority**: This is the core value proposition - without scope selection, users cannot direct research to their enterprise data at all. This must work before any other feature.

**Independent Test**: Can be fully tested by selecting "Enterprise Only" scope, submitting a research query, and verifying that only enterprise sources are consulted (no web searches occur).

**Acceptance Scenarios**:

1. **Given** the user is on the chat page ready to submit a query, **When** they look at the input controls, **Then** they see a scope selector with options "Enterprise Only", "Web Only", and "All Sources"
2. **Given** the user selects "Enterprise Only" scope, **When** they submit a research query, **Then** the system only searches enterprise data sources and does not perform web searches
3. **Given** the user selects "Web Only" scope, **When** they submit a research query, **Then** the system only performs web searches and does not query enterprise sources
4. **Given** the user selects "All Sources" scope, **When** they submit a research query, **Then** the system searches both enterprise and web sources

---

### User Story 2 - Enable/Disable Specific Data Sources (Priority: P2)

A researcher wants fine-grained control over which specific data sources are used. After selecting a scope, they can expand the selector to see all available sources and toggle individual sources on or off for their query.

**Why this priority**: Builds on P1 by adding granular control. Users can use the system with just scope selection, but specific source control improves precision.

**Independent Test**: Can be fully tested by expanding the source selector, disabling a specific Vector Search index, submitting a query, and verifying that disabled source is not queried.

**Acceptance Scenarios**:

1. **Given** the user has selected a scope, **When** they expand the source selector, **Then** they see a list of all available sources grouped by type (Vector Search, Genie, Knowledge Assistant)
2. **Given** the expanded source list shows available sources, **When** the user toggles off a specific source, **Then** that source is excluded from the research query
3. **Given** the user has disabled some sources, **When** they submit a query, **Then** only the enabled sources are consulted
4. **Given** the user toggles a source back on, **When** they submit a query, **Then** that source is included in the research

---

### User Story 3 - Remember Source Preferences (Priority: P3)

A researcher frequently uses the same source configuration. The system remembers their last-used scope and source selections so they don't have to reconfigure every time.

**Why this priority**: Quality of life improvement. The core functionality works without persistence, but returning users benefit from remembered preferences.

**Independent Test**: Can be fully tested by configuring source selections, refreshing the page, and verifying selections are preserved.

**Acceptance Scenarios**:

1. **Given** the user has selected a scope and specific sources, **When** they refresh the page or return later, **Then** their previous selections are restored
2. **Given** the user changes their scope selection, **When** they submit a query and later return, **Then** the new scope preference is remembered
3. **Given** the user is in a new browser session, **When** they open the chat, **Then** reasonable defaults are shown (All Sources scope with all sources enabled)

---

### Edge Cases

- What happens when no enterprise data sources are available? The system shows a message indicating no enterprise sources were discovered and defaults to web-only search.
- What happens when a previously-selected source becomes unavailable? The system removes it from the selection and notifies the user that the source is no longer accessible.
- What happens when the user selects "Enterprise Only" but has no enterprise sources enabled? The system shows an error message requiring at least one source to be enabled or scope changed.
- How does the system handle source selection for "simple" query mode? Source selection is hidden for simple mode since it doesn't perform external searches.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST display a source scope selector in the chat input area for "web_search" and "deep_research" query modes
- **FR-002**: System MUST provide three scope options: "Enterprise Only", "Web Only", and "All Sources"
- **FR-003**: System MUST hide the source scope selector when "simple" query mode is selected
- **FR-004**: System MUST allow users to expand the scope selector to view and toggle individual sources
- **FR-005**: System MUST group available sources by type (Vector Search, Genie, Knowledge Assistant) in the expanded view
- **FR-006**: System MUST show source status indicators (available, unavailable, syncing) for each source
- **FR-007**: System MUST apply the selected scope and enabled sources when processing research queries
- **FR-008**: System MUST prevent submission when "Enterprise Only" is selected but no sources are enabled
- **FR-009**: System MUST persist user's scope and source selections across browser sessions
- **FR-010**: System MUST show appropriate feedback when selected sources become unavailable
- **FR-011**: System MUST pass source selection parameters through the entire request flow from UI to research execution
- **FR-012**: System MUST default to "All Sources" scope with all discovered sources enabled for new users

### Key Entities

- **Source Scope**: The category of sources to search - can be "enterprise_only", "web_only", or "all"
- **Enabled Sources**: List of specific source identifiers that should be included in research
- **Disabled Sources**: List of specific source identifiers that should be excluded from research
- **Discovered Source**: A data source found through auto-discovery, with attributes like name, type, status, and capabilities

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can select and submit research with a specific source scope in under 10 seconds (excluding query typing time)
- **SC-002**: 100% of research queries respect the user's source scope selection (Enterprise Only queries never trigger web searches, and vice versa)
- **SC-003**: Source selection preferences persist correctly across 100% of browser refresh/return scenarios
- **SC-004**: Users can toggle individual sources on/off with a single click per source
- **SC-005**: Source selector displays discovered sources within 2 seconds of the chat page loading
- **SC-006**: 90% of users successfully complete their first source-scoped query without errors or confusion

## Assumptions

- The enterprise data source discovery system (from feature 007) is functioning and returns available sources
- The SourceScopeSelector UI component already exists and provides the scope selection interface
- The backend research request schema already supports source_scope, enabled_sources, and disabled_sources fields
- Users have appropriate permissions to access discovered enterprise data sources via OBO authentication
- The discovery hooks correctly fetch and cache available sources

## Dependencies

- **Feature 007 (Enterprise Data Sources)**: Provides discovery, schemas, and components that this feature integrates
- **OBO Authentication**: Required for enterprise source access validation
- **Discovery Cache**: Provides the list of available sources to display in the selector

## Out of Scope

- Creating new data source discovery functionality (already exists)
- Building new UI components for source selection (SourceScopeSelector already exists)
- Modifying how individual source types (Vector Search, Genie, KA) perform queries (already implemented)
- Per-chat or per-query source preferences (only global user preferences in this release)
