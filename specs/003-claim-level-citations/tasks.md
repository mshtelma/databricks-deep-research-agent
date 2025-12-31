# Tasks: Claim-Level Citation Granularity

**Feature**: 003-claim-level-citations
**Input**: Design documents from `/specs/003-claim-level-citations/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.yaml

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Includes exact file paths in descriptions

## Path Conventions

- **Backend**: `src/` at repository root (Python)
- **Frontend**: `frontend/src/` (TypeScript/React)
- **Tests**: `tests/` (backend), `frontend/tests/` (frontend), `e2e/` (Playwright)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and configuration for claim-level citations feature

- [x] T001 Add citation_verification configuration section to config/app.yaml with all 6 stage settings
- [x] T002 [P] Create CitationVerificationConfig Pydantic model in src/core/app_config.py
- [x] T003 [P] Create src/services/citation/__init__.py with module exports
- [x] T004 [P] Create citation verification prompts directory src/agent/prompts/citation/
- [x] T005 [P] Add citation type definitions in frontend/src/types/citation.ts

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Database schema and core models that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

### Database Migrations

- [x] T006 Create Alembic migration for claims table in src/db/migrations/
- [x] T007 Create Alembic migration for evidence_spans table in src/db/migrations/
- [x] T008 Create Alembic migration for citations table in src/db/migrations/
- [x] T009 Create Alembic migration for numeric_claim_details table in src/db/migrations/
- [x] T010 Create Alembic migration for citation_corrections table in src/db/migrations/
- [x] T011 Create Alembic migration for verification_summaries table in src/db/migrations/
- [x] T012 Add source location columns (total_pages, detected_sections, content_type) to sources table migration

### SQLAlchemy Models

- [x] T013 [P] Define ClaimType, VerificationVerdict, ConfidenceLevel, CorrectionType, DerivationType enums in src/models/enums.py
- [x] T014 [P] Create Claim SQLAlchemy model in src/models/claim.py
- [x] T015 [P] Create EvidenceSpan SQLAlchemy model in src/models/evidence_span.py
- [x] T016 [P] Create Citation SQLAlchemy model in src/models/citation.py
- [x] T017 [P] Create NumericClaimDetails SQLAlchemy model in src/models/numeric_claim.py
- [x] T018 [P] Create CitationCorrection SQLAlchemy model in src/models/citation_correction.py
- [x] T019 [P] Create VerificationSummary SQLAlchemy model in src/models/verification_summary.py
- [x] T020 Update Source model with location columns in src/models/source.py
- [x] T021 Add model imports to src/models/__init__.py

### Pydantic Schemas (API Layer)

- [x] T022 [P] Create ClaimResponse, MessageClaimsResponse schemas in src/schemas/citation.py
- [x] T023 [P] Create EvidenceSpanResponse, ClaimEvidenceResponse schemas in src/schemas/citation.py
- [x] T024 [P] Create VerificationSummary, CorrectionMetrics schemas in src/schemas/citation.py
- [x] T025 [P] Create NumericClaimDetail, QAVerificationResult schemas in src/schemas/citation.py
- [x] T026 [P] Create CitationCorrectionResponse, ProvenanceExport schemas in src/schemas/citation.py

### Core Services

- [x] T027 Create ClaimService with CRUD operations in src/services/claim_service.py
- [x] T028 Create EvidenceSpanService with CRUD operations in src/services/evidence_span_service.py
- [x] T029 Create CitationService with CRUD operations in src/services/citation_service.py
- [x] T030 Create VerificationSummaryService in src/services/verification_summary_service.py

### State Extensions

- [x] T031 Add ClaimInfo, EvidenceInfo dataclasses to src/agent/state.py
- [x] T032 Add evidence_pool, claims, verification_summary fields to ResearchState in src/agent/state.py

**Checkpoint**: Foundation complete - all data models, migrations, and core services ready

---

## Phase 3: User Story 1 - View Evidence Card for Any Claim (Priority: P1)

**Goal**: Users can click any citation marker to see evidence card with source metadata, supporting quote, and verification verdict

**Independent Test**: Click any citation in a research response → Evidence card displays source title, URL, quote, verification badge

### Stage 1: Evidence Pre-Selection (Backend)

- [x] T033 [US1] Create EVIDENCE_PRESELECTION_PROMPT in src/agent/prompts/citation/evidence_selection.py
- [x] T034 [US1] Implement EvidencePreSelector.segment_into_spans() method in src/services/citation/evidence_selector.py
- [x] T035 [US1] Implement EvidencePreSelector.compute_relevance() with hybrid keyword+semantic scoring in src/services/citation/evidence_selector.py
- [x] T036 [US1] Implement EvidencePreSelector.select_evidence_spans() main method in src/services/citation/evidence_selector.py

### Stage 2: Interleaved Generation (Backend)

- [x] T037 [US1] Create INTERLEAVED_GENERATION_PROMPT in src/agent/prompts/citation/interleaved_generation.py
- [x] T038 [US1] Implement InterleavedGenerator.select_best_evidence() method in src/services/citation/claim_generator.py
- [x] T039 [US1] Implement InterleavedGenerator.generate_constrained_claim() method in src/services/citation/claim_generator.py
- [x] T040 [US1] Implement InterleavedGenerator.synthesize_with_interleaving() async generator in src/services/citation/claim_generator.py

### Stage 4: Isolated Verification (Backend)

- [x] T041 [US1] Create ISOLATED_VERIFICATION_PROMPT with 4-tier verdict in src/agent/prompts/citation/verification.py
- [x] T042 [US1] Implement IsolatedVerifier.verify_with_isolation() method in src/services/citation/isolated_verifier.py
- [x] T043 [US1] Implement IsolatedVerifier.parse_verdict() for SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED in src/services/citation/isolated_verifier.py

### Synthesizer Integration (Backend)

- [ ] T044 [US1] Modify synthesizer.py to call EvidencePreSelector before generation in src/agent/nodes/synthesizer.py
- [ ] T045 [US1] Modify synthesizer.py to use InterleavedGenerator for claim generation in src/agent/nodes/synthesizer.py
- [ ] T046 [US1] Modify synthesizer.py to call IsolatedVerifier for each claim in src/agent/nodes/synthesizer.py
- [ ] T047 [US1] Add claim persistence after verification in src/agent/nodes/synthesizer.py

### API Endpoints (Backend)

- [x] T048 [US1] Create GET /messages/{id}/claims endpoint in src/api/v1/citations.py
- [x] T049 [US1] Create GET /claims/{id} endpoint in src/api/v1/citations.py
- [x] T050 [US1] Create GET /claims/{id}/evidence endpoint in src/api/v1/citations.py
- [x] T051 [US1] Register citation routes in src/api/v1/__init__.py

### Streaming Events (Backend)

- [x] T052 [US1] Add ClaimVerifiedEvent schema in src/schemas/streaming.py
- [x] T053 [US1] Emit claim_verified SSE event during synthesis in src/api/v1/research.py

### Frontend Components (Frontend)

- [x] T054 [P] [US1] Create CitationMarker component with superscript styling in frontend/src/components/citations/CitationMarker.tsx
- [x] T055 [P] [US1] Create VerificationBadge component (green/amber/red/purple) in frontend/src/components/citations/VerificationBadge.tsx
- [x] T056 [P] [US1] Create SourceMetadata subcomponent in frontend/src/components/citations/SourceMetadata.tsx
- [x] T057 [P] [US1] Create EvidenceQuote subcomponent with highlighting in frontend/src/components/citations/EvidenceQuote.tsx
- [x] T058 [US1] Create EvidenceCard popover component in frontend/src/components/citations/EvidenceCard.tsx
- [x] T059 [US1] Create useCitations hook for claim/citation state in frontend/src/hooks/useCitations.ts
- [x] T060 [US1] Create citationPlugin for react-markdown to parse citation markers in frontend/src/lib/markdown/citationPlugin.ts
- [x] T061 [US1] Integrate CitationMarker into MarkdownRenderer in frontend/src/components/common/MarkdownRenderer.tsx
- [x] T062 [US1] Modify AgentMessage to render claims with EvidenceCards in frontend/src/components/chat/AgentMessage.tsx

### Streaming Event Handling (Frontend)

- [x] T063 [US1] Add claim_verified event handler in useStreamingQuery hook in frontend/src/hooks/useStreamingQuery.ts

**Checkpoint**: User Story 1 complete - Evidence cards work for all claims

---

## Phase 4: User Story 2 - Numeric Claims with Full Provenance (Priority: P1)

**Goal**: Numeric claims display as interactive chips with normalization, derivation tracking, and QA verification

**Independent Test**: Find a numeric claim → Expand chip → See raw value, normalized value, unit, and derivation info

### Stage 3: Confidence Classification (Backend)

- [x] T064 [US2] Create CONFIDENCE_CLASSIFICATION_PROMPT in src/agent/prompts/citation/confidence.py
- [x] T065 [US2] Implement ConfidenceClassifier.detect_quote_match() in src/services/citation/confidence_classifier.py
- [x] T066 [US2] Implement ConfidenceClassifier.detect_hedging_words() in src/services/citation/confidence_classifier.py
- [x] T067 [US2] Implement ConfidenceClassifier.classify_confidence() returning HIGH/MEDIUM/LOW in src/services/citation/confidence_classifier.py
- [ ] T068 [US2] Integrate confidence classification into synthesizer pipeline in src/agent/nodes/synthesizer.py

### Stage 6: Numeric QA Verification (Backend)

- [x] T069 [US2] Create NUMERIC_DETECTION_PROMPT in src/agent/prompts/citation/numeric_qa.py
- [x] T070 [US2] Create QA_GENERATION_PROMPT for numeric questions in src/agent/prompts/citation/numeric_qa.py
- [x] T071 [US2] Implement NumericVerifier.detect_numeric_claims() in src/services/citation/numeric_verifier.py
- [x] T072 [US2] Implement NumericVerifier.generate_qa_pairs() in src/services/citation/numeric_verifier.py
- [x] T073 [US2] Implement NumericVerifier.answer_from_evidence() in src/services/citation/numeric_verifier.py
- [x] T074 [US2] Implement NumericVerifier.normalize_and_compare() for numeric matching in src/services/citation/numeric_verifier.py
- [x] T075 [US2] Implement NumericVerifier.verify_numeric_claim() main method in src/services/citation/numeric_verifier.py

### Numeric Claim Persistence (Backend)

- [ ] T076 [US2] Create NumericClaimService for numeric claim CRUD in src/services/numeric_claim_service.py
- [ ] T077 [US2] Integrate numeric detection and QA verification into synthesizer in src/agent/nodes/synthesizer.py

### Streaming Events (Backend)

- [ ] T078 [US2] Add NumericClaimDetectedEvent schema in src/schemas/streaming.py
- [ ] T079 [US2] Emit numeric_claim_detected SSE event in src/api/v1/research.py

### Frontend Components (Frontend)

- [x] T080 [P] [US2] Create NumericChip component with distinct styling in frontend/src/components/citations/NumericChip.tsx
- [ ] T081 [P] [US2] Create NormalizedValue subcomponent in frontend/src/components/citations/NormalizedValue.tsx
- [ ] T082 [P] [US2] Create DerivationInfo subcomponent in frontend/src/components/citations/DerivationInfo.tsx
- [ ] T083 [P] [US2] Create AssumptionsNote subcomponent in frontend/src/components/citations/AssumptionsNote.tsx
- [x] T084 [US2] Create NumericDetails expandable popover in frontend/src/components/citations/NumericDetails.tsx
- [x] T085 [US2] Integrate NumericChip rendering into AgentMessage in frontend/src/components/chat/AgentMessage.tsx

### Streaming Event Handling (Frontend)

- [ ] T086 [US2] Add numeric_claim_detected event handler in useStreamingQuery in frontend/src/hooks/useStreamingQuery.ts

**Checkpoint**: User Story 2 complete - Numeric claims show full provenance with QA verification

---

## Phase 5: User Story 3 - Atomic Claim Decomposition (Priority: P2)

**Goal**: Multi-fact sentences are automatically split into atomic claims, each with its own citation

**Independent Test**: Submit query producing multi-fact sentences → Each atomic claim has distinct clickable citation

### Claim Extraction (Backend)

- [ ] T087 [US3] Create ATOMIC_CLAIM_EXTRACTION_PROMPT in src/agent/prompts/citation/claim_extraction.py
- [ ] T088 [US3] Implement ClaimExtractor.decompose_sentence() in src/services/citation/claim_extractor.py
- [ ] T089 [US3] Implement ClaimExtractor.extract_atomic_claims() main method in src/services/citation/claim_extractor.py
- [ ] T090 [US3] Integrate atomic claim extraction into InterleavedGenerator in src/services/citation/claim_generator.py

### Multi-Citation Support (Backend)

- [ ] T091 [US3] Implement ClaimExtractor.find_all_supporting_spans() for multi-span claims in src/services/citation/claim_extractor.py
- [ ] T092 [US3] Update Citation model to support multiple evidence spans per claim in src/models/citation.py

### Frontend Updates (Frontend)

- [ ] T093 [US3] Update CitationMarker to handle multiple citations per sentence in frontend/src/components/citations/CitationMarker.tsx
- [ ] T094 [US3] Update EvidenceCard to show multiple supporting passages in frontend/src/components/citations/EvidenceCard.tsx

**Checkpoint**: User Story 3 complete - Multi-fact sentences decomposed with individual citations

---

## Phase 6: User Story 4 - Verification Status Overview (Priority: P2)

**Goal**: Users see summary of claim verification status (Supported/Partial/Unsupported/Contradicted counts) with warning indicators

**Independent Test**: Generate research response → View verification summary panel → Counts match actual claim verdicts

### Stage 5: Citation Correction (Backend)

- [x] T095 [US4] Create CITATION_CORRECTION_PROMPT in src/agent/prompts/citation/correction.py
- [x] T096 [US4] Implement CitationCorrector.citation_entails() check in src/services/citation/citation_corrector.py
- [x] T097 [US4] Implement CitationCorrector.find_better_citation() from pool in src/services/citation/citation_corrector.py
- [x] T098 [US4] Implement CitationCorrector.correct_citations() main method in src/services/citation/citation_corrector.py
- [ ] T099 [US4] Create CitationCorrectionService for correction CRUD in src/services/citation_correction_service.py
- [ ] T100 [US4] Integrate citation correction into synthesizer pipeline in src/agent/nodes/synthesizer.py

### Verification Summary (Backend)

- [x] T101 [US4] Implement VerificationSummaryService.compute_summary() in src/services/verification_summary_service.py
- [x] T102 [US4] Implement VerificationSummaryService.check_warning_threshold() in src/services/verification_summary_service.py
- [ ] T103 [US4] Create verification summary after all claims verified in src/agent/nodes/synthesizer.py

### API Endpoints (Backend)

- [x] T104 [US4] Create GET /messages/{id}/verification-summary endpoint in src/api/v1/citations.py
- [ ] T105 [US4] Create GET /messages/{id}/correction-metrics endpoint in src/api/v1/citations.py

### Streaming Events (Backend)

- [ ] T106 [US4] Add VerificationSummaryEvent schema in src/schemas/streaming.py
- [ ] T107 [US4] Add CitationCorrectedEvent schema in src/schemas/streaming.py
- [ ] T108 [US4] Emit verification_summary SSE event after all claims verified in src/api/v1/research.py
- [ ] T109 [US4] Emit citation_corrected SSE event during correction in src/api/v1/research.py

### Frontend Components (Frontend)

- [ ] T110 [P] [US4] Create SupportBreakdown component (pie chart or bars) in frontend/src/components/citations/SupportBreakdown.tsx
- [x] T111 [P] [US4] Create WarningBanner component for high unsupported rate in frontend/src/components/citations/WarningBanner.tsx
- [x] T112 [US4] Create VerificationSummary panel component in frontend/src/components/citations/VerificationSummary.tsx
- [x] T113 [US4] Add claim highlighting by verdict category in frontend/src/components/citations/VerificationSummary.tsx
- [x] T114 [US4] Integrate VerificationSummary into AgentMessage in frontend/src/components/chat/AgentMessage.tsx

### Streaming Event Handling (Frontend)

- [ ] T115 [US4] Add verification_summary event handler in useStreamingQuery in frontend/src/hooks/useStreamingQuery.ts
- [ ] T116 [US4] Add citation_corrected event handler in useStreamingQuery in frontend/src/hooks/useStreamingQuery.ts

**Checkpoint**: User Story 4 complete - Verification summary shows claim breakdown with warnings

---

## Phase 7: User Story 5 - Structured Provenance Export (Priority: P3)

**Goal**: Users can export complete claim-to-evidence mappings in JSON/CSV format for audit trails

**Independent Test**: Generate response → Click Export Provenance → Download contains all claims, citations, verdicts

### Export Service (Backend)

- [ ] T117 [US5] Create ProvenanceExporter.export_message_json() in src/services/provenance_exporter.py
- [ ] T118 [US5] Create ProvenanceExporter.export_message_csv() in src/services/provenance_exporter.py
- [ ] T119 [US5] Create ProvenanceExporter.export_chat_json() in src/services/provenance_exporter.py
- [ ] T120 [US5] Create ProvenanceExporter.export_chat_csv() in src/services/provenance_exporter.py

### API Endpoints (Backend)

- [x] T121 [US5] Create GET /messages/{id}/provenance endpoint in src/api/v1/citations.py
- [ ] T122 [US5] Create GET /chats/{id}/provenance endpoint in src/api/v1/citations.py
- [ ] T123 [US5] Add format query param (json/csv) to provenance endpoints in src/api/v1/citations.py

### Frontend Components (Frontend)

- [ ] T124 [P] [US5] Create ExportButton component in frontend/src/components/citations/ExportButton.tsx
- [ ] T125 [US5] Create ExportModal with format selection in frontend/src/components/citations/ExportModal.tsx
- [ ] T126 [US5] Add export button to AgentMessage or chat header in frontend/src/components/chat/AgentMessage.tsx

**Checkpoint**: User Story 5 complete - Provenance export works for messages and chats

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Integration testing, performance optimization, edge cases

### Performance Optimization

- [ ] T127 [P] Add parallel verification for high-confidence claims in src/services/citation/isolated_verifier.py
- [ ] T128 [P] Add caching for repeated evidence spans in src/services/citation/evidence_selector.py
- [ ] T129 [P] Add batch confidence classification in src/services/citation/confidence_classifier.py

### Edge Cases

- [ ] T130 Implement source unavailable fallback (cached metadata display) in frontend/src/components/citations/EvidenceCard.tsx
- [ ] T131 Implement long evidence span truncation with "Show more" in frontend/src/components/citations/EvidenceQuote.tsx
- [ ] T132 Implement conflicting evidence display in frontend/src/components/citations/EvidenceCard.tsx
- [ ] T133 Add numeric claim false positive filtering in src/services/citation/numeric_verifier.py

### Authorization

- [x] T134 Add authorization checks to all citation endpoints in src/api/v1/citations.py
- [x] T135 Ensure claims only accessible to chat session owner in src/services/claim_service.py

### Configuration Validation

- [ ] T136 Add startup validation for citation_verification config in src/core/app_config.py
- [ ] T137 Add feature toggle for citation verification in config/app.yaml

### Documentation

- [ ] T138 Update quickstart.md with actual command examples after implementation
- [ ] T139 Add citation API to existing OpenAPI documentation

### Test Infrastructure (3-Tier Architecture)

- [x] T140 [P] Create config/app.test.yaml with minimal test settings (fast models, 1 iteration)
- [x] T141 [P] Add APP_CONFIG_PATH env var support in src/core/app_config.py
- [x] T142 [P] Move mocked citation pipeline test to tests/unit/services/citation/test_pipeline.py
- [x] T143 [P] Create tests/integration/conftest.py with shared fixtures and test config
- [x] T144 [P] Create tests/integration/test_citation_pipeline.py with real LLM tests
- [x] T145 [P] Create tests/complex/ directory with conftest.py (production config)
- [x] T146 [P] Create tests/complex/test_complex_research.py with long-running tests
- [x] T147 [P] Update pyproject.toml with unit/integration/complex markers
- [x] T148 [P] Update Makefile with test-unit, test-integration, test-complex targets
- [x] T149 [P] Update spec.md and plan.md with test infrastructure documentation

**Checkpoint**: Test infrastructure complete - 3-tier test hierarchy operational

---

## Phase 9: Draft Chat with Deferred Persistence (US6)

**Purpose**: Implement "draft chat" functionality where new chats exist only in local state until the first message is successfully processed.

### Backend Authorization

- [x] T150 [US6] Add get_by_id() method to ChatService in src/services/chat_service.py
- [x] T151 [US6] Create _verify_chat_access() with owned-or-missing logic in src/api/v1/research.py
- [x] T152 [US6] Update stream_research_endpoint() authorization in src/api/v1/research.py
- [x] T153 [US6] Update stream_research_with_history() authorization in src/api/v1/research.py

### Backend Persistence

- [x] T154 [US6] Add user_id parameter to persist_complete_research() in src/agent/persistence.py
- [x] T155 [US6] Add chat creation with INSERT ON CONFLICT in src/agent/persistence.py
- [x] T156 [US6] Add chat.updated_at update after message persist in src/agent/persistence.py
- [x] T157 [US6] Add PersistenceCompletedEvent schema in src/schemas/streaming.py
- [x] T158 [US6] Emit persistence_completed event in orchestrator in src/agent/orchestrator.py
- [x] T159 [US6] Add is_draft flag to OrchestrationConfig in src/agent/orchestrator.py

### Frontend Draft Management

- [x] T160 [P] [US6] Create useDraftChats hook in frontend/src/hooks/useDraftChats.ts
- [x] T161 [P] [US6] Add PersistenceCompletedEvent type in frontend/src/types/index.ts
- [x] T162 [US6] Add persistence_completed handler in useStreamingQuery in frontend/src/hooks/useStreamingQuery.ts

### Frontend Integration

- [x] T163 [US6] Add draft state management in frontend/src/pages/ChatPage.tsx
- [x] T164 [US6] Fix missing ChatSidebar props in frontend/src/pages/ChatPage.tsx
- [x] T165 [US6] Add beforeunload warning for drafts in frontend/src/pages/ChatPage.tsx
- [x] T166 [US6] Add persistence failure retry UI in frontend/src/pages/ChatPage.tsx
- [x] T167 [US6] Update handleNewChat to create drafts in frontend/src/pages/ChatPage.tsx

### E2E Tests

- [x] T168 [US6] Add draft chat URL test in e2e/tests/chat-management.spec.ts
- [x] T169 [US6] Add draft persistence test in e2e/tests/chat-management.spec.ts
- [x] T170 [US6] Add draft menu disabled test in e2e/tests/chat-management.spec.ts

**Checkpoint**: Draft chat feature complete - instant new chat creation, deferred persistence

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (BLOCKS ALL USER STORIES)
    ↓
    ├─→ Phase 3: US1 (Evidence Cards) ─┐
    │                                   │
    ├─→ Phase 4: US2 (Numeric Claims) ──┼──→ Phase 8: Polish
    │   (depends on Stage 3,6)          │
    │                                   │
    ├─→ Phase 5: US3 (Claim Decomp) ────┤
    │                                   │
    ├─→ Phase 6: US4 (Verification) ────┤
    │   (depends on Stage 5)            │
    │                                   │
    └─→ Phase 7: US5 (Export) ──────────┘
```

### User Story Dependencies

| Story | Depends On | Notes |
|-------|------------|-------|
| US1 (P1) | Phase 2 | Core evidence cards - MVP |
| US2 (P1) | Phase 2, US1 Stage 1-2 | Shares evidence pre-selection |
| US3 (P2) | Phase 2, US1 | Extends claim extraction |
| US4 (P2) | Phase 2, US1 | Adds verification summary |
| US5 (P3) | Phase 2 | Can be done in parallel with others |

### 6-Stage Pipeline Dependencies

```
Stage 1: Evidence Pre-Selection (US1)
    ↓
Stage 2: Interleaved Generation (US1)
    ↓
Stage 3: Confidence Classification (US2) ─────┐
    ↓                                         │
Stage 4: Isolated Verification (US1) ←────────┘
    ↓                    (routing by confidence)
Stage 5: Citation Correction (US4)
    ↓
Stage 6: Numeric QA Verification (US2)
```

### Parallel Opportunities

**Within Phase 2 (Foundational)**:
```
T013-T019: All model files can be created in parallel
T022-T026: All schema files can be created in parallel
```

**Within Phase 3 (US1)**:
```
T054-T057: Frontend subcomponents can be created in parallel
```

**Within Phase 4 (US2)**:
```
T080-T083: Frontend subcomponents can be created in parallel
```

**Across User Stories** (after Phase 2):
- US1 and US5 can proceed in parallel (US5 is export-only)
- US3 and US4 can start once US1 is partially complete (Stage 1-2)

---

## Parallel Example: Phase 2 Models

```bash
# Launch all SQLAlchemy models together:
Task: "Create Claim SQLAlchemy model in src/models/claim.py"
Task: "Create EvidenceSpan SQLAlchemy model in src/models/evidence_span.py"
Task: "Create Citation SQLAlchemy model in src/models/citation.py"
Task: "Create NumericClaimDetails SQLAlchemy model in src/models/numeric_claim.py"
Task: "Create CitationCorrection SQLAlchemy model in src/models/citation_correction.py"
Task: "Create VerificationSummary SQLAlchemy model in src/models/verification_summary.py"
```

## Parallel Example: US1 Frontend Components

```bash
# Launch all frontend subcomponents together:
Task: "Create CitationMarker component in frontend/src/components/citations/CitationMarker.tsx"
Task: "Create VerificationBadge component in frontend/src/components/citations/VerificationBadge.tsx"
Task: "Create SourceMetadata subcomponent in frontend/src/components/citations/SourceMetadata.tsx"
Task: "Create EvidenceQuote subcomponent in frontend/src/components/citations/EvidenceQuote.tsx"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T032)
3. Complete Phase 3: User Story 1 (T033-T063)
4. **STOP and VALIDATE**: Evidence cards work for all claims
5. Deploy/demo if ready

### Incremental Delivery

1. **MVP**: Setup + Foundational + US1 → Evidence cards work
2. **+Numeric**: Add US2 → Numeric chips with QA verification
3. **+Decomposition**: Add US3 → Multi-fact sentences split
4. **+Summary**: Add US4 → Verification overview + warnings
5. **+Export**: Add US5 → Provenance export for compliance
6. **Polish**: Phase 8 → Performance, edge cases, hardening

### Parallel Team Strategy

With 3 developers after Foundational:
- **Dev A**: US1 (Evidence Cards) - Core pipeline
- **Dev B**: US2 (Numeric Claims) - After Stage 1-2 from Dev A
- **Dev C**: US5 (Export) + US4 (Summary) - Independent work

---

## Summary

| Phase | Tasks | Parallel Tasks | Critical Path |
|-------|-------|----------------|---------------|
| Setup | 5 | 4 | T001 → T002 |
| Foundational | 27 | 15 | T006-T012 migrations sequential |
| US1 Evidence Cards | 31 | 8 | Stages 1→2→4→API→Frontend |
| US2 Numeric Claims | 23 | 5 | Stages 3→6→Frontend |
| US3 Claim Decomposition | 8 | 0 | Depends on US1 |
| US4 Verification Summary | 22 | 3 | Stage 5→API→Frontend |
| US5 Provenance Export | 10 | 1 | Independent |
| Polish | 13 | 4 | Final phase |
| US6 Draft Chat | 21 | 3 | Backend→Frontend→E2E |
| **Total** | **160** | **43** | |

**Task Count by User Story**:
- US1: 31 tasks (core MVP)
- US2: 23 tasks (numeric claims)
- US3: 8 tasks (decomposition)
- US4: 22 tasks (verification summary)
- US5: 10 tasks (export)
- US6: 21 tasks (draft chat with deferred persistence)
- Setup/Foundation/Polish: 45 tasks

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1) = 63 tasks

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- 6-stage pipeline is the core architecture - follow the stage order
- Tests are NOT included as spec.md did not explicitly request TDD
