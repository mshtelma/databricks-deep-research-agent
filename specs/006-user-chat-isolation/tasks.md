# Tasks: User Chat Isolation

**Input**: Design documents from `/specs/006-user-chat-isolation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Updated**: 2026-01-25 - Added User Stories 5-9 (Profile UI and Incognito Chats)

**Tests**: Tests are included in this task list to ensure proper verification of security-critical changes.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Task Summary

| Priority | User Story | Tasks | Status |
|----------|------------|-------|--------|
| P1 | US1-3 (Core Isolation) | T001-T020 | ✅ Complete |
| P2 | US4 (Seamless UX) | T021-T023 | ✅ Complete |
| P2 | US5 (Profile Display) | T031-T040 | ✅ Complete |
| P2 | US7-8 (Incognito Chats) | T041-T060 | ✅ Complete |
| P3 | US6 (Profile Polish) | T061-T064 | Pending |
| P3 | US9 (Incognito Polish) | T065-T069 | Pending |
| - | Deployment & Verification | T027-T030, T070-T073 | Pending |
| **Total** | | **73** | |

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No project initialization needed - modifying existing codebase

This feature modifies an existing codebase. No setup tasks required.

**Checkpoint**: Ready to begin foundational tasks.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core authentication infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete - these tasks enable user identity resolution.

- [x] T001 Add `get_user_workspace_client(token: str) -> WorkspaceClient` function in src/deep_research/core/auth.py (after line 65)
- [x] T002 Update `extract_obo_token()` docstring to remove DEPRECATED notice in src/deep_research/core/auth.py (lines 85-100)
- [x] T003 Modify `get_current_user_identity()` to prioritize OBO token extraction in src/deep_research/middleware/auth.py (lines 18-67)
- [x] T004 Add info-level logging for OBO auth success with user email and id in src/deep_research/middleware/auth.py
- [x] T005 Run type checks to verify new functions have proper annotations: `make typecheck`

**Checkpoint**: Foundation ready - OBO authentication flow is in place. User story implementation can now begin.

---

## Phase 3: User Story 1 - Private Chat Ownership (Priority: P1) 🎯 MVP

**Goal**: Users see only their own chats when logged in. Chats are associated with real user identity, not service principal.

**Independent Test**: Log in as User A, create a chat. Log in as User B, verify User B cannot see User A's chat in the list.

**Why MVP**: This is the core privacy fix. Once complete, new chats are properly isolated by user.

### Tests for User Story 1

- [x] T006 [P] [US1] Create unit test for `get_user_workspace_client()` in tests/unit/core/test_auth_obo.py
- [x] T007 [P] [US1] Create unit test for `extract_obo_token()` with valid/missing headers in tests/unit/core/test_auth_obo.py
- [x] T008 [P] [US1] Create integration test for OBO auth flow in middleware in tests/integration/middleware/test_auth_middleware.py

### Implementation for User Story 1

- [x] T009 [US1] Verify existing `ChatService.get_for_user()` correctly filters by user_id in src/deep_research/services/chat_service.py (read-only verification)
- [x] T010 [US1] Verify existing `ChatService.list()` correctly filters by user_id in src/deep_research/services/chat_service.py (read-only verification)
- [x] T011 [US1] Run existing unit tests to ensure no regressions: `make test`

**Checkpoint**: User Story 1 complete. New chats are associated with real user IDs and filtered correctly. Run `make test` to verify.

---

## Phase 4: User Story 2 - Chat Access Control (Priority: P1)

**Goal**: Direct URL access to another user's chat returns 404. API modification attempts are rejected.

**Independent Test**: Get User A's chat ID, attempt to access it as User B via direct URL `/chat/{id}`. Should return 404.

### Tests for User Story 2

- [x] T012 [P] [US2] Create integration test for direct chat access denial in tests/integration/api/test_chat_access_control.py

### Implementation for User Story 2

- [x] T013 [US2] Verify existing `verify_chat_ownership()` returns NotFoundError (not 403) in src/deep_research/api/v1/utils/authorization.py (read-only verification)
- [x] T014 [US2] Verify all chat endpoints use `verify_chat_ownership()` consistently in src/deep_research/api/v1/chats.py (read-only verification)
- [x] T015 [US2] Verify all message endpoints use `verify_chat_ownership()` consistently in src/deep_research/api/v1/messages.py (read-only verification)

**Checkpoint**: User Story 2 complete. Direct access attempts return 404. Run `make test-integration` to verify.

---

## Phase 5: User Story 3 - Research Session Protection (Priority: P1)

**Goal**: Users cannot cancel or access another user's research sessions.

**Independent Test**: User A starts research. User B attempts to cancel via `POST /research/{session_id}/cancel`. Should return 404.

### Tests for User Story 3

- [x] T016 [P] [US3] Create integration test for research session cancel denial in tests/integration/api/test_research_access_control.py

### Implementation for User Story 3

- [x] T017 [US3] Add ownership verification to `cancel_research()` endpoint in src/deep_research/api/v1/research.py (lines 34-64)
- [x] T018 [US3] Import ChatService and Message model in cancel_research function in src/deep_research/api/v1/research.py
- [x] T019 [US3] Verify ownership chain: session -> message -> chat -> user_id match in src/deep_research/api/v1/research.py
- [x] T020 [US3] Return NotFoundError (404) for unauthorized access to prevent information leakage in src/deep_research/api/v1/research.py

**Checkpoint**: User Story 3 complete. Research sessions are protected. Run `make test-integration` to verify.

---

## Phase 6: User Story 4 - Seamless User Experience (Priority: P2)

**Goal**: Multi-user isolation works transparently. New users see empty chat list. Auth errors show clear messages.

**Independent Test**: New user logs in and sees empty chat list. Existing user sees their chats across devices.

### Implementation for User Story 4

- [x] T021 [US4] Verify anonymous fallback is disabled in production mode in src/deep_research/middleware/auth.py
- [x] T022 [US4] Verify HTTPException with 401 status is raised when auth fails in production in src/deep_research/middleware/auth.py
- [x] T023 [US4] Add warning-level logging for OBO auth failures with error details in src/deep_research/middleware/auth.py

**Checkpoint**: User Story 4 complete. UX is seamless and auth failures are handled gracefully.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation, and deployment

- [x] T024 Run full type check suite: `make typecheck` (pre-existing errors, new code type-safe)
- [x] T025 Run full unit test suite: `make test` (1030 passed, 1 pre-existing failure)
- [x] T026 Run integration test suite: `make test-integration` (25 passed)
- [ ] T027 [P] Deploy to dev environment: `make deploy TARGET=dev`
- [ ] T028 [P] Check logs for OBO auth success: `make logs TARGET=dev SEARCH="--search 'OBO auth'"`
- [ ] T029 Manual multi-user verification: Two users, verify isolation
- [ ] T030 Verify database user_ids are real users not service principal: `SELECT user_id, title FROM chats ORDER BY created_at DESC LIMIT 5;`

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)           → N/A (no setup needed)
         ↓
Phase 2 (Foundational)    → BLOCKS all user stories
         ↓
Phase 3-6 (User Stories)  → Can proceed sequentially OR in parallel
         ↓
Phase 7 (Polish)          → Depends on all user stories complete
```

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Phase 2 completion. No dependencies on other stories.
- **User Story 2 (P1)**: Depends on Phase 2 completion. No dependencies on other stories. (Existing code already handles this)
- **User Story 3 (P1)**: Depends on Phase 2 completion. No dependencies on other stories. (New security fix)
- **User Story 4 (P2)**: Depends on Phase 2 completion. No dependencies on other stories. (UX verification)

### Within Each User Story

1. Tests first (verify they fail) → marked with [P] for parallel
2. Implementation tasks (sequential)
3. Verification at checkpoint

### Parallel Opportunities

**Phase 2 (Foundational)**:
- T001 and T002 can run in parallel (different functions in same file, but independent)

**User Story Tests**:
- All test tasks (T006-T008, T012, T016) marked [P] can run in parallel

**User Stories Themselves**:
- US1, US2, US3, US4 can run in parallel after Phase 2 completes (different files, no conflicts)

**Deployment**:
- T027 and T028 can run in parallel

---

## Parallel Example: Phase 2 Foundation

```bash
# These can be implemented together (same file, but independent functions):
Task: "T001 Add get_user_workspace_client() function in src/deep_research/core/auth.py"
Task: "T002 Update extract_obo_token() docstring in src/deep_research/core/auth.py"
```

## Parallel Example: User Story Tests

```bash
# Launch all test tasks together:
Task: "T006 [P] [US1] Create unit test for get_user_workspace_client()"
Task: "T007 [P] [US1] Create unit test for extract_obo_token()"
Task: "T008 [P] [US1] Create integration test for OBO auth flow"
Task: "T012 [P] [US2] Create integration test for direct chat access denial"
Task: "T016 [P] [US3] Create integration test for research session cancel denial"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational (T001-T005)
2. Complete Phase 3: User Story 1 (T006-T011)
3. **STOP and VALIDATE**: Test with two users
4. Deploy to dev for initial verification

### Incremental Delivery

1. Phase 2 → Foundation ready (OBO auth works)
2. User Story 1 → Chat list isolation works → **MVP READY**
3. User Story 3 → Research session security fix → **CRITICAL SECURITY**
4. User Story 2 → Direct access control (mostly verification - already works)
5. User Story 4 → UX polish
6. Phase 7 → Full verification and deployment

### Recommended Order

Since US1, US2, and US3 are all P1 priority and US3 fixes a security vulnerability:

1. **Phase 2**: Foundation (MUST complete first)
2. **US1**: Core privacy fix (enables correct user_id)
3. **US3**: Security fix for cancel_research (CRITICAL)
4. **US2**: Verify existing access controls (mostly read-only)
5. **US4**: UX verification (P2 priority)
6. **Phase 7**: Deploy and verify

---

## Summary

| Phase | Tasks | Parallel | Files Modified |
|-------|-------|----------|----------------|
| Phase 2: Foundation | T001-T005 | T001-T002 | auth.py, middleware/auth.py |
| Phase 3: US1 | T006-T011 | T006-T008 | tests/*, services/* (read-only) |
| Phase 4: US2 | T012-T015 | T012 | tests/*, api/* (read-only) |
| Phase 5: US3 | T016-T020 | T016 | tests/*, api/v1/research.py |
| Phase 6: US4 | T021-T023 | None | middleware/auth.py (verification) |
| Phase 7: Polish | T024-T030 | T027-T028 | None (verification only) |

**Total Tasks**: 30
**Files Modified**: 3 (auth.py, middleware/auth.py, api/v1/research.py)
**New Test Files**: 3 (test_auth_obo.py, test_chat_access_control.py, test_research_access_control.py)

---

## Phase 8: User Story 5 - User Profile Display (Priority: P2)

**Goal**: Display authenticated user's identity in the sidebar with avatar, name, and expandable dropdown.

**Independent Test**: Log in and verify profile shows your name/email. Click to expand dropdown with full details.

**Dependencies**: Phase 2 (Foundational) must be complete. Can run in parallel with Phases 3-7.

### Backend Tasks for User Story 5

- [x] T031 [P] [US5] Create `UserProfileResponse` Pydantic schema in src/deep_research/schemas/user.py (NEW file)
  - Fields: `user_id: str`, `email: str`, `display_name: str`, `workspace: str | None`

- [x] T032 [US5] Create user router with `GET /api/v1/user/profile` endpoint in src/deep_research/api/v1/user.py (NEW file)
  - Depends on `get_current_user` for identity
  - Returns `UserProfileResponse` from `request.state.user`

- [x] T033 [US5] Register user router in src/deep_research/api/v1/__init__.py
  - Import and include `user.router`

- [ ] T034 [P] [US5] Create unit test for profile endpoint in tests/unit/api/test_user_profile.py

### Frontend Tasks for User Story 5

- [x] T035 [P] [US5] Add `UserProfile` TypeScript interface to frontend/src/types/index.ts
  - Fields: `userId`, `email`, `displayName`, `workspace`

- [x] T036 [P] [US5] Create `useUserProfile` hook in frontend/src/hooks/useUserProfile.ts (NEW file)
  - TanStack Query for `GET /api/v1/user/profile`
  - 5-minute stale time for caching

- [x] T037 [P] [US5] Create avatar color utility in frontend/src/utils/avatarColors.ts (NEW file)
  - `getAvatarColor(userId: string): string` - hash-based color selection
  - `getInitials(displayName: string, email: string): string` - extract initials

- [x] T038 [US5] Create `UserAvatar` component in frontend/src/components/user/UserAvatar.tsx (NEW file)
  - Props: `userId`, `displayName`, `email`, `size`
  - Circular avatar with initials and deterministic background color

- [x] T039 [US5] Create `UserDropdown` component in frontend/src/components/user/UserDropdown.tsx (NEW file)
  - Shows full email, workspace info
  - Smooth open/close animations

- [x] T040 [US5] Create `UserProfile` component in frontend/src/components/user/UserProfile.tsx (NEW file)
  - Combines avatar, display name, dropdown trigger
  - Handles loading and error states

**Checkpoint**: User Story 5 complete. Profile displays in sidebar with avatar and dropdown.

---

## Phase 9: Integrate Profile into Sidebar

**Purpose**: Connect the UserProfile component to the main UI

- [x] T041 [US5] Import and add `UserProfile` to sidebar footer in frontend/src/components/chat/ChatSidebar.tsx
  - Position at bottom of sidebar
  - Consistent styling with existing UI

- [ ] T042 [P] [US5] Create component test for `UserProfile` in frontend/src/__tests__/components/UserProfile.test.tsx
  - Test loading, success, and error states
  - Test dropdown expand/collapse

**Checkpoint**: Profile integrated into main UI. Users can identify their logged-in account.

---

## Phase 10: User Story 7-8 - Incognito Chats Backend (Priority: P2)

**Goal**: Create infrastructure for ephemeral chat sessions with server-side storage.

**Independent Test**: Create incognito chat, conduct research, close browser, verify chat is deleted.

**Dependencies**: Phase 2 (Foundational) must be complete. Can run in parallel with Profile work.

### Database & Models

- [x] T043 [US7] Create `ChatType` enum in src/deep_research/models/chat.py
  - Values: `REGULAR = "regular"`, `INCOGNITO = "incognito"`

- [x] T044 [US7] Create `IncognitoSession` model in src/deep_research/models/incognito_session.py (NEW file)
  - Fields: `id`, `user_id`, `session_token`, `last_activity`, `expires_at`, `created_at`
  - Indexes on `session_token` (unique), `expires_at`, `user_id`

- [x] T045 [US7] Add incognito fields to `Chat` model in src/deep_research/models/chat.py
  - `chat_type: ChatType` (default REGULAR, indexed)
  - `incognito_session_id: UUID | None` (FK to incognito_sessions, CASCADE delete)

- [x] T046 [US7] Create Alembic migration 013_incognito_support.py in src/deep_research/db/migrations/versions/
  - Create `chattype` enum
  - Create `incognito_sessions` table
  - Add `chat_type` and `incognito_session_id` columns to `chats`
  - Run: `make db-migrate-local` to test

### Services

- [x] T047 [US7] Create `SessionService` in src/deep_research/services/session_service.py (NEW file)
  - `MAX_INCOGNITO_CHATS = 5`
  - `SESSION_TTL_HOURS = 1`
  - Methods: `get_or_create_session()`, `touch_session()`, `count_incognito_chats()`, `cleanup_expired()`

- [ ] T048 [P] [US7] Create unit tests for SessionService in tests/unit/services/test_session_service.py
  - Test session creation, TTL extension, quota enforcement, cleanup

### Schemas

- [x] T049 [P] [US7] Create incognito session schemas in src/deep_research/schemas/session.py (NEW file)
  - `IncognitoSessionStatus`: `has_session`, `chat_count`, `max_chats`, `expires_at`

- [x] T050 [P] [US7] Extend chat schemas in src/deep_research/schemas/chat.py
  - Add `chat_type: ChatType` to response schemas
  - Add `incognito: bool = False` to create request

### API Endpoints

- [x] T051 [US7] Add `POST /api/v1/chats/incognito` endpoint in src/deep_research/api/v1/chats.py
  - Create/get incognito session from cookie
  - Enforce 5-chat limit
  - Set `incognito_session` cookie (httpOnly, SameSite=Strict, 1hr max-age)

- [x] T052 [US7] Add `GET /api/v1/chats/incognito` endpoint in src/deep_research/api/v1/chats.py
  - List incognito chats for current browser session
  - Include session expiry info

- [x] T053 [US8] Add `POST /api/v1/chats/{chat_id}/convert` endpoint in src/deep_research/api/v1/chats.py
  - Convert incognito to permanent (update `chat_type`, clear session FK)
  - Preserve chat ID and all content

- [x] T054 [US7] Add `GET /api/v1/session/incognito` endpoint in src/deep_research/api/v1/chats.py
  - Return session status and quota

- [ ] T055 [US7] Add `chat_type` query parameter to `GET /api/v1/chats` in src/deep_research/api/v1/chats.py
  - Values: `regular` (default), `incognito`, `all`

### Background Cleanup

- [ ] T056 [US7] Create background cleanup task for expired sessions in src/deep_research/core/background.py
  - Run every 5 minutes
  - Delete sessions where `expires_at < now()`
  - Log cleanup stats

### Integration Tests

- [ ] T057 [P] [US7] Create integration tests in tests/integration/api/test_incognito_chats.py
  - Test session creation with cookie
  - Test quota enforcement (max 5 chats)
  - Test chat conversion
  - Test session cleanup cascade

**Checkpoint**: Incognito backend complete. Sessions tracked server-side with TTL cleanup.

---

## Phase 11: User Story 7-8 - Incognito Chats Frontend (Priority: P2)

**Goal**: UI for creating, viewing, and managing incognito chats.

### TypeScript Types

- [x] T058 [P] [US7] Add incognito types to frontend/src/types/index.ts
  - `ChatType = 'regular' | 'incognito'`
  - Extend `Chat` interface with `chatType`
  - `IncognitoSessionStatus` interface

### Hooks

- [x] T059 [US7] Create `useIncognitoChats` hook in frontend/src/hooks/useIncognitoChats.ts (NEW file)
  - Query: list incognito chats
  - Query: session status
  - Mutation: create incognito chat
  - Mutation: convert to permanent

### UI Components

- [x] T060 [US7] Add incognito option to `NewChatButton` in frontend/src/components/chat/ChatSidebar.tsx
  - Dropdown or secondary button for incognito creation
  - Disable when at quota limit

- [x] T061 [US8] Add incognito section to `ChatSidebar` in frontend/src/components/chat/ChatSidebar.tsx
  - Separate section header with incognito icon
  - List incognito chats with distinct styling
  - Session expiry countdown

- [x] T062 [US8] Create `IncognitoIndicator` component in frontend/src/components/incognito/IncognitoIndicator.tsx (NEW file)
  - Persistent indicator in chat header
  - Privacy icon (mask or eye-slash)
  - "Temporary chat" label

- [x] T063 [US8] Create `KeepChatDialog` component in frontend/src/components/incognito/KeepChatDialog.tsx (NEW file)
  - "Keep this chat" button in incognito chats
  - Confirmation dialog explaining conversion
  - Calls convert mutation

### Component Tests

- [ ] T064 [P] [US7-8] Create component tests in frontend/src/__tests__/components/
  - `IncognitoIndicator.test.tsx`
  - `KeepChatDialog.test.tsx`

**Checkpoint**: Incognito frontend complete. Users can create, manage, and convert incognito chats.

---

## Phase 12: User Story 6 - Profile Visual Polish (Priority: P3)

**Goal**: Refine profile component visual appearance to professional standards.

- [ ] T065 [US6] Polish `UserAvatar` styling in frontend/src/components/user/UserAvatar.tsx
  - Smooth circular shape with proper anti-aliasing
  - Subtle shadow/border
  - Refined typography for initials
  - Hover state effects

- [ ] T066 [US6] Polish `UserDropdown` animations in frontend/src/components/user/UserDropdown.tsx
  - Smooth fade/slide animation on open
  - Subtle drop shadow
  - Clear visual hierarchy
  - Proper z-index handling

- [ ] T067 [US6] Add responsive behavior in frontend/src/components/user/UserProfile.tsx
  - Avatar-only mode on narrow viewports
  - Dropdown remains accessible
  - Graceful transitions

- [ ] T068 [P] [US6] Visual review against design criteria
  - Consistent with app design system
  - Professional appearance
  - Cross-browser testing (Chrome, Firefox, Safari, Edge)

**Checkpoint**: User Story 6 complete. Profile has polished, professional appearance.

---

## Phase 13: User Story 9 - Incognito Visual Polish (Priority: P3)

**Goal**: Refine incognito visual indicators for clarity without distraction.

- [ ] T069 [US9] Polish `IncognitoIndicator` design in frontend/src/components/incognito/IncognitoIndicator.tsx
  - Appropriate privacy icon (mask or eye-slash)
  - Subtle header tint
  - Non-intrusive styling

- [ ] T070 [US9] Create `IncognitoBanner` component in frontend/src/components/incognito/IncognitoBanner.tsx (NEW file)
  - Brief explanation of incognito behavior
  - Dismissible banner
  - Close confirmation for unsaved content

- [ ] T071 [US9] Add incognito tooltips across components
  - Tooltip on indicator: "This chat will be deleted when you close it"
  - Tooltip on section header
  - Quick (<500ms) tooltip display

- [ ] T072 [US9] Polish incognito section styling in frontend/src/components/chat/ChatSidebar.tsx
  - Subtle background differentiation
  - Consistent iconography
  - Expiry countdown styling

- [ ] T073 [P] [US9] Visual review for clarity
  - Users can distinguish incognito vs regular within 2 seconds
  - Non-intrusive but clear indicators

**Checkpoint**: User Story 9 complete. Incognito mode is visually clear and polished.

---

## Phase 14: End-to-End Testing

**Purpose**: Full integration testing across all new features

- [ ] T074 [US5] Create E2E test for user profile in e2e/tests/user-profile.spec.ts
  - Profile displays on login
  - Avatar shows correct initials
  - Dropdown shows full details

- [ ] T075 [US7-8] Create E2E test for incognito chats in e2e/tests/incognito-chats.spec.ts
  - Create incognito chat
  - Conduct research in incognito
  - Convert to permanent
  - Verify session expiry behavior

- [ ] T076 Full E2E suite run: `make e2e`

**Checkpoint**: All E2E tests pass. Features work end-to-end.

---

## Phase 15: Final Deployment & Verification

**Purpose**: Deploy all features and verify in production-like environment

- [ ] T027 [P] Deploy to dev environment: `make deploy TARGET=dev`
- [ ] T028 [P] Check logs for OBO auth success: `make logs TARGET=dev SEARCH="--search 'OBO auth'"`
- [ ] T029 Manual multi-user verification: Two users, verify isolation
- [ ] T030 Verify database user_ids are real users not service principal: `SELECT user_id, title FROM chats ORDER BY created_at DESC LIMIT 5;`
- [ ] T077 Verify profile displays correctly in deployed environment
- [ ] T078 Verify incognito chat creation and cleanup in deployed environment
- [ ] T079 Run database migration on remote: `make db-migrate-remote TARGET=dev`

**Checkpoint**: Feature fully deployed and verified in dev environment.

---

## Updated Dependencies & Execution Order

### Phase Dependencies (Updated)

```
Phase 1-2 (Foundation)    → BLOCKS all user stories
         ↓
Phase 3-6 (US1-4)         → Core isolation (✅ Complete)
         ↓
Phase 7 (Polish Core)     → Type checks, tests (✅ Complete)
         ↓
┌────────────────────────────────────────────┐
│     Can proceed in PARALLEL:               │
│  Phase 8-9 (US5: Profile)                  │
│  Phase 10-11 (US7-8: Incognito)            │
└────────────────────────────────────────────┘
         ↓
Phase 12-13 (US6, US9)    → Visual polish (after P2 features)
         ↓
Phase 14 (E2E Testing)    → Depends on all features complete
         ↓
Phase 15 (Deployment)     → Final verification
```

### Parallel Opportunities (Updated)

**Profile vs Incognito Backend**:
- T031-T042 (Profile) and T043-T057 (Incognito Backend) can run in parallel

**Frontend Components**:
- T058-T064 (Incognito Frontend) depends on T043-T057 (Incognito Backend)
- But can run in parallel with T065-T068 (Profile Polish)

**Tests**:
- All test tasks marked [P] can run in parallel within their phase

---

## Updated Summary

| Phase | Tasks | Status | Files Modified |
|-------|-------|--------|----------------|
| Phase 2: Foundation | T001-T005 | ✅ Complete | auth.py, middleware/auth.py |
| Phase 3-6: US1-4 | T006-T023 | ✅ Complete | tests/*, api/v1/research.py |
| Phase 7: Core Polish | T024-T026 | ✅ Complete | None (verification) |
| Phase 8-9: US5 Profile | T031-T042 | Pending | schemas/user.py, api/v1/user.py, frontend/components/user/* |
| Phase 10-11: US7-8 Incognito | T043-T064 | Pending | models/*, services/session_service.py, api/v1/chats.py, frontend/components/incognito/* |
| Phase 12: US6 Profile Polish | T065-T068 | Pending | frontend/components/user/* |
| Phase 13: US9 Incognito Polish | T069-T073 | Pending | frontend/components/incognito/* |
| Phase 14: E2E Testing | T074-T076 | Pending | e2e/tests/* |
| Phase 15: Deployment | T027-T030, T077-T079 | Pending | None (deployment) |

**Total Tasks**: 73
**New Backend Files**: 4 (schemas/user.py, api/v1/user.py, models/incognito_session.py, services/session_service.py)
**New Frontend Files**: ~8 (components/user/*, components/incognito/*, hooks/*, utils/*)
**New Test Files**: ~6 (various)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1-4 (Core Isolation) is complete - focus on US5-9 (Profile + Incognito)
- Profile (US5-6) and Incognito (US7-9) can be developed in parallel
- Database migration (T046) must run before any incognito API work
- Commit after each phase or logical group
- Stop at any checkpoint to validate story independently
