# Implementation Plan: User Chat Isolation

**Branch**: `006-user-chat-isolation` | **Date**: 2026-01-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/006-user-chat-isolation/spec.md`

## Summary

Implement multi-user chat isolation with three components:
1. **Core Isolation (P1)**: Enforce user_id ownership on all chat/message/research operations
2. **User Profile UI (P2)**: Display authenticated user identity in sidebar with avatar and dropdown
3. **Incognito Chats (P2)**: Ephemeral chat sessions stored server-side, deleted on session end or 1-hour idle timeout

The codebase already has `user_id` on chats and basic auth infrastructure. The implementation extends existing patterns with enhanced verification, session management for incognito chats, and new frontend components.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x (frontend)
**Primary Dependencies**: FastAPI, SQLAlchemy (async), React 18, TanStack Query
**Storage**: PostgreSQL (Databricks Lakebase) - existing + server-side session store for incognito
**Testing**: pytest (unit/integration), Playwright (E2E)
**Target Platform**: Databricks Apps (Linux server), modern browsers
**Project Type**: Web application (backend + frontend)
**Performance Goals**: <200ms auth latency (SC-005), <1s profile load (SC-007), <30s incognito cleanup (SC-014)
**Constraints**: 5 concurrent incognito chats (FR-026), 1-hour idle timeout (FR-021)
**Scale/Scope**: Multi-user deployment, existing ~12 migrations, ~30 API endpoints

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Clients and Workspace Integration | ✅ PASS | Auth uses `WorkspaceClient.current_user.me()` via OBO token; LLM calls use `get_workspace_client()` |
| II. Typing-First Python | ✅ PASS | All new code will have full type annotations; existing patterns use Pydantic + dataclasses |
| III. Avoid Runtime Introspection | ✅ PASS | Using explicit `UserIdentity` dataclass, not `hasattr`; Pydantic for validation |
| IV. Linting and Static Type Enforcement | ✅ PASS | Project uses mypy strict + ruff; CI enforces before merge |

**Gate Result**: PASSED - No violations. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/006-user-chat-isolation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── README.md        # API contract documentation
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
# Backend (Python/FastAPI)
src/deep_research/
├── models/
│   ├── chat.py              # Extend: add chat_type enum, session_id
│   └── incognito_session.py # NEW: server-side session model
├── services/
│   ├── chat_service.py      # Extend: incognito chat methods
│   └── session_service.py   # NEW: session management
├── api/v1/
│   ├── chats.py             # Extend: incognito endpoints
│   ├── user.py              # NEW: user profile endpoint
│   └── utils/
│       └── authorization.py # Extend: enhanced verification
├── middleware/
│   └── auth.py              # Extend: session tracking
├── schemas/
│   ├── chat.py              # Extend: chat_type field
│   ├── user.py              # NEW: profile schema
│   └── session.py           # NEW: session schema
└── core/
    └── auth.py              # Already has UserIdentity

# Frontend (TypeScript/React)
frontend/src/
├── components/
│   ├── chat/
│   │   ├── ChatSidebar.tsx       # Extend: incognito section
│   │   └── NewChatButton.tsx     # Extend: incognito option
│   ├── user/                     # NEW directory
│   │   ├── UserProfile.tsx       # Profile display component
│   │   ├── UserAvatar.tsx        # Avatar with initials
│   │   └── UserDropdown.tsx      # Expandable dropdown
│   └── incognito/                # NEW directory
│       ├── IncognitoIndicator.tsx
│       ├── IncognitoBanner.tsx
│       └── KeepChatDialog.tsx
├── hooks/
│   ├── useUserProfile.ts         # NEW: profile data hook
│   └── useIncognitoChats.ts      # NEW: incognito management
├── api/
│   └── client.ts                 # Extend: user + incognito endpoints
├── types/
│   └── index.ts                  # Extend: ChatType, UserProfile
└── utils/
    └── avatarColors.ts           # NEW: deterministic color generation

# Tests
tests/
├── unit/
│   ├── services/
│   │   └── test_session_service.py
│   └── core/
│       └── test_auth_obo.py
├── integration/
│   ├── api/
│   │   ├── test_chat_isolation.py
│   │   └── test_incognito_chats.py
│   └── middleware/
│       └── test_auth_middleware.py
└── e2e/
    └── tests/
        ├── chat-isolation.spec.ts
        └── incognito-chats.spec.ts

frontend/src/__tests__/
├── components/
│   ├── UserProfile.test.tsx
│   └── IncognitoIndicator.test.tsx
└── hooks/
    └── useIncognitoChats.test.ts
```

**Structure Decision**: Web application pattern. Backend extends existing `src/deep_research/` structure. Frontend extends existing `frontend/src/` structure. New components in dedicated directories (`user/`, `incognito/`) to maintain separation. Tests mirror source structure.

## Complexity Tracking

> No violations requiring justification. Design follows existing patterns.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Incognito storage | Server-side session store (PostgreSQL) | Survives page refresh (FR-020), handles research results, avoids localStorage limits |
| Session tracking | DB-backed with TTL cleanup | No Redis needed, persistence for crash recovery |
| Chat type | Enum field on existing Chat model | Minimal schema change, reuses existing relationships |

---

## Constitution Check (Post-Design)

*Re-evaluated after Phase 1 design completion.*

| Principle | Status | Post-Design Evidence |
|-----------|--------|---------------------|
| I. Clients and Workspace Integration | ✅ PASS | User profile uses `WorkspaceClient.current_user.me()`; no direct API calls |
| II. Typing-First Python | ✅ PASS | New models use `Mapped[]` annotations; Pydantic schemas fully typed |
| III. Avoid Runtime Introspection | ✅ PASS | `ChatType` enum, explicit FK relationships; no `hasattr` usage |
| IV. Linting and Static Type Enforcement | ✅ PASS | All new code designed for mypy strict compatibility |

**Post-Design Gate Result**: PASSED - Design complies with constitution.

---

## Phase Outputs

### Phase 0: Research ✅
- [research.md](./research.md) - 12 research questions resolved

### Phase 1: Design ✅
- [data-model.md](./data-model.md) - Entity schema with migration
- [contracts/README.md](./contracts/README.md) - API contracts
- [quickstart.md](./quickstart.md) - Implementation guide

### Phase 2: Tasks (Next)
- Run `/speckit.tasks` to generate task breakdown
