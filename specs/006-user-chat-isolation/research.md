# Research: User Chat Isolation

**Feature**: 006-user-chat-isolation
**Date**: 2026-01-25
**Status**: Complete

## Research Questions

### Q1: How does Databricks Apps forward user identity?

**Decision**: Use `x-forwarded-access-token` HTTP header

**Rationale**:
- Databricks Apps proxy automatically forwards the authenticated user's OAuth token via this header
- This is the official mechanism documented by Databricks for user authorization in apps
- The token can be used to create a `WorkspaceClient` that resolves to the actual user's identity

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Service principal only | Cannot differentiate between users |
| Custom auth header | Non-standard, requires additional infrastructure |
| Session cookies | Not supported in Databricks Apps architecture |

**Source**: [Databricks Apps Authorization Documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth)

---

### Q2: How to create a user-context WorkspaceClient?

**Decision**: Use `WorkspaceClient(host=host, token=user_token)`

**Rationale**:
- The Databricks SDK supports token-based authentication
- Pass the user's OAuth token directly to create a client that acts as that user
- Call `client.current_user.me()` to resolve the actual user identity

**Code Pattern**:
```python
def get_user_workspace_client(token: str) -> WorkspaceClient:
    settings = get_settings()
    host = settings.databricks_host
    if not host:
        sp_client = get_workspace_client()
        host = sp_client.config.host
    return WorkspaceClient(host=host, token=token)
```

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Environment variable override | Would affect all clients, not per-request |
| Custom HTTP client | Bypasses SDK, loses type safety |

---

### Q3: What is the existing code state?

**Decision**: Activate existing `extract_obo_token()` function

**Findings**:
- `extract_obo_token()` exists at `src/deep_research/core/auth.py:85-100`
- Currently marked as `DEPRECATED` with comment "OBO authentication is not currently used"
- The function correctly reads `x-forwarded-access-token` header
- Simply needs to be activated in the middleware

**Current Code**:
```python
def extract_obo_token(headers: dict[str, str]) -> str | None:
    """Extract OBO (On-Behalf-Of) token from request headers.

    DEPRECATED: OBO authentication is not currently used.  # <-- Remove this
    ...
    """
    return headers.get("x-forwarded-access-token")
```

---

### Q4: What security vulnerabilities exist?

**Decision**: Fix `cancel_research` endpoint (CRITICAL)

**Findings**:
- `POST /research/{session_id}/cancel` at `src/deep_research/api/v1/research.py:34-64`
- **No user ownership verification** - any authenticated user can cancel ANY research session
- Other endpoints correctly use `verify_chat_ownership()` or `verify_message_ownership()`

**Fix Required**:
```python
# Before cancellation, verify ownership via message -> chat -> user_id
session = await service.get(session_id)
message = await db.get(Message, session.message_id)
chat = await chat_service.get_by_id(message.chat_id)
if chat.user_id != user.user_id:
    raise NotFoundError("ResearchSession", str(session_id))  # 404 for privacy
```

---

### Q5: What is the dual-client strategy?

**Decision**: Keep service principal for backend, user token for identity only

**Rationale**:
- **User-context client**: Only used for identity resolution (`current_user.me()`)
- **Service principal client**: Used for all backend operations (LLM calls, search, etc.)
- This ensures backend operations have consistent permissions regardless of which user is logged in

**Architecture**:
```
Request → Middleware
           ├── OBO Token → User WorkspaceClient → UserIdentity (stored in request.state.user)
           └── SP Credentials → SP WorkspaceClient (stored in request.state.workspace_client)
```

---

## Resolved Unknowns Summary

| Unknown | Resolution |
|---------|------------|
| User identity extraction | `x-forwarded-access-token` header |
| Client creation | `WorkspaceClient(host, token)` |
| Existing code reuse | Activate `extract_obo_token()` |
| Security gaps | Fix `cancel_research` endpoint |
| Dual-client strategy | User client for identity, SP client for operations |

---

### Q6: How to implement server-side session storage for incognito chats?

**Decision**: Use PostgreSQL table with TTL-based cleanup

**Rationale**:
- PostgreSQL is already available (Lakebase)
- No additional infrastructure needed (no Redis required)
- Background cleanup task runs on configurable interval
- Survives server restarts (important for Databricks Apps)

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Redis/Memcached | Additional infrastructure, not available in Databricks Apps |
| Browser sessionStorage | Doesn't work across tabs, can't store research results |
| In-memory dict | Lost on server restart, doesn't scale to multiple instances |

**Data Model**:
```python
class IncognitoSession(Base):
    __tablename__ = "incognito_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(index=True)
    session_token: Mapped[str] = mapped_column(unique=True, index=True)
    last_activity: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime]  # last_activity + 1 hour
```

---

### Q7: How to differentiate regular vs incognito chats?

**Decision**: Add `chat_type` enum column to existing Chat model

**Rationale**:
- Minimal schema change (single column addition)
- Reuses all existing chat relationships (messages, sources, research_sessions)
- Easy filtering in queries
- Clear semantics in code

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Separate `IncognitoChat` table | Would duplicate schema, complicate queries |
| Boolean `is_incognito` flag | Less extensible if new types needed |
| Metadata field | Type safety concerns, harder to index |

**Code Pattern**:
```python
class ChatType(str, Enum):
    REGULAR = "regular"
    INCOGNITO = "incognito"

class Chat(BaseModel):
    chat_type: Mapped[ChatType] = mapped_column(
        SQLAlchemyEnum(ChatType),
        default=ChatType.REGULAR,
        index=True
    )
    incognito_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("incognito_sessions.id", ondelete="CASCADE"),
        nullable=True
    )
```

---

### Q8: How to track browser sessions for incognito lifecycle?

**Decision**: Server-generated session token in httpOnly cookie

**Rationale**:
- httpOnly prevents XSS access to session token
- Server controls session creation/validation
- Works across page refreshes within same browser session
- Cookie automatically sent with all requests

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| localStorage token | Vulnerable to XSS, persists across sessions |
| sessionStorage token | Lost on tab close, not shared across tabs |
| URL parameter | Bookmarkable, security risk |

**Code Pattern**:
```python
async def ensure_incognito_session(request: Request, response: Response):
    session_token = request.cookies.get("incognito_session")
    if not session_token:
        session_token = secrets.token_urlsafe(32)
        response.set_cookie(
            "incognito_session",
            session_token,
            httponly=True,
            samesite="strict",
            max_age=3600  # 1 hour
        )
    return session_token
```

---

### Q9: How to convert incognito chat to permanent?

**Decision**: Update `chat_type` in place, preserving chat ID

**Rationale**:
- Same chat ID preserved (no URL change needed)
- All messages, sources, research sessions preserved
- Simple atomic update
- User sees chat immediately in regular list

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Create new chat + copy | Expensive, changes URL, complex |
| Move to separate table | Would require data migration |
| Clone with new ID | URL changes, confusing UX |

---

### Q10: Where to display user profile in UI?

**Decision**: Sidebar footer with avatar + name, expandable dropdown

**Rationale**:
- Matches common SaaS patterns (Slack, Discord, Linear)
- Always visible without taking main content space
- Dropdown provides additional details without cluttering
- Avatar provides quick visual identity confirmation

**Alternatives Considered**:
| Alternative | Why Rejected |
|-------------|--------------|
| Header right corner | Takes space from chat title |
| Full sidebar section | Too prominent for secondary info |
| Floating widget | Obstructs content |

**Visual Design**:
```
┌─────────────────────────┐
│ 🕵️ Incognito (2)       │
│  └─ [🕵️] Temp research │
├─────────────────────────┤
│ [👤] John Doe    ▼      │  ← Sidebar footer
│     john@company.com    │     (on click dropdown)
└─────────────────────────┘
```

---

### Q11: How to generate deterministic avatar colors?

**Decision**: Hash user_id to select from predefined color palette

**Rationale**:
- Deterministic: same user always gets same color
- Visually distinct: curated palette ensures readability
- No external dependency: client-side computation
- Consistent across sessions: uses stable user_id

**Code Pattern**:
```typescript
const AVATAR_COLORS = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
  '#8B5CF6', '#EC4899', '#06B6D4', '#F97316'
];

function getAvatarColor(userId: string): string {
  const hash = userId.split('').reduce(
    (acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0
  );
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}
```

---

### Q12: How to enforce incognito chat limit?

**Decision**: Server-side count check with client-side pre-check for UX

**Rationale**:
- Server-side: Authoritative, prevents bypass
- Client-side: Better UX, shows limit before attempt
- Count per session: Multiple devices get independent limits

**Code Pattern**:
```python
MAX_INCOGNITO_CHATS = 5

async def create_incognito_chat(user_id: str, session_id: UUID) -> Chat:
    count = await self.db.scalar(
        select(func.count(Chat.id))
        .where(Chat.incognito_session_id == session_id)
        .where(Chat.chat_type == ChatType.INCOGNITO)
    )
    if count >= MAX_INCOGNITO_CHATS:
        raise ValidationError(
            f"Maximum {MAX_INCOGNITO_CHATS} incognito chats reached."
        )
    # Create chat...
```

---

## Resolved Unknowns Summary (Updated)

| Unknown | Resolution |
|---------|------------|
| User identity extraction | `x-forwarded-access-token` header |
| Client creation | `WorkspaceClient(host, token)` |
| Existing code reuse | Activate `extract_obo_token()` |
| Security gaps | Fix `cancel_research` endpoint |
| Dual-client strategy | User client for identity, SP client for operations |
| Incognito storage | PostgreSQL table with TTL cleanup |
| Chat differentiation | `chat_type` enum on Chat model |
| Session tracking | httpOnly cookie with server-generated token |
| Incognito conversion | Update in place, preserve chat ID |
| Profile placement | Sidebar footer with dropdown |
| Avatar colors | Hash-based selection from palette |
| Chat limit | Server-side enforcement |

## Next Phase

All research questions resolved. Proceed to Phase 1: Design & Contracts.
