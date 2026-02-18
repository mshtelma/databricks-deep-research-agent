# Data Model: User Chat Isolation

**Feature**: 006-user-chat-isolation
**Date**: 2026-01-25 (Updated)

## Overview

This feature has two data model aspects:
1. **User Isolation** - No schema changes needed; existing `user_id` column works
2. **Incognito Chats** - Requires new table and columns for ephemeral session storage

## Existing Entities

### UserIdentity (Runtime Object)

**Location**: `src/deep_research/core/auth.py`
**Type**: Frozen dataclass (not persisted)

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `str` | Unique identifier from Databricks workspace |
| `email` | `str` | User's email address |
| `display_name` | `str` | Human-readable name |

**Factory Methods**:
- `from_workspace_user(user: Any) -> UserIdentity` - Create from Databricks API response
- `anonymous() -> UserIdentity` - Create anonymous user for development

**Key Change**: No schema change. The values populated will now be real user IDs instead of service principal IDs.

---

### Chat (Persistent Entity)

**Location**: `src/deep_research/models/chat.py`
**Table**: `chats`

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `id` | `UUID` | PK | Chat identifier |
| `user_id` | `str(255)` | NOT NULL, INDEX | **Owner's unique identifier** |
| `title` | `str(500)` | NULLABLE | Chat display name |
| `status` | `ChatStatus` | DEFAULT ACTIVE | ACTIVE, ARCHIVED, DELETED |
| `deleted_at` | `datetime` | NULLABLE | Soft delete timestamp |
| `metadata_` | `JSONB` | DEFAULT {} | Additional data |

**Indexes**:
- `idx_chats_user_status` on `(user_id, status)` - Optimizes list queries

**Key Change**: No schema change. The `user_id` column already exists and is indexed.

---

### Message (Persistent Entity)

**Location**: `src/deep_research/models/message.py`
**Table**: `messages`

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `id` | `UUID` | PK | Message identifier |
| `chat_id` | `UUID` | FK → chats.id, CASCADE | Parent chat |
| `role` | `MessageRole` | NOT NULL | user, assistant, system |
| `content` | `TEXT` | NOT NULL | Message content |
| ... | ... | ... | Other fields unchanged |

**Ownership**: Inherited from parent `Chat.user_id` via foreign key relationship.

**Key Change**: None. Ownership verification uses `message.chat.user_id`.

---

### ResearchSession (Persistent Entity)

**Location**: `src/deep_research/models/research_session.py`
**Table**: `research_sessions`

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `id` | `UUID` | PK | Session identifier |
| `message_id` | `UUID` | FK → messages.id | Associated message |
| `user_id` | `str(255)` | INDEX | Direct user lookup (denormalized) |
| `status` | `ResearchStatus` | NOT NULL | IN_PROGRESS, COMPLETED, FAILED, CANCELLED |
| ... | ... | ... | Other fields unchanged |

**Ownership Paths**:
1. Direct: `session.user_id` (denormalized for quick lookup)
2. Via message: `session.message.chat.user_id` (authoritative)

**Key Change**: The security fix uses path #2 to verify ownership in `cancel_research`.

---

## Entity Relationships

```
┌─────────────────┐
│   UserIdentity  │  (Runtime only - not persisted)
│   - user_id     │
│   - email       │
│   - display_name│
└────────┬────────┘
         │ owns (via user_id match)
         ▼
┌─────────────────┐
│      Chat       │
│   - id          │
│   - user_id ◄───┼──── Ownership key
│   - title       │
│   - status      │
└────────┬────────┘
         │ has many (FK: chat_id)
         ▼
┌─────────────────┐
│    Message      │
│   - id          │
│   - chat_id     │
│   - role        │
│   - content     │
└────────┬────────┘
         │ has one (FK: message_id)
         ▼
┌─────────────────┐
│ ResearchSession │
│   - id          │
│   - message_id  │
│   - user_id     │  (denormalized)
│   - status      │
└─────────────────┘
```

## Ownership Verification Patterns

### Pattern 1: Chat Access (Existing)

```python
# In ChatService
async def get_for_user(chat_id: UUID, user_id: str) -> Chat | None:
    return await self._session.execute(
        select(Chat)
        .where(Chat.id == chat_id)
        .where(Chat.user_id == user_id)
        .where(Chat.deleted_at.is_(None))
    )
```

### Pattern 2: Message Access (Existing)

```python
# In authorization.py
async def verify_message_ownership(message_id: UUID, user_id: str, db: AsyncSession) -> Message:
    message = await db.get(Message, message_id)
    if message.chat.user_id != user_id:
        raise NotFoundError("Message", str(message_id))
    return message
```

### Pattern 3: Research Session Access (NEW - to be implemented)

```python
# In research.py cancel_research endpoint
session = await service.get(session_id)
message = await db.get(Message, session.message_id)
chat = await chat_service.get_by_id(message.chat_id)
if chat.user_id != user.user_id:
    raise NotFoundError("ResearchSession", str(session_id))
```

## Migration Notes

### Migration 013: Add Incognito Support

**Required for incognito chats feature.**

```python
# src/deep_research/db/migrations/versions/013_incognito_support.py

def upgrade():
    # 1. Create ChatType enum
    chat_type_enum = postgresql.ENUM('regular', 'incognito', name='chattype')
    chat_type_enum.create(op.get_bind())

    # 2. Create incognito_sessions table
    op.create_table(
        'incognito_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('session_token', sa.String(64), nullable=False, unique=True),
        sa.Column('last_activity', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )

    # 3. Add columns to chats table
    op.add_column('chats', sa.Column(
        'chat_type',
        chat_type_enum,
        nullable=False,
        server_default='regular'
    ))
    op.add_column('chats', sa.Column(
        'incognito_session_id',
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey('incognito_sessions.id', ondelete='CASCADE'),
        nullable=True
    ))

    # 4. Create indexes
    op.create_index('idx_chats_type', 'chats', ['chat_type'])
    op.create_index('idx_chats_incognito_session', 'chats', ['incognito_session_id'])

def downgrade():
    op.drop_index('idx_chats_incognito_session')
    op.drop_index('idx_chats_type')
    op.drop_column('chats', 'incognito_session_id')
    op.drop_column('chats', 'chat_type')
    op.drop_table('incognito_sessions')
    postgresql.ENUM(name='chattype').drop(op.get_bind())
```

---

## New Entities (for Incognito Chats)

### IncognitoSession

Server-side session tracking for incognito chat lifecycle.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | UUID | PK, default uuid4 | Unique session identifier |
| `user_id` | string(255) | indexed, not null | Databricks user ID owning this session |
| `session_token` | string(64) | unique, indexed, not null | Browser cookie token (url-safe 32 chars) |
| `last_activity` | datetime | not null, default now | Last interaction timestamp |
| `expires_at` | datetime | not null, indexed | Computed: last_activity + 1 hour |
| `created_at` | datetime | not null, default now | Session creation time |

**Relationships**:
- One-to-Many → `Chat` (via `incognito_session_id`)

**Lifecycle**:
1. Created on first incognito chat request if no valid session exists
2. `last_activity` updated on any incognito chat interaction
3. `expires_at` recalculated as `last_activity + 1 hour`
4. Deleted by background cleanup when `expires_at < now()`
5. Cascade deletes all associated incognito chats

---

### Chat (Modifications)

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `chat_type` | ChatType enum | not null, default 'regular', indexed | NEW: Distinguishes regular vs incognito |
| `incognito_session_id` | UUID \| None | FK → incognito_sessions, nullable, on delete CASCADE | NEW: Links to session for incognito chats |

**New Enum**:
```python
class ChatType(str, Enum):
    REGULAR = "regular"
    INCOGNITO = "incognito"
```

**Validation Rules**:
- If `chat_type == INCOGNITO`, then `incognito_session_id` MUST be set
- If `chat_type == REGULAR`, then `incognito_session_id` MUST be null

---

## Entity Relationship Diagram (Updated)

```
┌─────────────────────┐     ┌──────────────────────┐
│   UserIdentity      │     │  IncognitoSession    │
│   (runtime only)    │     │  (NEW table)         │
├─────────────────────┤     ├──────────────────────┤
│ user_id: str        │     │ id: UUID (PK)        │
│ email: str          │     │ user_id: str (idx)   │
│ display_name: str   │     │ session_token: str   │
└─────────────────────┘     │ last_activity: dt    │
         │                  │ expires_at: dt       │
         │ owns             └──────────────────────┘
         │                           │
         ▼                           │ 1:N (cascade delete)
┌─────────────────────────────────────────────────────┐
│                        Chat                          │
├─────────────────────────────────────────────────────┤
│ id: UUID (PK)                                        │
│ user_id: str (idx)                                   │
│ title: str | None                                    │
│ status: ChatStatus                                   │
│ chat_type: ChatType (NEW) ─────────────────────────┐ │
│ incognito_session_id: UUID | None (FK) (NEW) ──────┼─┘
│ ...                                                  │
└─────────────────────────────────────────────────────┘
         │
         │ 1:N (cascade delete)
         ▼
┌─────────────────────────────────────────────────────┐
│                      Message                         │
└─────────────────────────────────────────────────────┘
```

---

## Query Patterns (Incognito)

### List user's regular chats
```sql
SELECT * FROM chats
WHERE user_id = :user_id
  AND chat_type = 'regular'
  AND deleted_at IS NULL
ORDER BY updated_at DESC;
```

### List user's incognito chats (for current session)
```sql
SELECT c.* FROM chats c
JOIN incognito_sessions s ON c.incognito_session_id = s.id
WHERE s.session_token = :session_token
  AND c.chat_type = 'incognito'
  AND c.deleted_at IS NULL
ORDER BY c.updated_at DESC;
```

### Count incognito chats per session
```sql
SELECT COUNT(*) FROM chats
WHERE incognito_session_id = :session_id
  AND chat_type = 'incognito'
  AND deleted_at IS NULL;
```

### Cleanup expired sessions (background task)
```sql
DELETE FROM incognito_sessions
WHERE expires_at < NOW();
-- Cascade deletes associated chats
```

### Convert incognito to regular
```sql
UPDATE chats
SET chat_type = 'regular',
    incognito_session_id = NULL,
    updated_at = NOW()
WHERE id = :chat_id
  AND chat_type = 'incognito';
```
