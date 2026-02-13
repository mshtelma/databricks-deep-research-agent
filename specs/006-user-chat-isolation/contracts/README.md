# API Contracts: User Chat Isolation

**Feature**: 006-user-chat-isolation
**Date**: 2026-01-25 (Updated)

## Overview

This document covers API contracts for:
1. **User Isolation** - Behavioral changes to existing endpoints (no signature changes)
2. **User Profile API** - New endpoint for profile display
3. **Incognito Chats API** - New endpoints for ephemeral chat management

---

## Part 1: User Isolation (Behavioral Changes)

**No API contract changes required for core isolation.**

This modifies the **server-side behavior** of existing endpoints without changing their signatures or response formats:

| Endpoint | Change |
|----------|--------|
| All endpoints | `user_id` in responses now reflects actual user, not service principal |
| `POST /research/{session_id}/cancel` | Now enforces ownership (returns 404 for unauthorized) |

## Behavioral Changes

### Before (Bug)

```
GET /api/v1/chats
Authorization: Bearer <user_a_token>

Response:
{
  "chats": [
    {"id": "...", "user_id": "service-principal-id", ...},  // All users see same chats
    {"id": "...", "user_id": "service-principal-id", ...}
  ]
}
```

### After (Fixed)

```
GET /api/v1/chats
Authorization: Bearer <user_a_token>
x-forwarded-access-token: <user_a_obo_token>

Response:
{
  "chats": [
    {"id": "...", "user_id": "user-a-id", ...},  // Only User A's chats
  ]
}
```

## Existing Contracts (Unchanged)

The following API contracts remain unchanged:

- `GET /api/v1/chats` - List chats (now filtered by real user_id)
- `POST /api/v1/chats` - Create chat (now with real user_id)
- `GET /api/v1/chats/{chat_id}` - Get chat (ownership already enforced)
- `PATCH /api/v1/chats/{chat_id}` - Update chat (ownership already enforced)
- `DELETE /api/v1/chats/{chat_id}` - Delete chat (ownership already enforced)
- `GET /api/v1/chats/{chat_id}/messages` - List messages (ownership already enforced)
- `POST /api/v1/chats/{chat_id}/messages` - Send message (ownership already enforced)
- `POST /api/v1/research/{session_id}/cancel` - Cancel research (**now enforces ownership**)

## Client Compatibility

Frontend clients require **no changes for isolation**:
- The `x-forwarded-access-token` header is automatically added by Databricks Apps proxy
- Response formats remain identical
- Only the `user_id` values change to reflect real users

---

## Part 2: User Profile API (New)

### GET /api/v1/user/profile

Returns the current authenticated user's profile information.

**Request**:
```http
GET /api/v1/user/profile HTTP/1.1
```

**Response** (200 OK):
```json
{
  "user_id": "12345",
  "email": "john.doe@company.com",
  "display_name": "John Doe",
  "workspace": "demo-workspace"
}
```

**Response Schema** (Pydantic):
```python
class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    workspace: str | None
```

**TypeScript**:
```typescript
interface UserProfile {
  userId: string;
  email: string;
  displayName: string;
  workspace: string | null;
}
```

**Error Responses**:
- `401 Unauthorized` - No valid authentication
- `503 Service Unavailable` - Cannot resolve user identity

---

## Part 3: Incognito Chats API (New)

### POST /api/v1/chats/incognito

Creates a new incognito chat for the current session.

**Request**:
```http
POST /api/v1/chats/incognito HTTP/1.1
Content-Type: application/json
Cookie: incognito_session=<token>

{
  "title": "Optional initial title"
}
```

**Response** (201 Created):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": null,
  "status": "active",
  "chat_type": "incognito",
  "created_at": "2026-01-25T10:30:00Z",
  "updated_at": "2026-01-25T10:30:00Z",
  "message_count": 0
}
```

**Headers Set**:
```http
Set-Cookie: incognito_session=<token>; HttpOnly; SameSite=Strict; Max-Age=3600
```

**Error Responses**:
- `400 Bad Request` - Maximum 5 incognito chats reached
- `401 Unauthorized` - No valid authentication

---

### GET /api/v1/chats/incognito

Lists incognito chats for the current browser session.

**Request**:
```http
GET /api/v1/chats/incognito HTTP/1.1
Cookie: incognito_session=<token>
```

**Response** (200 OK):
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Research topic",
      "status": "active",
      "chat_type": "incognito",
      "created_at": "2026-01-25T10:30:00Z",
      "updated_at": "2026-01-25T10:35:00Z",
      "message_count": 3
    }
  ],
  "total": 1,
  "session_expires_at": "2026-01-25T11:35:00Z"
}
```

---

### POST /api/v1/chats/{chat_id}/convert

Converts an incognito chat to a permanent (regular) chat.

**Request**:
```http
POST /api/v1/chats/{chat_id}/convert HTTP/1.1
Cookie: incognito_session=<token>
```

**Response** (200 OK):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Research topic",
  "status": "active",
  "chat_type": "regular",
  "created_at": "2026-01-25T10:30:00Z",
  "updated_at": "2026-01-25T10:40:00Z",
  "message_count": 5
}
```

**Error Responses**:
- `404 Not Found` - Chat not found or not an incognito chat

---

### GET /api/v1/session/incognito

Returns incognito session status and remaining quota.

**Request**:
```http
GET /api/v1/session/incognito HTTP/1.1
Cookie: incognito_session=<token>
```

**Response** (200 OK):
```json
{
  "has_session": true,
  "chat_count": 2,
  "max_chats": 5,
  "expires_at": "2026-01-25T11:30:00Z"
}
```

---

## Modified Existing Endpoints

### GET /api/v1/chats

**New Query Parameter**: `chat_type`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_type` | string | `regular` | Filter: `regular`, `incognito`, `all` |

**Response** (updated schema):
```json
{
  "items": [
    {
      "id": "...",
      "title": "...",
      "status": "active",
      "chat_type": "regular",
      "created_at": "...",
      "updated_at": "...",
      "message_count": 5
    }
  ],
  "total": 1
}
```

---

### POST /api/v1/chats

**New Optional Field**: `incognito`

```json
{
  "title": "Optional title",
  "incognito": false
}
```

---

## TypeScript Types (Frontend)

```typescript
// frontend/src/types/index.ts

export type ChatType = 'regular' | 'incognito';

export interface Chat {
  id: string;
  title: string | null;
  status: 'active' | 'archived' | 'deleted';
  chatType: ChatType;  // NEW
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface UserProfile {
  userId: string;
  email: string;
  displayName: string;
  workspace: string | null;
}

export interface IncognitoSessionStatus {
  hasSession: boolean;
  chatCount: number;
  maxChats: number;
  expiresAt: string | null;
}
```

---

## Security Notes

1. **User Isolation**: 404 returned for both "not found" and "not authorized"
2. **Incognito Session Cookie**: `HttpOnly; SameSite=Strict; Max-Age=3600`
3. **Incognito Limit**: Server enforces max 5 concurrent per session
