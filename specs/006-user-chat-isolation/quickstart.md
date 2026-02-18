# Quickstart: User Chat Isolation Implementation

**Feature**: 006-user-chat-isolation
**Date**: 2026-01-25 (Updated)

## Implementation Scope

| Component | Estimated Effort |
|-----------|------------------|
| Core Isolation (P1) | ~60 lines, 3 files |
| User Profile UI (P2) | ~200 lines, 5 files |
| Incognito Chats (P2) | ~500 lines, 10 files |
| Total | ~760 lines, 18 files |

## Prerequisites

- Python 3.11+
- Access to Databricks workspace
- Local development environment set up (`make install`)

## Implementation Steps

### Step 1: Add User WorkspaceClient Function

**File**: `src/deep_research/core/auth.py`

Add this function after `get_workspace_client()` (around line 65):

```python
def get_user_workspace_client(token: str) -> WorkspaceClient:
    """Create WorkspaceClient using user's OAuth token for identity resolution.

    Used to resolve the actual end user's identity from the x-forwarded-access-token
    header in Databricks Apps deployments. This client should ONLY be used for
    identity resolution - backend operations use the service principal client.

    Args:
        token: User's OAuth access token from x-forwarded-access-token header.

    Returns:
        WorkspaceClient configured with the user's token.
    """
    settings = get_settings()
    host = settings.databricks_host
    if not host:
        # Derive host from existing service principal client
        sp_client = get_workspace_client()
        host = sp_client.config.host

    return WorkspaceClient(host=host, token=token)
```

Also update `extract_obo_token()` docstring - remove the `DEPRECATED` notice (lines 85-100):

```python
def extract_obo_token(headers: dict[str, str]) -> str | None:
    """Extract OBO (On-Behalf-Of) token from request headers.

    In Databricks Apps, the user's OAuth token is forwarded as
    'x-forwarded-access-token' header for user identity resolution.

    Args:
        headers: Request headers dictionary (keys should be lowercase).

    Returns:
        OAuth token if present, None otherwise.
    """
    return headers.get("x-forwarded-access-token")
```

---

### Step 2: Update Auth Middleware

**File**: `src/deep_research/middleware/auth.py`

Modify `get_current_user_identity()` to prioritize OBO token. Replace the function body (lines 18-67):

```python
async def get_current_user_identity(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserIdentity:
    """FastAPI dependency to get current user identity.

    Priority order:
    1. OBO token from x-forwarded-access-token (actual user in Databricks Apps)
    2. Service principal auth (fallback for local development)
    3. Anonymous (development mode only)

    Args:
        request: FastAPI request object.
        settings: Application settings.

    Returns:
        UserIdentity of the authenticated user.

    Raises:
        HTTPException: If all authentication methods fail in production.
    """
    from deep_research.core.auth import extract_obo_token, get_user_workspace_client

    # Priority 1: OBO token (actual user in Databricks Apps)
    obo_token = extract_obo_token(dict(request.headers))
    if obo_token:
        try:
            user_client = get_user_workspace_client(obo_token)
            current_user = user_client.current_user.me()
            user = UserIdentity.from_workspace_user(current_user)

            # Keep service principal client for backend operations
            sp_client = get_workspace_client()
            request.state.user = user
            request.state.workspace_client = sp_client

            logger.info(f"OBO auth successful: user={user.email}, id={user.user_id}")
            return user

        except Exception as e:
            logger.warning(f"OBO auth failed, falling back to SP: {e}")

    # Priority 2: Service principal auth (existing logic)
    try:
        client = get_workspace_client()
        user = get_current_user(client)

        request.state.user = user
        request.state.workspace_client = client

        logger.debug(f"Service principal auth successful: user={user.email}")
        return user

    except Exception as e:
        logger.warning(f"Service principal auth failed: {e}")

    # Priority 3: Anonymous (development mode only)
    if not settings.is_production:
        user = UserIdentity.anonymous()
        request.state.user = user
        logger.debug("Using anonymous user (development mode)")
        return user

    # All methods failed in production
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication failed",
        headers={"WWW-Authenticate": "Bearer"},
    )
```

---

### Step 3: Fix Research Session Security

**File**: `src/deep_research/api/v1/research.py`

Modify `cancel_research()` endpoint to add ownership verification (lines 34-64):

```python
@router.post("/{session_id}/cancel", response_model=CancelResearchResponse)
async def cancel_research(
    session_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CancelResearchResponse:
    """Cancel in-progress research.

    Stops the research operation within 2 seconds. Partial results are preserved.
    Only the owner of the research session can cancel it.
    """
    from deep_research.models.message import Message

    service = ResearchSessionService(db)

    # Get session first to check if it exists
    session = await service.get(session_id)
    if not session:
        raise NotFoundError("ResearchSession", str(session_id))

    # SECURITY: Verify ownership via message -> chat -> user_id
    message = await db.get(Message, session.message_id)
    if not message:
        raise NotFoundError("ResearchSession", str(session_id))

    chat_service = ChatService(db)
    chat = await chat_service.get_by_id(message.chat_id)
    if not chat or chat.user_id != user.user_id:
        # Return 404 instead of 403 to prevent information leakage
        raise NotFoundError("ResearchSession", str(session_id))

    # Proceed with cancellation (existing logic)
    session = await service.cancel(session_id)

    # Get partial results if available
    partial_results = None
    if session and session.observations:
        partial_results = "\n\n".join(
            obs.get("observation", "") for obs in session.observations if obs.get("observation")
        )

    await db.commit()

    return CancelResearchResponse(
        session_id=session_id,
        status="cancelled",
        partial_results=partial_results if partial_results else None,
    )
```

---

## Verification

### 1. Run Type Checks

```bash
make typecheck
```

Expected: No new type errors

### 2. Run Unit Tests

```bash
make test
```

Expected: All existing tests pass

### 3. Run Integration Tests

```bash
make test-integration
```

Expected: Auth flow tests pass

### 4. Manual Testing

```bash
# Deploy to dev
make deploy TARGET=dev

# Check logs for OBO auth
make logs TARGET=dev SEARCH="--search 'OBO auth'"
```

Expected: Logs show "OBO auth successful" with real user email

### 5. Multi-User Test

1. Open app as User A → Create a chat
2. Open app as User B (different browser/incognito)
3. User B should NOT see User A's chat
4. User B attempting to access User A's chat URL should get 404

---

## Rollback

If issues occur, revert the 3 files:

```bash
git checkout HEAD~1 -- src/deep_research/core/auth.py
git checkout HEAD~1 -- src/deep_research/middleware/auth.py
git checkout HEAD~1 -- src/deep_research/api/v1/research.py
```

The system will fall back to service principal authentication (shared identity).

---

## Part 2: User Profile UI

### Step 4: Add User Profile Endpoint

**File**: `src/deep_research/api/v1/user.py` (NEW)

```python
"""User profile API endpoints."""
from fastapi import APIRouter

from deep_research.middleware.auth import CurrentUser
from deep_research.schemas.user import UserProfileResponse

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(user: CurrentUser) -> UserProfileResponse:
    """Get current user's profile information."""
    return UserProfileResponse(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        workspace=None,  # Can be extracted from host if needed
    )
```

**File**: `src/deep_research/schemas/user.py` (NEW)

```python
"""User-related Pydantic schemas."""
from pydantic import BaseModel


class UserProfileResponse(BaseModel):
    """User profile information."""

    user_id: str
    email: str
    display_name: str
    workspace: str | None
```

Register router in `src/deep_research/api/v1/__init__.py`:

```python
from deep_research.api.v1.user import router as user_router

api_v1_router.include_router(user_router)
```

---

### Step 5: Add Frontend User Profile Component

**File**: `frontend/src/components/user/UserAvatar.tsx` (NEW)

```tsx
import React from 'react';

const AVATAR_COLORS = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
  '#8B5CF6', '#EC4899', '#06B6D4', '#F97316',
];

function getAvatarColor(userId: string): string {
  const hash = userId.split('').reduce(
    (acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0
  );
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

function getInitials(displayName: string, email: string): string {
  if (displayName) {
    const parts = displayName.split(' ');
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return displayName.substring(0, 2).toUpperCase();
  }
  return email.substring(0, 2).toUpperCase();
}

interface UserAvatarProps {
  userId: string;
  displayName: string;
  email: string;
  size?: 'sm' | 'md' | 'lg';
}

export function UserAvatar({ userId, displayName, email, size = 'md' }: UserAvatarProps) {
  const color = getAvatarColor(userId);
  const initials = getInitials(displayName, email);

  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm',
    lg: 'w-10 h-10 text-base',
  };

  return (
    <div
      className={`${sizeClasses[size]} rounded-full flex items-center justify-center font-medium text-white`}
      style={{ backgroundColor: color }}
    >
      {initials}
    </div>
  );
}
```

**File**: `frontend/src/components/user/UserProfile.tsx` (NEW)

```tsx
import React, { useState } from 'react';
import { useUserProfile } from '@/hooks/useUserProfile';
import { UserAvatar } from './UserAvatar';
import { ChevronDown, ChevronUp } from 'lucide-react';

export function UserProfile() {
  const { data: profile, isLoading, error } = useUserProfile();
  const [isOpen, setIsOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="p-3 border-t flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-muted animate-pulse" />
        <div className="flex-1">
          <div className="h-4 bg-muted rounded animate-pulse w-20" />
        </div>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="p-3 border-t text-sm text-muted-foreground">
        Signed in
      </div>
    );
  }

  return (
    <div className="border-t">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-3 flex items-center gap-3 hover:bg-accent transition-colors"
      >
        <UserAvatar
          userId={profile.userId}
          displayName={profile.displayName}
          email={profile.email}
        />
        <div className="flex-1 text-left min-w-0">
          <p className="text-sm font-medium truncate">{profile.displayName}</p>
        </div>
        {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>

      {isOpen && (
        <div className="px-3 pb-3 text-sm text-muted-foreground">
          <p className="truncate">{profile.email}</p>
          {profile.workspace && (
            <p className="text-xs mt-1">Workspace: {profile.workspace}</p>
          )}
        </div>
      )}
    </div>
  );
}
```

**File**: `frontend/src/hooks/useUserProfile.ts` (NEW)

```tsx
import { useQuery } from '@tanstack/react-query';
import { userApi } from '@/api/client';
import type { UserProfile } from '@/types';

export function useUserProfile() {
  return useQuery<UserProfile>({
    queryKey: ['user', 'profile'],
    queryFn: userApi.getProfile,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: Infinity,
  });
}
```

---

## Part 3: Incognito Chats

### Step 6: Database Migration

**File**: `src/deep_research/db/migrations/versions/013_incognito_support.py` (NEW)

```bash
# Generate migration
cd src/deep_research/db/migrations
alembic revision -m "add_incognito_support"
```

Then edit the generated file with the migration from `data-model.md`.

---

### Step 7: Add Incognito Session Model

**File**: `src/deep_research/models/incognito_session.py` (NEW)

```python
"""Incognito session model for ephemeral chat storage."""
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.models.base import BaseModel


class IncognitoSession(BaseModel):
    """Server-side session for incognito chat lifecycle."""

    __tablename__ = "incognito_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    session_token: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )

    # Relationship to chats
    chats: Mapped[list["Chat"]] = relationship(back_populates="incognito_session")

    def touch(self) -> None:
        """Update last_activity and extend expiration."""
        self.last_activity = datetime.utcnow()
        self.expires_at = self.last_activity + timedelta(hours=1)
```

---

### Step 8: Update Chat Model

**File**: `src/deep_research/models/chat.py`

Add to existing model:

```python
from enum import Enum
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import ENUM as SQLAlchemyEnum


class ChatType(str, Enum):
    REGULAR = "regular"
    INCOGNITO = "incognito"


class Chat(BaseModel):
    # ... existing fields ...

    # NEW: Incognito support
    chat_type: Mapped[ChatType] = mapped_column(
        SQLAlchemyEnum(ChatType, name="chattype"),
        default=ChatType.REGULAR,
        index=True,
    )
    incognito_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("incognito_sessions.id", ondelete="CASCADE"),
        nullable=True,
    )
    incognito_session: Mapped["IncognitoSession | None"] = relationship(
        back_populates="chats"
    )
```

---

### Step 9: Add Frontend Incognito Components

**File**: `frontend/src/components/incognito/IncognitoIndicator.tsx` (NEW)

```tsx
import { EyeOff } from 'lucide-react';

interface IncognitoIndicatorProps {
  showTooltip?: boolean;
}

export function IncognitoIndicator({ showTooltip = true }: IncognitoIndicatorProps) {
  return (
    <div
      className="flex items-center gap-1 text-muted-foreground"
      title={showTooltip ? "This chat will be deleted when you close it" : undefined}
    >
      <EyeOff className="h-4 w-4" />
      <span className="text-xs">Incognito</span>
    </div>
  );
}
```

---

## Verification Checklist

### Core Isolation
- [ ] OBO auth logs show real user email
- [ ] User A cannot see User B's chats
- [ ] Direct URL access to other user's chat returns 404
- [ ] Research cancellation respects ownership

### User Profile
- [ ] Profile displays in sidebar footer
- [ ] Avatar shows correct initials
- [ ] Avatar color is consistent across refreshes
- [ ] Dropdown expands/collapses smoothly

### Incognito Chats
- [ ] Can create incognito chat
- [ ] Incognito chats show distinct visual treatment
- [ ] Closing browser deletes incognito chats
- [ ] Can convert incognito to regular chat
- [ ] Maximum 5 incognito chats enforced
- [ ] 1-hour idle timeout works

---

## Rollback

### Full Rollback
```bash
git checkout main -- src/deep_research/
git checkout main -- frontend/src/

# Rollback migration
cd src/deep_research/db/migrations
alembic downgrade -1
```

### Partial Rollback (keep isolation, remove incognito)
```bash
# Revert incognito-specific files only
git checkout HEAD~1 -- src/deep_research/models/incognito_session.py
git checkout HEAD~1 -- frontend/src/components/incognito/

# Downgrade migration
alembic downgrade 012
```
