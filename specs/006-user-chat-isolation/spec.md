# Feature Specification: User Chat Isolation

**Feature Branch**: `006-user-chat-isolation`
**Created**: 2026-01-25
**Status**: Draft
**Input**: User description: "multi-user chat feature. all users should be able to see/change own chats and have no access to other user's chats."
**Enhancement 1**: User profile UI - display authenticated user information with polished visual design
**Enhancement 2**: Incognito/temporary chats - ephemeral research sessions that auto-delete for enhanced privacy

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Private Chat Ownership (Priority: P1)

As a user of the deep research application, I want my chat conversations to be private so that only I can view and manage my own research chats, and other users cannot access my conversations.

**Why this priority**: This is the core privacy requirement. Without proper user identity isolation, the entire multi-user deployment is fundamentally broken from a privacy and security perspective.

**Independent Test**: Can be fully tested by having two different users log into the application and verifying each user only sees their own chats. Delivers the fundamental value of private conversations.

**Acceptance Scenarios**:

1. **Given** I am logged in as User A, **When** I create a new chat and send messages, **Then** the chat is associated with my unique user identity and only visible to me
2. **Given** User A has created chats, **When** User B logs in and views the chat list, **Then** User B sees only their own chats (not User A's chats)
3. **Given** I am logged in as User A, **When** I view my chat list, **Then** I see all chats I have created and none created by other users

---

### User Story 2 - Chat Access Control (Priority: P1)

As a user, I want to be prevented from accessing another user's chat even if I know the chat identifier, so that my conversations remain protected from unauthorized access.

**Why this priority**: This is a critical security requirement. Even if chat isolation works at the list level, direct URL access must also be protected.

**Independent Test**: Can be tested by attempting to access a known chat ID belonging to another user via direct URL or API call. Delivers security against unauthorized access attempts.

**Acceptance Scenarios**:

1. **Given** User A has a chat with ID "abc-123", **When** User B attempts to access chat "abc-123" via direct URL, **Then** User B receives an access denied response (shown as "not found" to prevent information leakage)
2. **Given** User A has a chat, **When** User B attempts to modify User A's chat via API, **Then** the modification is rejected
3. **Given** User A is viewing their own chat, **When** they attempt to edit or delete it, **Then** the operation succeeds

---

### User Story 3 - Research Session Protection (Priority: P1)

As a user, I want my in-progress research sessions to be protected so that other users cannot interfere with or cancel my ongoing research.

**Why this priority**: Research sessions can take several minutes and generate valuable results. Allowing others to cancel them would cause data loss and poor user experience.

**Independent Test**: Can be tested by having User A start a research session and User B attempting to cancel it. Delivers protection of in-progress work.

**Acceptance Scenarios**:

1. **Given** User A has a research session in progress, **When** User B attempts to cancel User A's research session, **Then** the cancellation is rejected
2. **Given** User A has a research session in progress, **When** User A attempts to cancel their own research, **Then** the cancellation succeeds
3. **Given** User A's research completes, **When** User B attempts to view the research results, **Then** User B cannot access the results

---

### User Story 4 - Seamless User Experience (Priority: P2)

As a user, I want the multi-user isolation to work transparently without requiring me to take any additional steps, so that I can focus on my research tasks.

**Why this priority**: Good UX requires that security features work invisibly. Users shouldn't need to configure or manage access controls.

**Independent Test**: Can be tested by observing a new user's first interaction with the system. Delivers frictionless onboarding.

**Acceptance Scenarios**:

1. **Given** I am a new user logging in for the first time, **When** I access the application, **Then** I see an empty chat list ready for me to start my first conversation
2. **Given** I am an existing user, **When** I log in from a different device or browser, **Then** I see all my existing chats
3. **Given** the system experiences a temporary issue resolving my identity, **When** I access the application, **Then** I see a clear message explaining authentication is required (not another user's data)

---

### User Story 5 - User Profile Display (Priority: P2)

As a user, I want to see my identity displayed in the application interface so that I have visual confirmation of who I am logged in as, building trust and confidence in the privacy isolation.

**Why this priority**: Visual identity confirmation reinforces the multi-user isolation feature. Users gain confidence that they're seeing their own data when they can see their name/avatar. This is a P2 because the core isolation (P1) must work first, but this significantly enhances user experience.

**Independent Test**: Can be tested by logging in as a user and verifying their profile information is displayed correctly. Delivers immediate visual feedback of identity.

**Acceptance Scenarios**:

1. **Given** I am logged in as a user, **When** I view the application, **Then** I see my profile information displayed prominently in the sidebar
2. **Given** I am logged in as a user with an email address, **When** I view my profile display, **Then** I see an avatar generated from my name or email (initials-based or gravatar-style)
3. **Given** I am logged in as a user, **When** I view my profile display, **Then** I see my display name (or email if no display name is available)
4. **Given** I am logged in as a user, **When** I hover over or click my profile area, **Then** I can access additional profile details via a dropdown or popover
5. **Given** I am viewing my profile dropdown, **When** I look at the options, **Then** I see my full email address and an option to understand my workspace connection

---

### User Story 6 - Polished Visual Design (Priority: P3)

As a user, I want the user profile UI to have a polished, professional appearance that matches the overall application design aesthetic, creating a cohesive and trustworthy experience.

**Why this priority**: While functional profile display (P2) comes first, visual polish significantly impacts perceived quality and user trust. This story focuses on the aesthetic refinement.

**Independent Test**: Can be visually reviewed against design criteria and compared with professional application standards. Delivers professional appearance.

**Acceptance Scenarios**:

1. **Given** the profile component is displayed, **When** I view it, **Then** it uses consistent styling with the rest of the application (matching colors, typography, spacing)
2. **Given** the profile avatar is displayed, **When** I view it, **Then** it has a visually appealing shape (circular), appropriate size, and smooth rendering
3. **Given** the profile avatar uses initials, **When** different users are logged in, **Then** the avatar background color varies based on the user's identity (deterministic color generation)
4. **Given** the profile dropdown is open, **When** I view it, **Then** it has smooth animations, appropriate shadows, and clear visual hierarchy
5. **Given** I interact with the profile area, **When** I hover or click, **Then** there are subtle visual feedback effects (hover states, transitions)

---

### User Story 7 - Incognito Chat Creation (Priority: P2)

As a user, I want to start a temporary/incognito chat session so that I can conduct sensitive research without it being saved to my permanent chat history.

**Why this priority**: Privacy-conscious users need a way to research sensitive topics without permanent records. This complements the user isolation feature by giving users control over their own data retention. P2 because core isolation (P1) must work first.

**Independent Test**: Can be tested by creating an incognito chat, conducting research, then closing/refreshing the browser and verifying the chat is no longer accessible. Delivers privacy control to users.

**Acceptance Scenarios**:

1. **Given** I am logged in and viewing my chat list, **When** I look for the option to create a new chat, **Then** I see a clearly labeled option to start an incognito/temporary chat (e.g., "New Incognito Chat" with a distinctive icon)
2. **Given** I click to create an incognito chat, **When** the chat opens, **Then** I see a clear visual indicator that this is a temporary session (e.g., incognito icon in header, subtle background tint, or banner)
3. **Given** I am in an incognito chat, **When** I send a message and receive a research response, **Then** the chat functions identically to a regular chat in terms of research quality
4. **Given** I have an incognito chat open, **When** I close the browser tab or window, **Then** the chat and all its contents are permanently deleted
5. **Given** I have an incognito chat open, **When** I navigate away from it to another chat and return, **Then** the incognito chat is still available during my active session

---

### User Story 8 - Incognito Chat Visibility and Management (Priority: P2)

As a user, I want incognito chats to be visually distinct and separated from my regular chats so that I can easily identify which conversations are temporary.

**Why this priority**: Users need clear visual differentiation to avoid confusion about which chats will persist. This prevents the frustrating scenario of losing important research that was accidentally started in incognito mode.

**Independent Test**: Can be tested by having both regular and incognito chats active and verifying they are visually distinct. Delivers clarity and prevents user errors.

**Acceptance Scenarios**:

1. **Given** I have both regular and incognito chats, **When** I view my chat list, **Then** incognito chats appear in a separate section or with a distinct visual treatment (icon, color, or grouping)
2. **Given** I am viewing an incognito chat, **When** I look at the chat header or title area, **Then** there is a persistent visual indicator (incognito icon or label) that cannot be missed
3. **Given** I am in an incognito chat, **When** I want to save this conversation permanently, **Then** I see an option to "Keep this chat" or "Save to history" that converts it to a regular chat
4. **Given** I accidentally started important research in incognito mode, **When** I click "Keep this chat", **Then** the chat is converted to a regular chat and appears in my normal chat history
5. **Given** I have multiple incognito chats open, **When** I close my browser session, **Then** all incognito chats are deleted but my regular chats remain intact

---

### User Story 9 - Incognito Mode Visual Polish (Priority: P3)

As a user, I want the incognito chat experience to have a polished, intuitive design that clearly communicates the temporary nature of the session without being intrusive.

**Why this priority**: Visual polish for incognito mode ensures users understand and trust the privacy feature. Poor design could lead to confusion or mistrust. P3 because functional incognito (P2) must work first.

**Independent Test**: Can be visually reviewed for clarity, consistency, and user understanding through usability testing. Delivers professional, trustworthy appearance.

**Acceptance Scenarios**:

1. **Given** I view the incognito chat creation button, **When** I examine it, **Then** it has an appropriate icon (e.g., masked face, private browsing icon, or eye-with-slash) that universally conveys privacy
2. **Given** I am in an incognito chat, **When** I view the overall appearance, **Then** there is a subtle but clear visual distinction (e.g., slightly different header color, incognito badge) that doesn't distract from the content
3. **Given** the incognito visual treatment is applied, **When** compared to the regular chat, **Then** the distinction is obvious within 2 seconds of viewing
4. **Given** I hover over the incognito indicator, **When** I view the tooltip or info, **Then** I see a brief explanation of what incognito mode means ("This chat will be deleted when you close it")
5. **Given** I am about to close an incognito chat with content, **When** I initiate close, **Then** I see a brief confirmation that acknowledges the data will be lost (non-blocking, can be dismissed quickly)

---

### Edge Cases

- What happens when a user's authentication token expires mid-session?
  - System should gracefully handle re-authentication without data loss
- What happens if user identity cannot be determined?
  - System should deny access with a clear error message, never fall back to showing shared data
- What happens to existing chats created before user isolation was implemented?
  - Not applicable: no production data exists prior to this feature
- What happens if a user attempts to access a chat that doesn't exist?
  - System returns "not found" (same response as unauthorized access to prevent enumeration attacks)
- What happens if the user's display name is unavailable?
  - Profile displays email address as fallback, or username portion of email if more concise
- What happens if both display name and email are unavailable?
  - Profile displays a generic identifier ("User") with the avatar showing a default icon
- What happens if the user identity API call fails while loading the profile?
  - Profile shows a loading state initially, then falls back to cached identity or shows "Signed in" without details
- What happens on narrow/mobile viewports?
  - Profile component adapts gracefully: avatar-only view with dropdown for details on constrained widths

#### Incognito Chat Edge Cases

- What happens if a user's session expires while they have an incognito chat with research in progress?
  - The research continues to completion, but the incognito chat is deleted when the session token is invalidated (data loss is expected behavior for incognito)
- What happens if the user refreshes the page during an active incognito session?
  - Incognito chats persist during page refresh within the same browser session; they are only deleted on tab/window close or session end
- What happens if an incognito chat's research takes longer than expected (several minutes)?
  - The chat remains active and accessible throughout the research duration; timer-based deletion (if any) only starts after research completion
- What happens if a user tries to bookmark or share an incognito chat URL?
  - Incognito chats have non-persistent URLs; accessing a deleted incognito chat URL returns "not found"
- What happens if a user has multiple tabs with the same incognito chat open?
  - Incognito chat is shared across tabs in the same browser session; closing one tab doesn't delete it until all tabs with that session are closed
- What happens if storage/memory limits are approached with many incognito chats?
  - System should enforce a reasonable limit (e.g., max 5 concurrent incognito chats) and prompt the user to close some if exceeded
- What happens when a user converts an incognito chat to permanent?
  - The chat ID may change (to a permanent one), requiring a redirect; all content and history is preserved in the conversion

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST identify each user by their unique identity from the authentication provider (not by a shared service account)
- **FR-002**: System MUST associate every new chat with the creating user's unique identifier
- **FR-003**: System MUST filter chat listings to show only chats owned by the requesting user
- **FR-004**: System MUST verify user ownership before allowing read access to any chat
- **FR-005**: System MUST verify user ownership before allowing modification of any chat (update, delete, archive)
- **FR-006**: System MUST verify user ownership before allowing cancellation of research sessions
- **FR-007**: System MUST verify user ownership before allowing access to research results or events
- **FR-008**: System MUST return identical error responses for "not found" and "access denied" scenarios to prevent information leakage
- **FR-009**: System MUST reject requests when user identity cannot be determined (no fallback to anonymous or shared identity in production)
- **FR-010**: System MUST log authentication events including user identity resolution method for audit purposes

#### User Profile Display Requirements

- **FR-011**: System MUST provide an API endpoint to retrieve the current user's profile information (display name, email, user identifier)
- **FR-012**: Application MUST display the authenticated user's identity in the sidebar area, visible at all times during normal use
- **FR-013**: Profile display MUST include a visual avatar representation derived from the user's identity (initials-based with deterministic color)
- **FR-014**: Profile display MUST include the user's display name, with email as fallback if display name is unavailable
- **FR-015**: Profile component MUST provide expandable details (dropdown/popover) showing full email and workspace information
- **FR-016**: Profile avatar colors MUST be deterministically generated from user identity to ensure consistency across sessions
- **FR-017**: Profile component MUST gracefully handle loading and error states without disrupting the application layout

#### Incognito/Temporary Chat Requirements

- **FR-018**: System MUST provide a user-accessible option to create an incognito/temporary chat distinct from regular chat creation
- **FR-019**: Incognito chats MUST be clearly visually distinguished from regular chats in both the chat list and within the chat view
- **FR-020**: Incognito chats MUST NOT appear in the user's persistent chat history after the browser session ends
- **FR-021**: Incognito chats MUST be automatically deleted when the user's browser session terminates (tab close, window close, or 1-hour idle timeout)
- **FR-022**: Incognito chats MUST function identically to regular chats for research operations (same quality, same features)
- **FR-023**: System MUST provide an option to convert an incognito chat to a permanent chat (save to history)
- **FR-024**: Incognito chats MUST be stored in temporary/session-scoped storage rather than permanent database storage
- **FR-025**: System MUST display a clear, persistent visual indicator within incognito chats showing their temporary nature
- **FR-026**: System MUST limit the maximum number of concurrent incognito chats per user to prevent resource abuse (default: 5)
- **FR-027**: Incognito chat URLs MUST NOT be shareable or bookmarkable across sessions (URLs become invalid after deletion)
- **FR-028**: System SHOULD display a tooltip or brief explanation of incognito mode when users hover over the incognito indicator
- **FR-029**: System SHOULD show a non-blocking confirmation when closing an incognito chat with content, acknowledging data loss

### Key Entities

- **User Identity**: Represents an authenticated user with unique identifier, email, and display name. Source of truth for ownership.
- **Chat**: A conversation container owned by exactly one user. Contains user_id foreign key linking to owner. Has a `chat_type` attribute distinguishing regular from incognito chats.
- **Incognito Chat**: A special type of chat with ephemeral storage; associated with the user's session rather than permanent storage. Automatically deleted on session termination.
- **Message**: Belongs to a chat; inherits ownership from parent chat (no direct user_id needed).
- **Research Session**: Associated with a message/chat; ownership verified through chat relationship.
- **Browser Session**: Represents the user's active browser session; used to scope incognito chat lifetime. Terminates on tab/window close or session timeout.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of newly created chats are associated with the correct individual user identity (verified by user_id matching authenticated user, not service principal)
- **SC-002**: Users can only see their own chats when viewing the chat list (zero cross-user chat visibility)
- **SC-003**: Direct access attempts to another user's chat return "not found" 100% of the time
- **SC-004**: Research session cancellation by unauthorized users fails 100% of the time
- **SC-005**: Authentication identity resolution adds less than 200ms latency to request handling
- **SC-006**: System correctly identifies users within 2 seconds of login across browser refresh or reconnection scenarios

#### User Profile Display Metrics

- **SC-007**: User profile information is displayed within 1 second of application load
- **SC-008**: 100% of users can identify which account they are logged in as by viewing the UI
- **SC-009**: Profile dropdown/popover opens within 200ms of user interaction
- **SC-010**: Profile component renders consistently across modern browsers (Chrome, Firefox, Safari, Edge)
- **SC-011**: Profile avatar colors are consistent for the same user across sessions and page refreshes

#### Incognito Chat Metrics

- **SC-012**: Incognito chat creation is available within 1 click/tap from the main chat interface
- **SC-013**: Users can distinguish between regular and incognito chats within 2 seconds of viewing the chat list
- **SC-014**: 100% of incognito chats are deleted within 30 seconds of session termination
- **SC-015**: Incognito chats provide identical research quality compared to regular chats (same response times, same depth)
- **SC-016**: Incognito-to-permanent conversion preserves 100% of chat content and history
- **SC-017**: Incognito indicator tooltip/explanation is visible within 500ms of hovering
- **SC-018**: Maximum 5 concurrent incognito chats are enforced per user session

## Clarifications

### Session 2026-01-25

- Q: How should legacy chats (pre-isolation) be handled? → A: Not applicable - no production data exists yet
- Q: Where should incognito chat data primarily reside during the session? → A: Server-side session store (backend manages ephemeral storage, cleaned on session end)
- Q: Should admins/support have any ability to view user chats for troubleshooting? → A: No admin concept exists; all users have equal access (own chats only)
- Q: How long should an idle session persist before incognito chats are cleaned up? → A: 1 hour idle timeout
- Q: How should concurrent sessions from the same user be handled? → A: Independent sessions; regular chats sync across devices, incognito chats are session-specific

## Assumptions

- The deployment environment (Databricks Apps) provides the authenticated user's identity via HTTP headers
- Users are already authenticated by the Databricks workspace before accessing the application
- The authentication mechanism provides a stable, unique user identifier that persists across sessions
- Service principal credentials remain available for backend operations (LLM calls, search) that don't require user context
- No production data exists prior to this feature (no migration concerns)
- No admin or privileged user roles exist; all authenticated users have equal access rights (own chats only)

### User Profile Display Assumptions

- The backend can extract user display name and email from the authentication headers or token
- Users accessing via Databricks Apps will have at minimum an email address available
- Avatar generation will be client-side based on user data (no external avatar service dependency)
- The initials-based avatar is acceptable; external avatar services (Gravatar) are not required
- Profile component placement is in the sidebar footer area, consistent with modern application patterns

### Incognito/Temporary Chat Assumptions

- Browser session termination can be reliably detected (via beforeunload, visibilitychange, or session heartbeat)
- Incognito chat data is stored in server-side session store; backend manages ephemeral storage and cleans up on session end
- Users understand the concept of "incognito" from browser incognito mode; no extensive education needed
- Incognito chats do not sync across devices/sessions (by definition, they are session-specific); regular chats sync across all user sessions
- The 5-chat limit is a reasonable default; this can be configurable in future if needed
- Research in incognito chats still uses the same LLM and search infrastructure; only storage differs
- Incognito chat conversion to permanent is a one-time, irreversible operation (chat becomes permanent)
- No "incognito-only" settings or preferences are needed; incognito is purely about data retention
