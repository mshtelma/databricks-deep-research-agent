# Research: Data Source Selection Integration

**Feature**: 008-data-source-selection
**Date**: 2026-02-05
**Status**: Complete

## Research Tasks Completed

### 1. Source Scope Field Propagation (P0)

**Question**: Do existing schema fields work end-to-end from OrchestrationConfig to planner/researcher?

**Findings**:

1. **OrchestrationConfig** (`src/deep_research/agent/orchestrator.py:137-205`) already has:
   ```python
   source_scope: str | None = field(default=None)
   enabled_sources: list[str] | None = field(default=None)
   disabled_sources: list[str] | None = field(default=None)
   ```

2. **ResearchRequest** (`src/deep_research/schemas/research_request.py:34-53`) already has:
   ```python
   source_scope: SourceScope | None = None
   enabled_sources: list[str] | None = None
   disabled_sources: list[str] = []
   ```

3. **SourceScope enum** (`src/deep_research/schemas/source_scope.py`) defines:
   ```python
   class SourceScope(str, Enum):
       ENTERPRISE_ONLY = "enterprise_only"
       WEB_ONLY = "web_only"
       ALL = "all"
   ```

**Decision**: The schema layer is complete. The gap is purely in the API/service layer threading these values from HTTP request to OrchestrationConfig.

**Alternatives Considered**: None needed - schema design is sound.

---

### 2. Frontend State Persistence (P1)

**Question**: Best practices for localStorage with TypeScript in this codebase?

**Findings**:

1. **Existing pattern found** in `frontend/src/hooks/useQueryMode.ts`:
   ```typescript
   const QUERY_MODE_KEY = 'queryMode'

   export function useQueryMode() {
     const [queryMode, setQueryModeState] = useState<QueryMode>(() => {
       const stored = localStorage.getItem(QUERY_MODE_KEY)
       return (stored as QueryMode) || 'deep_research'
     })

     const setQueryMode = useCallback((mode: QueryMode) => {
       localStorage.setItem(QUERY_MODE_KEY, mode)
       setQueryModeState(mode)
     }, [])

     return { queryMode, setQueryMode }
   }
   ```

2. **Pattern characteristics**:
   - Lazy initialization in useState
   - Type assertion with fallback default
   - Synchronous save on change (no debounce needed for small data)
   - Single localStorage key per preference

**Decision**: Follow the `useQueryMode` pattern for `useSourceScope`. Store as JSON to handle the object structure (scope + enabled/disabled lists).

**Implementation**:
```typescript
// frontend/src/hooks/useSourceScope.ts
const SOURCE_SCOPE_KEY = 'deep-research-source-scope'

interface SourceScopePreference {
  scope: SourceScope
  enabledSources: string[]
  disabledSources: string[]
}

const DEFAULT_PREFERENCE: SourceScopePreference = {
  scope: 'all',
  enabledSources: [],
  disabledSources: [],
}

export function useSourceScope() {
  const [preference, setPreferenceState] = useState<SourceScopePreference>(() => {
    try {
      const stored = localStorage.getItem(SOURCE_SCOPE_KEY)
      return stored ? JSON.parse(stored) : DEFAULT_PREFERENCE
    } catch {
      return DEFAULT_PREFERENCE
    }
  })

  const setPreference = useCallback((pref: SourceScopePreference) => {
    localStorage.setItem(SOURCE_SCOPE_KEY, JSON.stringify(pref))
    setPreferenceState(pref)
  }, [])

  return { preference, setPreference }
}
```

**Alternatives Considered**:
- React Context: Overkill for simple preference storage
- TanStack Query mutation: Not suitable for localStorage (no server sync)
- IndexedDB: Overkill for ~1KB of data

---

### 3. API Backward Compatibility (P1)

**Question**: Will adding optional fields to SubmitJobRequest break existing clients?

**Findings**:

1. **Pydantic default values**: All new fields have defaults
   ```python
   source_scope: SourceScope | None = None  # Optional
   enabled_sources: list[str] | None = None  # Optional
   disabled_sources: list[str] = []  # Defaults to empty list
   ```

2. **JSON deserialization behavior**: Missing keys in request body use defaults

3. **OpenAPI schema**: Optional fields show as non-required in Swagger UI

4. **Existing pattern**: `output_type` field follows same pattern:
   ```python
   output_type: str | None = Field(
       default=None,
       description="Output format type"
   )
   ```

**Decision**: Safe to add fields. All fields optional with sensible defaults. Existing clients continue to work unchanged - backend treats missing fields as "use defaults".

**Risk mitigation**:
- Default `source_scope=None` means "use system default" (equivalent to "all")
- Empty `disabled_sources=[]` means "don't exclude any sources"
- `enabled_sources=None` means "no whitelist, use all available"

---

### 4. Frontend/Backend Enum Alignment (P0)

**Question**: Do frontend and backend SourceScope values match?

**Findings**:

**Backend** (`src/deep_research/schemas/source_scope.py`):
```python
class SourceScope(str, Enum):
    ENTERPRISE_ONLY = "enterprise_only"
    WEB_ONLY = "web_only"
    ALL = "all"
```

**Frontend** (`frontend/src/types/dataSources.ts`):
```typescript
export type SourceScope = 'enterprise_only' | 'web_only' | 'all'
```

**Decision**: Values match exactly. No conversion layer needed. Frontend can send string literals directly in JSON.

---

### 5. SourceScopeSelector Props Analysis (P1)

**Question**: What props does the existing SourceScopeSelector expect?

**Findings** (`frontend/src/components/research/SourceScopeSelector.tsx`):

```typescript
interface SourceScopeSelectorProps {
  selectedScope: SourceScope
  onScopeChange: (scope: SourceScope) => void
  availableSources?: AvailableSource[]
  onSourceToggle?: (sourceId: string, enabled: boolean) => void
  disabled?: boolean
  compact?: boolean
  className?: string
}
```

**Integration notes**:
1. `availableSources` comes from `useDiscoveredSources()` hook
2. Need to transform `DiscoveredSource[]` to `AvailableSource[]` (may already be compatible)
3. `compact={true}` recommended for MessageInput (smaller UI footprint)

---

## Summary of Decisions

| Item | Decision | Rationale |
|------|----------|-----------|
| Schema layer | No changes needed | Already complete in Feature 007 |
| API fields | Optional with defaults | Backward compatible, safe rollout |
| localStorage | Follow useQueryMode pattern | Proven pattern in codebase |
| Enum values | Direct string match | No conversion needed |
| SourceScopeSelector | Use compact mode | Better fit in MessageInput |

## Open Items

None - all research questions resolved.

## Files Referenced

- `src/deep_research/agent/orchestrator.py` - OrchestrationConfig dataclass
- `src/deep_research/schemas/source_scope.py` - SourceScope enum, SourceScopeConfig
- `src/deep_research/schemas/research_request.py` - ResearchRequest model
- `src/deep_research/api/v1/jobs.py` - SubmitJobRequest model
- `src/deep_research/services/job_manager.py` - submit_job(), _run_job() methods
- `frontend/src/hooks/useQueryMode.ts` - localStorage pattern reference
- `frontend/src/hooks/useDiscoveredSources.ts` - Discovery hooks
- `frontend/src/components/research/SourceScopeSelector.tsx` - UI component
- `frontend/src/types/dataSources.ts` - Frontend types
