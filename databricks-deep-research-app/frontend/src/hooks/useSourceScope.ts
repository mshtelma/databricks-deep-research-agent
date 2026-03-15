/**
 * useSourceScope - Hook for managing source scope selection with localStorage persistence.
 *
 * Features (T046-T052):
 * - Persists user's source scope preference to localStorage
 * - Persists user's disabled sources list to localStorage
 * - Defaults to 'web_only' (enterprise sources require explicit opt-in)
 * - Provides simple get/set interface
 * - TypeScript-safe with SourceScope type
 * - Filters out stale source IDs on load (T055)
 */

import { useState, useCallback, useEffect } from 'react';
import type { SourceScope } from '@/types/dataSources';

const SCOPE_STORAGE_KEY = 'deep-research-source-scope';
const DISABLED_SOURCES_STORAGE_KEY = 'deep-research-disabled-sources';
const DEFAULT_SCOPE: SourceScope = 'web_only';

interface UseSourceScopeOptions {
  /** Initial scope if no preference is stored (defaults to 'web_only') */
  initialScope?: SourceScope;
  /** List of valid source IDs (for filtering stale entries) */
  validSourceIds?: string[];
}

interface UseSourceScopeReturn {
  /** Current source scope selection */
  scope: SourceScope;
  /** Update the source scope (persists to localStorage) */
  setScope: (scope: SourceScope) => void;
  /** Reset to default scope */
  resetScope: () => void;
  /** List of disabled source IDs */
  disabledSources: string[];
  /** Set the disabled sources list (persists to localStorage) */
  setDisabledSources: (sources: string[]) => void;
  /** Toggle a single source's enabled state */
  toggleSource: (sourceId: string, enabled: boolean) => void;
  /** Reset disabled sources to empty */
  resetDisabledSources: () => void;
}

/**
 * Parse and validate a stored scope value.
 * Returns the valid scope or null if invalid.
 */
function parseStoredScope(value: string | null): SourceScope | null {
  if (!value) return null;
  const validScopes: SourceScope[] = ['enterprise_only', 'web_only', 'all'];
  if (validScopes.includes(value as SourceScope)) {
    return value as SourceScope;
  }
  return null;
}

/**
 * Parse stored disabled sources list.
 * Returns an array of source IDs, filtering out invalid entries.
 */
function parseStoredDisabledSources(value: string | null, validIds?: string[]): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    // Filter to only strings
    const sourceIds = parsed.filter((id): id is string => typeof id === 'string');
    // If we have valid IDs, filter out stale entries (T055)
    if (validIds && validIds.length > 0) {
      return sourceIds.filter((id) => validIds.includes(id));
    }
    return sourceIds;
  } catch {
    return [];
  }
}

/**
 * Hook for managing source scope and disabled sources with localStorage persistence.
 *
 * @param options - Configuration options
 * @returns Object with scope/disabledSources values and setter functions
 *
 * @example
 * ```tsx
 * const { scope, setScope, disabledSources, toggleSource } = useSourceScope();
 *
 * return (
 *   <SourceScopeSelector
 *     selectedScope={scope}
 *     onScopeChange={setScope}
 *     availableSources={sources}
 *     onSourceToggle={toggleSource}
 *   />
 * );
 * ```
 */
export function useSourceScope(options?: UseSourceScopeOptions): UseSourceScopeReturn {
  const { initialScope = DEFAULT_SCOPE, validSourceIds } = options ?? {};

  // Initialize scope state from localStorage or fallback to initial scope
  const [scope, setScopeState] = useState<SourceScope>(() => {
    if (typeof window === 'undefined') return initialScope;
    try {
      const stored = localStorage.getItem(SCOPE_STORAGE_KEY);
      return parseStoredScope(stored) ?? initialScope;
    } catch {
      // localStorage may be unavailable (e.g., private browsing)
      return initialScope;
    }
  });

  // Initialize disabled sources from localStorage
  const [disabledSources, setDisabledSourcesState] = useState<string[]>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const stored = localStorage.getItem(DISABLED_SOURCES_STORAGE_KEY);
      return parseStoredDisabledSources(stored, validSourceIds);
    } catch {
      return [];
    }
  });

  // Persist scope to localStorage whenever it changes
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(SCOPE_STORAGE_KEY, scope);
    } catch {
      // Ignore localStorage errors (quota exceeded, private browsing, etc.)
      console.warn('[useSourceScope] Failed to persist scope to localStorage');
    }
  }, [scope]);

  // Persist disabled sources to localStorage whenever they change
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(DISABLED_SOURCES_STORAGE_KEY, JSON.stringify(disabledSources));
    } catch {
      console.warn('[useSourceScope] Failed to persist disabled sources to localStorage');
    }
  }, [disabledSources]);

  // Filter out stale source IDs when validSourceIds changes (T055)
  useEffect(() => {
    if (validSourceIds && validSourceIds.length > 0) {
      setDisabledSourcesState((prev) => {
        const filtered = prev.filter((id) => validSourceIds.includes(id));
        if (filtered.length !== prev.length) {
          return filtered;
        }
        return prev;
      });
    }
  }, [validSourceIds]);

  // Setter that validates and updates scope
  const setScope = useCallback((newScope: SourceScope) => {
    const validScopes: SourceScope[] = ['enterprise_only', 'web_only', 'all'];
    if (validScopes.includes(newScope)) {
      setScopeState(newScope);
    } else {
      console.warn('[useSourceScope] Invalid scope value:', newScope);
    }
  }, []);

  // Reset to default scope
  const resetScope = useCallback(() => {
    setScopeState(DEFAULT_SCOPE);
  }, []);

  // Set disabled sources directly
  const setDisabledSources = useCallback((sources: string[]) => {
    setDisabledSourcesState(sources);
  }, []);

  // Toggle a single source
  const toggleSource = useCallback((sourceId: string, enabled: boolean) => {
    setDisabledSourcesState((prev) => {
      if (enabled) {
        // Remove from disabled list
        return prev.filter((id) => id !== sourceId);
      } else {
        // Add to disabled list
        return prev.includes(sourceId) ? prev : [...prev, sourceId];
      }
    });
  }, []);

  // Reset disabled sources to empty
  const resetDisabledSources = useCallback(() => {
    setDisabledSourcesState([]);
  }, []);

  return {
    scope,
    setScope,
    resetScope,
    disabledSources,
    setDisabledSources,
    toggleSource,
    resetDisabledSources,
  };
}

export default useSourceScope;
