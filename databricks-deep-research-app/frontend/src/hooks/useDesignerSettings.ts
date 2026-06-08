/**
 * useDesignerSettings — small localStorage-backed preference store for the
 * Agent Designer chat surface.
 *
 * Currently surfaces a single preference: whether auto-repair (Layer 2
 * `NormalizationFix` records) details are shown as a full expandable pill
 * or collapsed to a compact "!" indicator.
 *
 * Keep this hook tiny — designer-specific preferences live here; broader
 * app preferences should go through their own dedicated hook (see
 * `useSourceScope` for the canonical pattern).
 */

import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'dr.designer.settings.v1';

export interface DesignerSettings {
  /**
   * When true, the NormalizationFix pill renders with a count and an
   * expandable detail panel. When false, the pill collapses to a single "!"
   * indicator without details. Default: true.
   */
  showAutoRepairDetails: boolean;
}

const DEFAULTS: DesignerSettings = {
  showAutoRepairDetails: true,
};

function readFromStorage(): DesignerSettings {
  if (typeof window === 'undefined' || !window.localStorage) {
    return DEFAULTS;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      const obj = parsed as Partial<DesignerSettings>;
      return {
        showAutoRepairDetails:
          typeof obj.showAutoRepairDetails === 'boolean'
            ? obj.showAutoRepairDetails
            : DEFAULTS.showAutoRepairDetails,
      };
    }
  } catch {
    // Corrupt storage / private browsing — fall through to defaults.
  }
  return DEFAULTS;
}

function writeToStorage(settings: DesignerSettings): void {
  if (typeof window === 'undefined' || !window.localStorage) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // localStorage may be full or unavailable; ignore.
  }
}

export interface UseDesignerSettings {
  settings: DesignerSettings;
  setShowAutoRepairDetails(value: boolean): void;
}

export function useDesignerSettings(): UseDesignerSettings {
  const [settings, setSettings] = useState<DesignerSettings>(() =>
    readFromStorage(),
  );

  useEffect(() => {
    writeToStorage(settings);
  }, [settings]);

  const setShowAutoRepairDetails = useCallback((value: boolean) => {
    setSettings((prev) =>
      prev.showAutoRepairDetails === value
        ? prev
        : { ...prev, showAutoRepairDetails: value },
    );
  }, []);

  return { settings, setShowAutoRepairDetails };
}
