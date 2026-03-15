/**
 * ModelConfigSection - Per-tier model endpoint override editor.
 *
 * Uses an input with datalist for autocomplete, allowing both YAML-configured
 * endpoints and arbitrary workspace serving endpoints. Users can type any
 * Databricks serving endpoint name as a model override.
 *
 * Part of 009-custom-agent-config (T026).
 */

import { cn } from '@/lib/utils';
import { useModelCatalog } from '@/hooks/useModelCatalog';

interface ModelConfigSectionProps {
  /** Current model overrides: tier name -> endpoint ID, or null for defaults */
  modelOverrides: Record<string, string> | null;
  /** Callback when overrides change */
  onChange: (overrides: Record<string, string> | null) => void;
  /** Whether the form is disabled */
  disabled?: boolean;
}

export function ModelConfigSection({
  modelOverrides,
  onChange,
  disabled = false,
}: ModelConfigSectionProps) {
  const { categories, endpoints, workspaceEndpoints, configEndpointNames, isLoading, error } =
    useModelCatalog();
  const categoryNames = Object.keys(categories);

  const handleTierChange = (tierName: string, value: string) => {
    const trimmed = value.trim();
    const current = { ...(modelOverrides || {}) };

    if (trimmed === '') {
      // Empty = "Use Default" — remove override for this tier
      delete current[tierName];
    } else {
      current[tierName] = trimmed;
    }

    // If all overrides removed, set to null
    onChange(Object.keys(current).length > 0 ? current : null);
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-medium">Model Overrides</h3>
          <p className="text-sm text-muted-foreground mt-1 animate-pulse">
            Loading model catalog...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-medium">Model Overrides</h3>
          <p className="text-sm text-destructive mt-1">
            Failed to load model catalog. Model overrides are unavailable.
          </p>
        </div>
      </div>
    );
  }

  if (categoryNames.length === 0) {
    return (
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-medium">Model Overrides</h3>
          <p className="text-sm text-muted-foreground mt-1">No model tiers configured.</p>
        </div>
      </div>
    );
  }

  const endpointNames = Object.keys(endpoints);

  // Build deduplicated workspace endpoint list:
  // Exclude endpoints whose name matches a YAML endpoint_identifier
  // (those are already shown under their YAML alias)
  const dedupedWorkspaceEndpoints = workspaceEndpoints.filter(
    (ep) => !configEndpointNames.includes(ep.name) && !endpointNames.includes(ep.name),
  );

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-medium">Model Overrides</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Override the default model endpoint for each tier. Type an endpoint name or select from
          suggestions. Clear the field to use the system default.
        </p>
      </div>

      <div className="space-y-3">
        {categoryNames.map((tierName) => {
          const category = categories[tierName];
          const currentOverride = modelOverrides?.[tierName] ?? '';
          const datalistId = `tier-endpoints-${tierName}`;

          return (
            <div key={tierName} className="flex items-center gap-3">
              <label className="w-32 text-sm font-medium capitalize shrink-0">{tierName}</label>
              <div className="relative flex-1">
                <input
                  type="text"
                  list={datalistId}
                  value={currentOverride}
                  onChange={(e) => handleTierChange(tierName, e.target.value)}
                  placeholder={`Default: ${category?.defaultEndpoints?.[0] ?? 'none'}`}
                  disabled={disabled}
                  className={cn(
                    'w-full h-9 rounded-md border border-input bg-transparent px-3 text-sm',
                    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                    'disabled:cursor-not-allowed disabled:opacity-50',
                    'placeholder:text-muted-foreground',
                  )}
                />
                <datalist id={datalistId}>
                  {endpointNames.map((ep) => (
                    <option key={`yaml-${ep}`} value={ep} />
                  ))}
                  {dedupedWorkspaceEndpoints
                    .filter((ep) => ep.state === 'READY')
                    .map((ep) => (
                      <option key={`ws-${ep.name}`} value={ep.name} />
                    ))}
                </datalist>
              </div>
              {/* Clear button — only shown when override is set */}
              {currentOverride && !disabled && (
                <button
                  type="button"
                  onClick={() => handleTierChange(tierName, '')}
                  className="text-muted-foreground hover:text-foreground shrink-0"
                  title="Reset to default"
                >
                  &times;
                </button>
              )}
            </div>
          );
        })}
      </div>

      {modelOverrides && Object.keys(modelOverrides).length > 0 && (
        <button
          type="button"
          onClick={() => onChange(null)}
          disabled={disabled}
          className="text-xs text-muted-foreground hover:text-foreground underline"
        >
          Reset all to defaults
        </button>
      )}
    </div>
  );
}
