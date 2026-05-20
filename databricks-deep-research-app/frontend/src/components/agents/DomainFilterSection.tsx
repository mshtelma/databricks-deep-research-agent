/**
 * DomainFilterSection - Per-agent domain whitelist/blacklist editor.
 *
 * Renders:
 * - Mode selector (None, Include, Exclude, Both) — binary search-time filter
 * - Textarea for include patterns (hard whitelist)
 * - Textarea for exclude patterns (hard blacklist)
 * - Textarea for PREFERRED patterns — soft ranking boost (NEW)
 * - Textarea for DEPRECATED patterns — soft ranking penalty (NEW)
 *
 * Filter and reputation are orthogonal: filter mode controls whether the
 * include/exclude textareas hard-filter results; preferred/deprecated
 * always re-rank survivors via the framework's source-admission scorer.
 *
 * Part of 009-custom-agent-config (T037); reputation fields added by PR 3
 * of the scaffolding quality plan.
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

type DomainFilterMode = 'include' | 'exclude' | 'both' | null;

interface DomainFilterSectionProps {
  /** Current filter mode (null = no filtering) */
  domainFilterMode: string | null;
  /** Include domain patterns (hard whitelist) */
  includeDomains: string[] | null;
  /** Exclude domain patterns (hard blacklist) */
  excludeDomains: string[] | null;
  /** Preferred domain patterns (soft ranking boost). Independent of mode. */
  preferredDomains?: string[] | null;
  /** Deprecated domain patterns (soft ranking penalty). Independent of mode. */
  deprecatedDomains?: string[] | null;
  /** Callback when any value changes */
  onChange: (
    mode: string | null,
    includeDomains: string[] | null,
    excludeDomains: string[] | null,
    preferredDomains?: string[] | null,
    deprecatedDomains?: string[] | null,
  ) => void;
  /** Whether the form is disabled */
  disabled?: boolean;
}

const DOMAIN_PATTERN_RE = /^[a-zA-Z0-9.*-]+$/;

const MODE_OPTIONS: { value: DomainFilterMode; label: string; description: string }[] = [
  { value: null, label: 'None', description: 'No domain filtering' },
  { value: 'include', label: 'Include', description: 'Only allow listed domains' },
  { value: 'exclude', label: 'Exclude', description: 'Block listed domains' },
  { value: 'both', label: 'Both', description: 'Allow listed, then block from those' },
];

function parseDomains(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function validateDomains(domains: string[]): string[] {
  const errors: string[] = [];
  for (const d of domains) {
    if (!DOMAIN_PATTERN_RE.test(d)) {
      errors.push(`Invalid pattern: "${d}"`);
    }
  }
  return errors;
}

export function DomainFilterSection({
  domainFilterMode,
  includeDomains,
  excludeDomains,
  preferredDomains = null,
  deprecatedDomains = null,
  onChange,
  disabled = false,
}: DomainFilterSectionProps) {
  const mode = (domainFilterMode as DomainFilterMode) ?? null;
  const showInclude = mode === 'include' || mode === 'both';
  const showExclude = mode === 'exclude' || mode === 'both';

  const [includeText, setIncludeText] = React.useState(
    (includeDomains ?? []).join('\n')
  );
  const [excludeText, setExcludeText] = React.useState(
    (excludeDomains ?? []).join('\n')
  );
  // Reputation lists are independent of mode — they're always editable
  // because they re-rank, not filter.
  const [preferredText, setPreferredText] = React.useState(
    (preferredDomains ?? []).join('\n')
  );
  const [deprecatedText, setDeprecatedText] = React.useState(
    (deprecatedDomains ?? []).join('\n')
  );

  const includeErrors = React.useMemo(
    () => (showInclude ? validateDomains(parseDomains(includeText)) : []),
    [includeText, showInclude]
  );
  const excludeErrors = React.useMemo(
    () => (showExclude ? validateDomains(parseDomains(excludeText)) : []),
    [excludeText, showExclude]
  );
  const preferredErrors = React.useMemo(
    () => validateDomains(parseDomains(preferredText)),
    [preferredText]
  );
  const deprecatedErrors = React.useMemo(
    () => validateDomains(parseDomains(deprecatedText)),
    [deprecatedText]
  );

  // Helper — call onChange forwarding the reputation lists (which are
  // independent of mode) so a mode change does not nuke them.
  const currentPreferred = () => parseDomains(preferredText);
  const currentDeprecated = () => parseDomains(deprecatedText);

  const handleModeChange = (newMode: DomainFilterMode) => {
    if (newMode === null) {
      onChange(null, null, null, currentPreferred(), currentDeprecated());
    } else {
      onChange(
        newMode,
        newMode === 'include' || newMode === 'both' ? parseDomains(includeText) : null,
        newMode === 'exclude' || newMode === 'both' ? parseDomains(excludeText) : null,
        currentPreferred(),
        currentDeprecated(),
      );
    }
  };

  const handleIncludeChange = (text: string) => {
    setIncludeText(text);
    onChange(mode, parseDomains(text), excludeDomains, currentPreferred(), currentDeprecated());
  };

  const handleExcludeChange = (text: string) => {
    setExcludeText(text);
    onChange(mode, includeDomains, parseDomains(text), currentPreferred(), currentDeprecated());
  };

  const handlePreferredChange = (text: string) => {
    setPreferredText(text);
    onChange(mode, includeDomains, excludeDomains, parseDomains(text), currentDeprecated());
  };

  const handleDeprecatedChange = (text: string) => {
    setDeprecatedText(text);
    onChange(mode, includeDomains, excludeDomains, currentPreferred(), parseDomains(text));
  };

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-medium">Domain Filter</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Restrict web search results to specific domains. Supports wildcards (e.g.,
          <code className="mx-1 text-xs bg-muted px-1 py-0.5 rounded">*.gov</code>,
          <code className="mx-1 text-xs bg-muted px-1 py-0.5 rounded">*.edu</code>).
        </p>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2">
        {MODE_OPTIONS.map((opt) => (
          <button
            key={opt.value ?? 'none'}
            type="button"
            onClick={() => handleModeChange(opt.value)}
            disabled={disabled}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm transition-colors border',
              mode === opt.value
                ? 'border-primary bg-primary/10 text-primary font-medium'
                : 'border-input text-muted-foreground hover:border-primary/50 hover:text-foreground'
            )}
            title={opt.description}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Include domains textarea */}
      {showInclude && (
        <div>
          <label className="text-sm font-medium mb-1.5 block">
            Include Domains (whitelist)
          </label>
          <textarea
            value={includeText}
            onChange={(e) => handleIncludeChange(e.target.value)}
            placeholder={"*.gov\n*.edu\nexample.com"}
            rows={4}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50',
              includeErrors.length > 0 && 'border-destructive'
            )}
          />
          {includeErrors.length > 0 && (
            <p className="text-xs text-destructive mt-1">{includeErrors.join(', ')}</p>
          )}
          <p className="text-xs text-muted-foreground mt-1">
            One domain pattern per line. Only results from these domains will be included.
          </p>
        </div>
      )}

      {/* Exclude domains textarea */}
      {showExclude && (
        <div>
          <label className="text-sm font-medium mb-1.5 block">
            Exclude Domains (blacklist)
          </label>
          <textarea
            value={excludeText}
            onChange={(e) => handleExcludeChange(e.target.value)}
            placeholder={"spam.com\nads.net\n*.click"}
            rows={4}
            disabled={disabled}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50',
              excludeErrors.length > 0 && 'border-destructive'
            )}
          />
          {excludeErrors.length > 0 && (
            <p className="text-xs text-destructive mt-1">{excludeErrors.join(', ')}</p>
          )}
          <p className="text-xs text-muted-foreground mt-1">
            One domain pattern per line. Results from these domains will be blocked.
          </p>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* Reputation (ranking) — independent of filter mode. Soft bias only.   */}
      {/* ------------------------------------------------------------------- */}
      <div className="border-t border-border pt-4 mt-2">
        <h4 className="text-sm font-medium">Source ranking (advanced)</h4>
        <p className="text-xs text-muted-foreground mt-1">
          These lists don&apos;t filter — they nudge the admission ranking so
          higher-quality sources appear first. They apply only to URLs that
          survive the filter above. Wildcards supported.
        </p>
      </div>

      {/* Preferred domains textarea — soft boost */}
      <div>
        <label className="text-sm font-medium mb-1.5 block">
          Preferred Domains (boost in ranking)
        </label>
        <textarea
          value={preferredText}
          onChange={(e) => handlePreferredChange(e.target.value)}
          placeholder={"*.gov\ninvestors.*\nofficial-vendor.com"}
          rows={3}
          disabled={disabled}
          className={cn(
            'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
            preferredErrors.length > 0 && 'border-destructive'
          )}
        />
        {preferredErrors.length > 0 && (
          <p className="text-xs text-destructive mt-1">{preferredErrors.join(', ')}</p>
        )}
        <p className="text-xs text-muted-foreground mt-1">
          One pattern per line. Sources from these domains rank higher in admission.
        </p>
      </div>

      {/* Deprecated domains textarea — soft penalty */}
      <div>
        <label className="text-sm font-medium mb-1.5 block">
          Deprecated Domains (penalty in ranking)
        </label>
        <textarea
          value={deprecatedText}
          onChange={(e) => handleDeprecatedChange(e.target.value)}
          placeholder={"content-farm.example\nai-generated.*"}
          rows={3}
          disabled={disabled}
          className={cn(
            'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm font-mono',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
            deprecatedErrors.length > 0 && 'border-destructive'
          )}
        />
        {deprecatedErrors.length > 0 && (
          <p className="text-xs text-destructive mt-1">{deprecatedErrors.join(', ')}</p>
        )}
        <p className="text-xs text-muted-foreground mt-1">
          One pattern per line. Sources from these domains rank lower in admission (not blocked).
        </p>
      </div>
    </div>
  );
}
