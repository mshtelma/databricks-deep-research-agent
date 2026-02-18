/**
 * DomainFilterSection - Per-agent domain whitelist/blacklist editor.
 *
 * Renders:
 * - Mode selector (None, Include, Exclude, Both)
 * - Textarea for include patterns (one per line)
 * - Textarea for exclude patterns (one per line)
 *
 * Part of 009-custom-agent-config (T037).
 */

import * as React from 'react';
import { cn } from '@/lib/utils';

type DomainFilterMode = 'include' | 'exclude' | 'both' | null;

interface DomainFilterSectionProps {
  /** Current filter mode (null = no filtering) */
  domainFilterMode: string | null;
  /** Include domain patterns */
  includeDomains: string[] | null;
  /** Exclude domain patterns */
  excludeDomains: string[] | null;
  /** Callback when any value changes */
  onChange: (
    mode: string | null,
    includeDomains: string[] | null,
    excludeDomains: string[] | null,
  ) => void;
  /** Whether the form is disabled */
  disabled?: boolean;
}

const DOMAIN_PATTERN_RE = /^[a-zA-Z0-9.*\-]+$/;

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

  const includeErrors = React.useMemo(
    () => (showInclude ? validateDomains(parseDomains(includeText)) : []),
    [includeText, showInclude]
  );
  const excludeErrors = React.useMemo(
    () => (showExclude ? validateDomains(parseDomains(excludeText)) : []),
    [excludeText, showExclude]
  );

  const handleModeChange = (newMode: DomainFilterMode) => {
    if (newMode === null) {
      onChange(null, null, null);
    } else {
      onChange(
        newMode,
        newMode === 'include' || newMode === 'both' ? parseDomains(includeText) : null,
        newMode === 'exclude' || newMode === 'both' ? parseDomains(excludeText) : null,
      );
    }
  };

  const handleIncludeChange = (text: string) => {
    setIncludeText(text);
    onChange(mode, parseDomains(text), excludeDomains);
  };

  const handleExcludeChange = (text: string) => {
    setExcludeText(text);
    onChange(mode, includeDomains, parseDomains(text));
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
    </div>
  );
}
