import { cn } from '@/lib/utils';

/**
 * Per-run report-style selectors: TONE + OUTPUT LANGUAGE.
 *
 * Both are optional (undefined => server default / unchanged synthesis). The
 * tone values are the lowercase framework ``Tone`` member names so they coerce
 * directly on the backend; the language values are free-form language names.
 *
 * Two compact native <select> dropdowns (the 17-tone enum is too wide for a
 * pill row); styling mirrors ResearchDepthSelector / QueryModeSelector.
 */

// Curated subset of the framework Tone enum (lowercase member names). Every
// value here is a valid Tone member; '' => default (omit from submission).
const TONE_OPTIONS: { value: string; label: string }[] = [
  { value: '', label: 'Default' },
  { value: 'objective', label: 'Objective' },
  { value: 'formal', label: 'Formal' },
  { value: 'analytical', label: 'Analytical' },
  { value: 'persuasive', label: 'Persuasive' },
  { value: 'informative', label: 'Informative' },
  { value: 'explanatory', label: 'Explanatory' },
  { value: 'descriptive', label: 'Descriptive' },
  { value: 'critical', label: 'Critical' },
  { value: 'comparative', label: 'Comparative' },
  { value: 'simple', label: 'Simple' },
  { value: 'casual', label: 'Casual' },
];

// '' => default (omit). Values are free-form language names sent verbatim.
const LANGUAGE_OPTIONS: { value: string; label: string }[] = [
  { value: '', label: 'Default' },
  { value: 'English', label: 'English' },
  { value: 'Spanish', label: 'Spanish' },
  { value: 'French', label: 'French' },
  { value: 'German', label: 'German' },
  { value: 'Portuguese', label: 'Portuguese' },
  { value: 'Italian', label: 'Italian' },
  { value: 'Japanese', label: 'Japanese' },
  { value: 'Korean', label: 'Korean' },
  { value: 'Chinese', label: 'Chinese' },
];

interface ReportStyleSelectorProps {
  tone: string;
  outputLanguage: string;
  onToneChange: (tone: string) => void;
  onLanguageChange: (language: string) => void;
  disabled?: boolean;
  className?: string;
}

const SELECT_CLASS = cn(
  'text-xs rounded border border-input bg-muted/50 px-1.5 py-1',
  'text-foreground',
  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
  'disabled:cursor-not-allowed disabled:opacity-50'
);

export function ReportStyleSelector({
  tone,
  outputLanguage,
  onToneChange,
  onLanguageChange,
  disabled = false,
  className,
}: ReportStyleSelectorProps) {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <label className="flex items-center gap-1">
        <span className="text-xs text-muted-foreground">Tone:</span>
        <select
          value={tone}
          onChange={(e) => onToneChange(e.target.value)}
          disabled={disabled}
          title="Report writing tone"
          data-testid="tone-select"
          className={SELECT_CLASS}
        >
          {TONE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1">
        <span className="text-xs text-muted-foreground">Language:</span>
        <select
          value={outputLanguage}
          onChange={(e) => onLanguageChange(e.target.value)}
          disabled={disabled}
          title="Report output language"
          data-testid="language-select"
          className={SELECT_CLASS}
        >
          {LANGUAGE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
