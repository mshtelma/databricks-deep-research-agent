/**
 * VariableInput - Dynamic form for template variables.
 *
 * Renders appropriate input controls based on variable type:
 * - string: text input
 * - number: number input
 * - boolean: checkbox
 * - array: tag input or multi-line
 * - object: JSON textarea
 *
 * Features:
 * - Validation for required variables
 * - Show default values as placeholders
 * - Error display for validation failures
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Input } from '@/components/ui/input';
import type {
  TemplateVariable,
  TemplateVariableType,
} from '@/types/templates';

interface VariableInputProps {
  /** Array of variable definitions */
  variables: TemplateVariable[];
  /** Current values for variables */
  values: Record<string, unknown>;
  /** Callback when values change */
  onChange: (values: Record<string, unknown>) => void;
  /** Validation errors by variable name */
  errors?: Record<string, string>;
  /** Whether inputs are disabled */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function VariableInput({
  variables,
  values,
  onChange,
  errors = {},
  disabled = false,
  className,
}: VariableInputProps) {
  const handleValueChange = (name: string, value: unknown) => {
    onChange({
      ...values,
      [name]: value,
    });
  };

  if (variables.length === 0) {
    return (
      <div className={cn('text-sm text-muted-foreground text-center py-4', className)}>
        No variables defined for this template.
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {variables.map((variable) => (
        <VariableField
          key={variable.name}
          variable={variable}
          value={values[variable.name]}
          onChange={(value) => handleValueChange(variable.name, value)}
          error={errors[variable.name]}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

interface VariableFieldProps {
  variable: TemplateVariable;
  value: unknown;
  onChange: (value: unknown) => void;
  error?: string;
  disabled?: boolean;
}

function VariableField({
  variable,
  value,
  onChange,
  error,
  disabled,
}: VariableFieldProps) {
  const { name, type, required, description } = variable;
  const defaultVal = variable.default;

  const getPlaceholder = () => {
    if (defaultVal !== undefined && defaultVal !== null) {
      if (typeof defaultVal === 'object') {
        return JSON.stringify(defaultVal);
      }
      return String(defaultVal);
    }
    return undefined;
  };

  return (
    <div className="space-y-1.5">
      <label
        htmlFor={`var-${name}`}
        className="text-sm font-medium flex items-center gap-2"
      >
        <span>{name}</span>
        {required && (
          <span className="text-destructive text-xs">*</span>
        )}
        <TypeBadge type={type} />
      </label>

      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}

      <VariableInputControl
        id={`var-${name}`}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={getPlaceholder()}
        disabled={disabled}
        hasError={!!error}
      />

      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}
    </div>
  );
}

interface VariableInputControlProps {
  id: string;
  type: TemplateVariableType;
  value: unknown;
  onChange: (value: unknown) => void;
  placeholder?: string;
  disabled?: boolean;
  hasError?: boolean;
}

function VariableInputControl({
  id,
  type,
  value,
  onChange,
  placeholder,
  disabled,
  hasError,
}: VariableInputControlProps) {
  const baseInputClass = cn(
    hasError && 'border-destructive focus-visible:ring-destructive'
  );

  switch (type) {
    case 'string':
      return (
        <Input
          id={id}
          type="text"
          value={(value as string) ?? ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={baseInputClass}
        />
      );

    case 'number':
      return (
        <Input
          id={id}
          type="number"
          value={value !== undefined ? String(value) : ''}
          onChange={(e) => {
            const val = e.target.value;
            onChange(val === '' ? undefined : Number(val));
          }}
          placeholder={placeholder}
          disabled={disabled}
          className={baseInputClass}
        />
      );

    case 'boolean':
      return (
        <div className="flex items-center gap-2">
          <input
            id={id}
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
            disabled={disabled}
            className={cn(
              'h-4 w-4 rounded border border-input',
              'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
          <span className="text-sm text-muted-foreground">
            {value ? 'Yes' : 'No'}
          </span>
        </div>
      );

    case 'array':
      return (
        <ArrayInput
          id={id}
          value={value as unknown[]}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          hasError={hasError}
        />
      );

    case 'object':
      return (
        <JsonInput
          id={id}
          value={value as Record<string, unknown>}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          hasError={hasError}
        />
      );

    default:
      return (
        <Input
          id={id}
          type="text"
          value={String(value ?? '')}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={baseInputClass}
        />
      );
  }
}

interface ArrayInputProps {
  id: string;
  value: unknown[] | undefined;
  onChange: (value: unknown[]) => void;
  placeholder?: string;
  disabled?: boolean;
  hasError?: boolean;
}

function ArrayInput({
  id,
  value,
  onChange,
  placeholder,
  disabled,
  hasError,
}: ArrayInputProps) {
  const [inputValue, setInputValue] = React.useState('');
  const items = Array.isArray(value) ? value : [];

  const handleAdd = () => {
    const trimmed = inputValue.trim();
    if (trimmed && !items.includes(trimmed)) {
      onChange([...items, trimmed]);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAdd();
    }
  };

  const handleRemove = (index: number) => {
    onChange(items.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      {/* Tags display */}
      {items.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {items.map((item, index) => (
            <span
              key={index}
              className={cn(
                'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs',
                'bg-secondary text-secondary-foreground'
              )}
            >
              {String(item)}
              {!disabled && (
                <button
                  type="button"
                  onClick={() => handleRemove(index)}
                  className="hover:text-destructive"
                >
                  <XIcon className="h-3 w-3" />
                </button>
              )}
            </span>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2">
        <Input
          id={id}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder ?? 'Type and press Enter to add'}
          disabled={disabled}
          className={cn(
            'flex-1',
            hasError && 'border-destructive focus-visible:ring-destructive'
          )}
        />
        <button
          type="button"
          onClick={handleAdd}
          disabled={disabled || !inputValue.trim()}
          className={cn(
            'px-3 py-1 rounded-md text-sm font-medium',
            'bg-secondary text-secondary-foreground',
            'hover:bg-secondary/80',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
        >
          Add
        </button>
      </div>
    </div>
  );
}

interface JsonInputProps {
  id: string;
  value: Record<string, unknown> | undefined;
  onChange: (value: Record<string, unknown>) => void;
  placeholder?: string;
  disabled?: boolean;
  hasError?: boolean;
}

function JsonInput({
  id,
  value,
  onChange,
  placeholder,
  disabled,
  hasError,
}: JsonInputProps) {
  const [jsonText, setJsonText] = React.useState(() =>
    value ? JSON.stringify(value, null, 2) : ''
  );
  const [parseError, setParseError] = React.useState<string | null>(null);

  const handleChange = (text: string) => {
    setJsonText(text);

    if (!text.trim()) {
      setParseError(null);
      onChange({});
      return;
    }

    try {
      const parsed = JSON.parse(text);
      if (typeof parsed === 'object' && !Array.isArray(parsed)) {
        setParseError(null);
        onChange(parsed);
      } else {
        setParseError('Must be a JSON object');
      }
    } catch {
      setParseError('Invalid JSON');
    }
  };

  return (
    <div className="space-y-1">
      <textarea
        id={id}
        value={jsonText}
        onChange={(e) => handleChange(e.target.value)}
        placeholder={placeholder ?? '{\n  "key": "value"\n}'}
        disabled={disabled}
        rows={4}
        className={cn(
          'w-full rounded-md border bg-transparent px-3 py-2 text-sm font-mono',
          'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1',
          'disabled:cursor-not-allowed disabled:opacity-50 resize-none',
          hasError || parseError
            ? 'border-destructive focus-visible:ring-destructive'
            : 'border-input focus-visible:ring-ring'
        )}
      />
      {parseError && (
        <p className="text-xs text-destructive">{parseError}</p>
      )}
    </div>
  );
}

function TypeBadge({ type }: { type: TemplateVariableType }) {
  const colors: Record<TemplateVariableType, string> = {
    string: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
    number: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    boolean: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    array: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
    object: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
  };

  return (
    <span
      className={cn(
        'text-xs px-1.5 py-0.5 rounded',
        colors[type] || colors.string
      )}
    >
      {type}
    </span>
  );
}

function XIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}

export default VariableInput;
