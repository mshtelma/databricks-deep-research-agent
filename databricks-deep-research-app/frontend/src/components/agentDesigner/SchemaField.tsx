/**
 * SchemaField — recursive JSON-Schema field renderer for the Agent Designer ConfigPanel.
 *
 * Supported widgets:
 *   1. text          — string input (default for type=string)
 *   2. multiline-text — textarea (type=string + format=multiline)
 *   3. number        — number input (type=integer | number)
 *   4. checkbox      — boolean input (type=boolean)
 *   5. select        — Radix Select (when schema.enum is present, overrides type mapping)
 *   6. array         — list with per-item SchemaField + Add/Remove buttons
 *   7. object        — nested fieldset, one SchemaField per schema.properties key
 *   8. password      — text input with type=password (x-widget="password")
 *   9. resource-select — datalist-backed resource picker with manual fallback
 *
 * x-widget extension overrides the default mapping:
 *   "code"     → textarea with font-mono styling
 *   "prompt"   → multi-line textarea
 *   "resource-select" → searchable resource picker
 *   "password" → password input
 *   unknown    → falls back to default + logs console.warn
 *
 * Styling tracks the Databricks Agentic Designer system: `field-input` look
 * (white surface, gray-lines border, navy-400 focus + lava focus ring), labels
 * use `text-db-navy-800`, descriptions use `text-db-gray-text`, required
 * markers use `text-db-lava-600`, errors use `text-db-lava-700`. Array Remove
 * (delete) buttons use the lava-300 / lava-800 affordance from the design.
 */

import * as React from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import { ChevronDown, Plus, X } from 'lucide-react';
import { useDesignerResources } from '@/hooks/useDesignerResources';
import type { DesignerResource } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Shared className recipes — keep styling tokens in one place so future
// adjustments propagate everywhere.
// ---------------------------------------------------------------------------

const FIELD_INPUT_CLASS =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-sans text-[13px] leading-[1.4] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const FIELD_INPUT_MONO_CLASS =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] leading-[1.4] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const FIELD_TEXTAREA_CLASS =
  'w-full resize-y rounded-db-md border border-db-gray-lines bg-white px-3 py-2 font-db-sans text-[13px] leading-[1.55] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const FIELD_TEXTAREA_MONO_CLASS =
  'w-full resize-y rounded-db-md border border-db-gray-lines bg-white px-3 py-2 font-db-mono text-[12px] leading-[1.55] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const LABEL_CLASS = 'mb-1 block font-db-sans text-[12px] font-medium text-db-navy-800';

const REQUIRED_MARK_CLASS = 'ml-0.5 text-db-lava-600';

const DESCRIPTION_CLASS = 'mt-1 text-[11px] text-db-gray-text';

const ERROR_ITEM_CLASS = 'text-[11px] text-db-lava-700';

// Icon-only delete affordance (btn-tiny recipe from the design system).
// A subtle ghost square that flips to the lava palette on hover/focus, so the
// destructive action is discoverable but doesn't shout for attention.
const BTN_DELETE_CLASS =
  'inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-db-md border border-transparent bg-transparent text-db-gray-text transition-colors hover:border-db-lava-400 hover:bg-db-lava-300 hover:text-db-lava-800 focus:outline-none focus:border-db-lava-400 focus:bg-db-lava-300 focus:text-db-lava-800 focus:shadow-db-focus';

const BTN_GHOST_CLASS =
  'inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1 font-db-sans text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium focus:outline-none focus:shadow-db-focus';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert snake_case or camelCase to Title Case for display. */
function toDisplayLabel(name: string): string {
  return name
    .replace(/[_-]/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface SchemaFieldProps {
  name: string;
  schema: Record<string, unknown>;
  value: unknown;
  onChange: (value: unknown) => void;
  required?: boolean;
  errors?: string[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function SchemaField({
  name,
  schema,
  value,
  onChange,
  required = false,
  errors = [],
}: SchemaFieldProps): React.ReactElement {
  const label = typeof schema['title'] === 'string' ? schema['title'] : toDisplayLabel(name);
  const description =
    typeof schema['description'] === 'string' ? schema['description'] : undefined;
  const xWidget = schema['x-widget'];
  const schemaType = schema['type'] as string | undefined;
  const enumValues = schema['enum'];

  // Sanitize name for use as HTML id (replace characters invalid in ids)
  const fieldId = `schema-field-${name.replace(/[^a-zA-Z0-9_-]/g, '_')}`;

  // -------------------------------------------------------------------------
  // Determine widget type
  // -------------------------------------------------------------------------

  let widgetKind:
    | 'text'
    | 'multiline-text'
    | 'number'
    | 'checkbox'
    | 'select'
    | 'array'
    | 'object'
    | 'password'
    | 'code'
    | 'prompt'
    | 'resource-select';

  if (Array.isArray(enumValues)) {
    widgetKind = 'select';
  } else if (xWidget !== undefined) {
    const w = String(xWidget);
    if (w === 'code') {
      widgetKind = 'code';
    } else if (w === 'prompt') {
      widgetKind = 'prompt';
    } else if (w === 'resource-select') {
      widgetKind = 'resource-select';
    } else if (w === 'password') {
      widgetKind = 'password';
    } else {
      console.warn(
        `agent-designer: unknown widget '${w}' for field '${name}'; falling back to default`,
      );
      widgetKind = resolveDefaultKind(schemaType, schema);
    }
  } else {
    widgetKind = resolveDefaultKind(schemaType, schema);
  }

  // -------------------------------------------------------------------------
  // Shared sub-elements
  // -------------------------------------------------------------------------

  const descEl = description ? <p className={DESCRIPTION_CLASS}>{description}</p> : null;

  const errEls =
    errors.length > 0 ? (
      <ul className="mt-1 space-y-0.5">
        {errors.map((e, i) => (
          <li key={i} className={ERROR_ITEM_CLASS}>
            {e}
          </li>
        ))}
      </ul>
    ) : null;

  // -------------------------------------------------------------------------
  // Render by widget kind
  // -------------------------------------------------------------------------

  if (widgetKind === 'checkbox') {
    return (
      <div className="mb-3.5">
        <label
          htmlFor={fieldId}
          className="flex cursor-pointer items-center gap-2 font-db-sans text-[12px] font-medium text-db-navy-800"
        >
          <input
            id={fieldId}
            type="checkbox"
            name={name}
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
            className="h-4 w-4 rounded-sm border-db-gray-lines text-db-lava-600 outline-none focus:shadow-db-focus"
          />
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        {descEl}
        {errEls}
      </div>
    );
  }

  if (widgetKind === 'select') {
    const opts = (enumValues as unknown[]).map(String);
    const currentVal = value !== undefined && value !== null ? String(value) : '';
    return (
      <div className="mb-3.5">
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        <SelectPrimitive.Root value={currentVal} onValueChange={(v) => onChange(v)}>
          <SelectPrimitive.Trigger
            id={fieldId}
            className={`flex items-center justify-between ${FIELD_INPUT_CLASS}`}
            aria-label={label}
          >
            <SelectPrimitive.Value placeholder="Select…">
              {currentVal || 'Select…'}
            </SelectPrimitive.Value>
            <SelectPrimitive.Icon className="ml-2 text-db-gray-text">
              <ChevronDown size={14} />
            </SelectPrimitive.Icon>
          </SelectPrimitive.Trigger>
          <SelectPrimitive.Portal>
            <SelectPrimitive.Content className="z-50 min-w-[8rem] overflow-hidden rounded-db-md border border-db-gray-lines bg-white font-db-sans shadow-db-md">
              <SelectPrimitive.Viewport className="p-1">
                {opts.map((opt) => (
                  <SelectPrimitive.Item
                    key={opt}
                    value={opt}
                    className="relative flex cursor-pointer select-none items-center rounded px-2 py-1.5 text-[13px] text-db-navy-800 outline-none transition-colors hover:bg-db-oat-medium focus:bg-db-oat-medium"
                  >
                    <SelectPrimitive.ItemText>{opt}</SelectPrimitive.ItemText>
                  </SelectPrimitive.Item>
                ))}
              </SelectPrimitive.Viewport>
            </SelectPrimitive.Content>
          </SelectPrimitive.Portal>
        </SelectPrimitive.Root>
        {descEl}
        {errEls}
      </div>
    );
  }

  if (widgetKind === 'number') {
    return (
      <div className="mb-3.5">
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        <input
          id={fieldId}
          type="number"
          name={name}
          value={value !== undefined && value !== null ? String(value) : ''}
          onChange={(e) => {
            const v = e.target.value;
            if (v === '' || v === '-') {
              onChange(undefined);
            } else {
              const parsed =
                schemaType === 'integer' ? parseInt(v, 10) : parseFloat(v);
              onChange(isNaN(parsed) ? undefined : parsed);
            }
          }}
          className={FIELD_INPUT_MONO_CLASS}
        />
        {descEl}
        {errEls}
      </div>
    );
  }

  if (widgetKind === 'multiline-text' || widgetKind === 'code' || widgetKind === 'prompt') {
    const isMono = widgetKind === 'code';
    const rows = widgetKind === 'code' || widgetKind === 'prompt' ? 6 : 4;
    return (
      <div className="mb-3.5">
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        <textarea
          id={fieldId}
          name={name}
          value={typeof value === 'string' ? value : ''}
          onChange={(e) => onChange(e.target.value)}
          rows={rows}
          className={isMono ? FIELD_TEXTAREA_MONO_CLASS : FIELD_TEXTAREA_CLASS}
        />
        {descEl}
        {errEls}
      </div>
    );
  }

  if (widgetKind === 'resource-select') {
    return (
      <ResourceSelectField
        fieldId={fieldId}
        name={name}
        label={label}
        schema={schema}
        value={value}
        onChange={onChange}
        required={required}
        description={descEl}
        errors={errEls}
      />
    );
  }

  if (widgetKind === 'password') {
    return (
      <div className="mb-3.5">
        <label htmlFor={fieldId} className={LABEL_CLASS}>
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        <input
          id={fieldId}
          type="password"
          name={name}
          value={typeof value === 'string' ? value : ''}
          onChange={(e) => onChange(e.target.value)}
          className={FIELD_INPUT_CLASS}
        />
        {descEl}
        {errEls}
      </div>
    );
  }

  if (widgetKind === 'array') {
    const itemSchema =
      schema['items'] && typeof schema['items'] === 'object'
        ? (schema['items'] as Record<string, unknown>)
        : {};
    const arr = Array.isArray(value) ? value : [];

    return (
      <div className="mb-3.5">
        <label className={LABEL_CLASS}>
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        <div className="space-y-2 rounded-db-md border border-db-gray-lines bg-db-oat-light p-3">
          {arr.length === 0 && (
            <p className="text-[11px] italic text-db-gray-text">No items yet.</p>
          )}
          {arr.map((item, idx) => (
            <div key={idx} className="flex items-start gap-2">
              <div className="min-w-0 flex-1">
                <SchemaField
                  name={`${name}[${idx}]`}
                  schema={itemSchema}
                  value={item}
                  onChange={(v) => {
                    const next = [...arr];
                    next[idx] = v;
                    onChange(next);
                  }}
                  errors={[]}
                />
              </div>
              <button
                type="button"
                onClick={() => {
                  const next = arr.filter((_, i) => i !== idx);
                  onChange(next);
                }}
                aria-label={`Remove item ${idx + 1}`}
                title="Remove item"
                className={`mt-1 ${BTN_DELETE_CLASS}`}
              >
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() => {
              const defaultItem = defaultValueForSchema(itemSchema);
              onChange([...arr, defaultItem]);
            }}
            className={BTN_GHOST_CLASS}
          >
            <Plus size={11} /> Add
          </button>
        </div>
        {descEl}
        {errEls}
      </div>
    );
  }

  if (widgetKind === 'object') {
    const properties =
      schema['properties'] && typeof schema['properties'] === 'object'
        ? (schema['properties'] as Record<string, Record<string, unknown>>)
        : {};
    const requiredKeys = Array.isArray(schema['required'])
      ? (schema['required'] as string[])
      : [];
    const obj =
      value && typeof value === 'object' && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : {};

    return (
      <div className="mb-3.5">
        <label className={LABEL_CLASS}>
          {label}
          {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
        </label>
        <fieldset className="space-y-0 rounded-db-md border border-db-gray-lines bg-db-oat-light p-3">
          <legend className="px-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
            {label}
          </legend>
          {Object.entries(properties).map(([key, propSchema]) => (
            <SchemaField
              key={key}
              name={key}
              schema={propSchema}
              value={obj[key]}
              onChange={(v) => {
                onChange({ ...obj, [key]: v });
              }}
              required={requiredKeys.includes(key)}
              errors={[]}
            />
          ))}
        </fieldset>
        {descEl}
        {errEls}
      </div>
    );
  }

  // Default: text input
  return (
    <div className="mb-3.5">
      <label htmlFor={fieldId} className={LABEL_CLASS}>
        {label}
        {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
      </label>
      <input
        id={fieldId}
        type="text"
        name={name}
        value={typeof value === 'string' ? value : ''}
        onChange={(e) => onChange(e.target.value)}
        className={FIELD_INPUT_CLASS}
      />
      {descEl}
      {errEls}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

type WidgetKind =
  | 'text'
  | 'multiline-text'
  | 'number'
  | 'checkbox'
  | 'select'
  | 'array'
  | 'object'
  | 'password';

function resolveDefaultKind(
  schemaType: string | undefined,
  schema: Record<string, unknown>,
): WidgetKind {
  if (schemaType === 'boolean') return 'checkbox';
  if (schemaType === 'integer' || schemaType === 'number') return 'number';
  if (schemaType === 'array') return 'array';
  if (schemaType === 'object') return 'object';
  if (schemaType === 'string' && schema['format'] === 'multiline') return 'multiline-text';
  return 'text';
}

function defaultValueForSchema(schema: Record<string, unknown>): unknown {
  const t = schema['type'] as string | undefined;
  if (t === 'string') return '';
  if (t === 'integer' || t === 'number') return 0;
  if (t === 'boolean') return false;
  if (t === 'array') return [];
  if (t === 'object') return {};
  return '';
}

function ResourceSelectField({
  fieldId,
  name,
  label,
  schema,
  value,
  onChange,
  required,
  description,
  errors,
}: {
  fieldId: string;
  name: string;
  label: string;
  schema: Record<string, unknown>;
  value: unknown;
  onChange: (value: unknown) => void;
  required: boolean;
  description: React.ReactNode;
  errors: React.ReactNode;
}): React.ReactElement {
  const sourceKind = typeof schema['x-source-kind'] === 'string'
    ? schema['x-source-kind']
    : '';
  const valueField = typeof schema['x-value-field'] === 'string'
    ? schema['x-value-field']
    : 'full_name';
  const listId = `${fieldId}-resources`;
  const resourceQuery = useDesignerResources(sourceKind ? [sourceKind] : [], Boolean(sourceKind));
  const resources = resourceQuery.data?.resources ?? [];
  const currentVal = typeof value === 'string' ? value : '';

  return (
    <div className="mb-3.5">
      <label htmlFor={fieldId} className={LABEL_CLASS}>
        {label}
        {required && <span className={REQUIRED_MARK_CLASS}>*</span>}
      </label>
      <input
        id={fieldId}
        type="text"
        list={listId}
        name={name}
        value={currentVal}
        onChange={(e) => onChange(e.target.value)}
        placeholder={resourceQuery.isLoading ? 'Loading resources...' : 'Select or type manually'}
        className={FIELD_INPUT_CLASS}
      />
      <datalist id={listId}>
        {resources.map((resource) => {
          const optionValue = resourceValue(resource, valueField);
          if (!optionValue) return null;
          return (
            <option key={`${resource.kind}:${optionValue}`} value={optionValue}>
              {resourceLabel(resource)}
            </option>
          );
        })}
      </datalist>
      {resourceQuery.isError && (
        <p className="mt-1 text-[11px] text-db-yellow-700">
          Could not load resources. You can still type the value manually.
        </p>
      )}
      {description}
      {errors}
    </div>
  );
}

function resourceValue(resource: DesignerResource, valueField: string): string {
  const direct = resource[valueField as keyof DesignerResource];
  if (typeof direct === 'string' && direct.length > 0) return direct;
  const metadataValue = resource.metadata[valueField];
  if (typeof metadataValue === 'string' && metadataValue.length > 0) return metadataValue;
  return resource.full_name || resource.name || resource.source_id || '';
}

function resourceLabel(resource: DesignerResource): string {
  const details = resource.full_name && resource.full_name !== resource.name
    ? resource.full_name
    : resource.description;
  return details ? `${resource.name} (${details})` : resource.name;
}
