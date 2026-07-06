/**
 * UcFunctionPicker — cascading, search-and-type Unity Catalog function picker.
 *
 * Three dependent datalist comboboxes (catalog → schema → function), each backed
 * by an OBO-scoped browse query (useUcBrowse). Changing a parent clears its
 * children; pasting a full `catalog.schema.function` into any field back-fills
 * all three. On a complete, valid FQN the chosen function's signature is fetched
 * (useUcFunctionSignature) and handed upward so the parent can auto-map params.
 *
 * Reusable + surface-agnostic: emits `{ function, params }` via onChange. Used by
 * both the Tool-node inspector (deterministic) and the declaration editors.
 */

import * as React from 'react';

import type { UcFunctionParam } from '@/api/agentDesigner';
import { useUcBrowse, useUcFunctionSignature } from '@/hooks/useDesignerResources';

const FIELD_INPUT =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus disabled:bg-db-oat-light disabled:text-db-gray-text';
const LABEL = 'mb-1 block font-db-sans text-[12px] font-medium text-db-navy-800';
const HINT = 'mt-1 text-[11px] text-db-gray-text';
const WARN = 'mt-1 text-[11px] text-db-yellow-700';

// Mirror the backend guards: a UC function FQN is three [A-Za-z0-9_] parts; a
// single segment is a legal identifier. Hyphenated catalogs are a v1 limitation.
const FQN_RE = /^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$/;
const IDENT_RE = /^[A-Za-z0-9_]*$/;

export interface UcFunctionValue {
  function: string;
  params?: UcFunctionParam[];
}

export interface UcFunctionSignatureInfo {
  params: UcFunctionParam[];
  scalar: boolean;
  warning?: string | null;
}

export interface UcFunctionPickerProps {
  value: UcFunctionValue;
  onChange: (value: UcFunctionValue) => void;
  /** Fired when a signature is fetched, so the parent can render param mapping. */
  onSignature?: (sig: UcFunctionSignatureInfo) => void;
  disabled?: boolean;
}

function splitFqn(fqn: string): { catalog: string; schema: string; fn: string } {
  const parts = (fqn || '').split('.');
  return { catalog: parts[0] ?? '', schema: parts[1] ?? '', fn: parts[2] ?? '' };
}

/** If a field receives a pasted 3-part FQN, return its parts so we can back-fill. */
function asFullFqn(raw: string): { catalog: string; schema: string; fn: string } | null {
  const trimmed = raw.trim();
  return FQN_RE.test(trimmed) ? splitFqn(trimmed) : null;
}

export function UcFunctionPicker({
  value,
  onChange,
  onSignature,
  disabled = false,
}: UcFunctionPickerProps): React.ReactElement {
  const idBase = React.useId();
  const initial = splitFqn(value.function ?? '');
  const [catalog, setCatalog] = React.useState(initial.catalog);
  const [schema, setSchema] = React.useState(initial.schema);
  const [fn, setFn] = React.useState(initial.fn);

  // Re-sync the three inputs when the external function changes (node switch /
  // programmatic set) but not on our own emits (guarded by the value ref).
  const lastValueFn = React.useRef(value.function);
  React.useEffect(() => {
    if (value.function !== lastValueFn.current) {
      lastValueFn.current = value.function;
      const s = splitFqn(value.function ?? '');
      setCatalog(s.catalog);
      setSchema(s.schema);
      setFn(s.fn);
    }
  }, [value.function]);

  const catalogsQ = useUcBrowse('uc_catalog', undefined, catalog, !disabled);
  const schemasQ = useUcBrowse('uc_schema', catalog || undefined, schema, !disabled);
  const functionsQ = useUcBrowse(
    'uc_function',
    catalog && schema ? `${catalog}.${schema}` : undefined,
    fn,
    !disabled,
  );

  const fqn = catalog && schema && fn ? `${catalog}.${schema}.${fn}` : '';
  const fqnValid = FQN_RE.test(fqn);
  const unsupported = Boolean((catalog || schema || fn)) &&
    !(IDENT_RE.test(catalog) && IDENT_RE.test(schema) && IDENT_RE.test(fn));

  // Emit on user action (not in a derived effect) to avoid render loops.
  const emit = React.useCallback(
    (c: string, s: string, f: string) => {
      const next = c && s && f ? `${c}.${s}.${f}` : '';
      lastValueFn.current = next;
      // Keep existing params only if the FQN is unchanged; otherwise clear until
      // the signature for the new function arrives.
      onChange({ function: next, params: next === value.function ? value.params : undefined });
    },
    [onChange, value.function, value.params],
  );

  const onCatalog = (raw: string): void => {
    const full = asFullFqn(raw);
    if (full) {
      setCatalog(full.catalog);
      setSchema(full.schema);
      setFn(full.fn);
      emit(full.catalog, full.schema, full.fn);
      return;
    }
    setCatalog(raw);
    setSchema('');
    setFn('');
    emit(raw, '', '');
  };
  const onSchema = (raw: string): void => {
    setSchema(raw);
    setFn('');
    emit(catalog, raw, '');
  };
  const onFn = (raw: string): void => {
    const full = asFullFqn(raw);
    if (full) {
      setCatalog(full.catalog);
      setSchema(full.schema);
      setFn(full.fn);
      emit(full.catalog, full.schema, full.fn);
      return;
    }
    setFn(raw);
    emit(catalog, schema, raw);
  };

  const sigQ = useUcFunctionSignature(fqnValid && !unsupported ? fqn : undefined, !disabled);

  // Push params up once per resolved function (ref-guarded so re-renders don't loop).
  const emittedSig = React.useRef<string>('');
  React.useEffect(() => {
    const data = sigQ.data;
    if (data && data.function === fqn && emittedSig.current !== data.function) {
      emittedSig.current = data.function;
      onChange({ function: fqn, params: data.params });
      onSignature?.({ params: data.params, scalar: data.scalar, warning: data.warning });
    }
  }, [sigQ.data, fqn, onChange, onSignature]);

  const level = (
    label: string,
    listId: string,
    val: string,
    onVal: (v: string) => void,
    options: Array<{ name: string; description?: string | null }>,
    isLoading: boolean,
    enabled: boolean,
  ): React.ReactElement => (
    <div className="mb-2.5">
      <label className={LABEL}>{label}</label>
      <input
        type="text"
        list={listId}
        value={val}
        disabled={disabled || !enabled}
        onChange={(e) => onVal(e.target.value)}
        placeholder={
          !enabled ? 'select the level above first' : isLoading ? 'Loading…' : 'search or type'
        }
        className={FIELD_INPUT}
        spellCheck={false}
        autoComplete="off"
      />
      <datalist id={listId}>
        {options.map((o) => (
          <option key={o.name} value={o.name}>
            {o.description ?? ''}
          </option>
        ))}
      </datalist>
    </div>
  );

  return (
    <div>
      {level(
        'Catalog',
        `${idBase}-cat`,
        catalog,
        onCatalog,
        catalogsQ.data?.resources ?? [],
        catalogsQ.isLoading,
        true,
      )}
      {level(
        'Schema',
        `${idBase}-sch`,
        schema,
        onSchema,
        schemasQ.data?.resources ?? [],
        schemasQ.isLoading,
        Boolean(catalog),
      )}
      {level(
        'Function',
        `${idBase}-fn`,
        fn,
        onFn,
        functionsQ.data?.resources ?? [],
        functionsQ.isLoading,
        Boolean(catalog && schema),
      )}

      {unsupported && (
        <p className={WARN}>
          Only simple names (letters, digits, underscore) are supported — hyphenated
          catalogs/schemas can’t be used as tool functions in this version.
        </p>
      )}
      {fqnValid && !unsupported && (
        <p className={HINT}>
          ✓ <span className="font-db-mono">{fqn}</span>
          {sigQ.isLoading
            ? ' · loading signature…'
            : sigQ.data
              ? sigQ.data.scalar
                ? ` · ${sigQ.data.params.length} parameter${sigQ.data.params.length === 1 ? '' : 's'}`
                : ' · non-scalar parameters (map manually)'
              : ''}
        </p>
      )}
      {sigQ.data?.warning && <p className={WARN}>{sigQ.data.warning}</p>}
      {(catalogsQ.isError || schemasQ.isError || functionsQ.isError) && (
        <p className={WARN}>
          Could not browse Unity Catalog (a SQL warehouse may be needed). You can still
          type a full <span className="font-db-mono">catalog.schema.function</span> above.
        </p>
      )}
    </div>
  );
}
