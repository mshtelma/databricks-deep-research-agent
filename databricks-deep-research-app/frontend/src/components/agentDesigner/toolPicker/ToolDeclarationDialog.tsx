/**
 * ToolDeclarationDialog — the search-first tool picker (tool-UX plan Phase 1).
 *
 * One dialog backs every "add a tool" surface (workspace registry, agent Tools
 * tab, tool steps). The first screen is a search box over grouped targets —
 * existing workflow tools, live Unity Catalog function search, and tool kinds
 * by family — instead of the old family/kind dropdown classification.
 *
 * Selection rules:
 * - Existing workflow tool → applied to the launch intent as-is (no re-declare).
 * - UC function result / pasted FQN → declared immediately: the live signature
 *   fetch fills `params`/`returns_table` (fail-soft: bare `function` on error)
 *   and the local name derives from the FQN tail (`pct_change`, `pct_change_2`).
 * - Tool kind with no required config → declared immediately with an
 *   auto-generated name.
 * - Tool kind with required config → a configure step (SchemaField form; the
 *   uc-function-picker/code widgets render here too). Local tool name lives
 *   under Advanced, pre-filled and collision-free.
 */

import * as React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { ArrowLeft, Search, Wrench, X as CloseIcon } from 'lucide-react';

import { getUcFunctionSignature } from '@/api/agentDesigner';
import { useUcBrowse, useUcFunctionSearch } from '@/hooks/useDesignerResources';
import { defaultConfigForSchema, requiredConfigErrors, schemaProperties } from '@/lib/jsonSchema';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import type { ToolDecl } from '@/types/ast';
import type { RegistryResponse, ToolKindSpec } from '@/types/agentDesigner';
import { LayerChip } from '../atoms';
import { SchemaField } from '../SchemaField';
import { UcFunctionPicker, type UcFunctionValue } from '../UcFunctionPicker';
import { suggestedToolName, uniqueToolName } from './naming';
import {
  familyForToolKind,
  TOOL_FAMILY_LABELS,
  TOOL_FAMILY_ORDER,
  type ToolFamily,
} from './toolKindFamilies';

export type AddToolIntent = 'workspace' | 'bind-agent' | 'select-tool-step';

export interface ToolDeclarationDialogProps {
  registry: RegistryResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /**
   * Fired when a tool has been declared OR an existing declared tool was
   * selected; the launcher applies its intent (bind / step-ref) in both cases.
   */
  onDeclared?: (tool: ToolDecl) => void;
  /** Launch context: labels the primary action ("Add and enable" / "Add and call"). */
  intent?: AddToolIntent;
  /** Pre-seed the search box on open (e.g. converting a direct ref's FQN). */
  initialQuery?: string;
}

const CTA_LABEL: Record<AddToolIntent, string> = {
  workspace: 'Add tool',
  'bind-agent': 'Add and enable',
  'select-tool-step': 'Add and call',
};

const IDENT = /^[a-z0-9_]+$/;
const FQN = /^[a-z0-9_]+\.[a-z0-9_]+\.[a-z0-9_]+$/;

type Stage =
  | { kind: 'search' }
  | { kind: 'configure'; spec: ToolKindSpec }
  | { kind: 'uc_browse' };

/** Map the free-text query (+ optional catalog scope) onto a UC search request. */
function parseUcQuery(
  query: string,
  scopeCatalog: string,
): { parent?: string; prefix: string } {
  const text = query.trim().toLowerCase();
  if (text.includes('.')) {
    const parts = text.split('.');
    const [first = '', second = ''] = parts;
    if (parts.length >= 3 && IDENT.test(first) && IDENT.test(second)) {
      return { parent: `${first}.${second}`, prefix: parts.slice(2).join('.') };
    }
    if (parts.length === 2 && IDENT.test(first)) {
      return { parent: first, prefix: second };
    }
  }
  const scope = scopeCatalog.trim().toLowerCase();
  return { parent: IDENT.test(scope) ? scope : undefined, prefix: text };
}

const GROUP_HEADING =
  'mb-1 mt-3 font-db-sans text-[11px] font-semibold uppercase tracking-[0.06em] text-db-gray-text';
const ROW_BUTTON =
  'flex w-full items-center gap-2.5 rounded-db-md border border-transparent px-2.5 py-2 text-left transition-colors hover:border-db-navy-300 hover:bg-db-oat-light';
const HINT = 'mt-1 text-[11px] leading-[1.45] text-db-gray-text';

export function ToolDeclarationDialog({
  registry,
  open,
  onOpenChange,
  onDeclared,
  intent = 'workspace',
  initialQuery,
}: ToolDeclarationDialogProps): React.ReactElement {
  const declaredTools = useAgentEditorStore((s) => s.ast?.tools) ?? [];

  const [stage, setStage] = React.useState<Stage>({ kind: 'search' });
  const [query, setQuery] = React.useState('');
  const [family, setFamily] = React.useState<ToolFamily | 'all'>('all');
  const [scopeCatalog, setScopeCatalog] = React.useState('');
  const [declaringFqn, setDeclaringFqn] = React.useState<string | null>(null);

  // Configure-stage state.
  const [toolName, setToolName] = React.useState('');
  const [config, setConfig] = React.useState<Record<string, unknown>>({});
  const [configErrors, setConfigErrors] = React.useState<Record<string, string[]>>({});
  const [nameError, setNameError] = React.useState<string | null>(null);

  // Browse-stage state.
  const [ucValue, setUcValue] = React.useState<UcFunctionValue>({ function: '' });

  React.useEffect(() => {
    if (!open) return;
    setStage({ kind: 'search' });
    setQuery(initialQuery ?? '');
    setFamily('all');
    setScopeCatalog('');
    setDeclaringFqn(null);
    setNameError(null);
    setUcValue({ function: '' });
  }, [open, initialQuery]);

  const deferredQuery = React.useDeferredValue(query);
  const normalizedQuery = deferredQuery.trim().toLowerCase();
  const ucEnabled = open && (family === 'all' || family === 'databricks');
  const { parent: ucParent, prefix: ucPrefix } = parseUcQuery(
    normalizedQuery,
    scopeCatalog,
  );
  const catalogsQuery = useUcBrowse('uc_catalog', undefined, ucEnabled);
  const ucSearch = useUcFunctionSearch(ucParent, ucPrefix, ucEnabled);
  const pastedFqn = FQN.test(normalizedQuery) ? normalizedQuery : null;

  const availableFamilies = TOOL_FAMILY_ORDER.filter((f) =>
    registry.tool_kinds.some((spec) => familyForToolKind(spec) === f),
  );

  const matchingExisting = declaredTools
    .filter((tool) => {
      if (!normalizedQuery) return true;
      return (
        tool.name.toLowerCase().includes(normalizedQuery) ||
        tool.kind.toLowerCase().includes(normalizedQuery)
      );
    })
    .slice(0, 8);

  const matchingKinds = registry.tool_kinds.filter((spec) => {
    if (family !== 'all' && familyForToolKind(spec) !== family) return false;
    if (!normalizedQuery) return true;
    return (
      spec.kind.toLowerCase().includes(normalizedQuery) ||
      spec.label.toLowerCase().includes(normalizedQuery)
    );
  });

  const finishWith = React.useCallback(
    (tool: ToolDecl) => {
      onDeclared?.(tool);
      onOpenChange(false);
    },
    [onDeclared, onOpenChange],
  );

  const declareNow = React.useCallback(
    (kind: string, base: string, cfg: Record<string, unknown>): boolean => {
      const name = uniqueToolName(base, useAgentEditorStore.getState().ast?.tools ?? []);
      const ok = useAgentEditorStore.getState().declareTool(kind, name, cfg);
      if (ok) finishWith({ kind, name, config: cfg });
      return ok;
    },
    [finishWith],
  );

  const declareUcFunction = React.useCallback(
    async (fqn: string, preset?: Pick<UcFunctionValue, 'params' | 'returns_table'>) => {
      setDeclaringFqn(fqn);
      const cfg: Record<string, unknown> = { function: fqn };
      try {
        if (preset?.params !== undefined || preset?.returns_table !== undefined) {
          if (preset.params?.length) cfg['params'] = preset.params;
          if (preset.returns_table) cfg['returns_table'] = true;
        } else {
          const sig = await getUcFunctionSignature(fqn);
          if (sig.scalar && sig.params.length > 0) cfg['params'] = sig.params;
          if (sig.returns_table) cfg['returns_table'] = true;
        }
      } catch {
        // Fail-soft (plan: manual fallback everywhere): declare with the bare
        // FQN; save-time introspection fills params when the warehouse allows.
      }
      declareNow('uc_function', suggestedToolName('uc_function', fqn), cfg);
      setDeclaringFqn(null);
    },
    [declareNow],
  );

  const openConfigure = React.useCallback((spec: ToolKindSpec) => {
    setStage({ kind: 'configure', spec });
    setToolName('');
    setConfig(defaultConfigForSchema(spec.config_schema));
    setConfigErrors({});
    setNameError(null);
  }, []);

  const handleKindClick = React.useCallback(
    (spec: ToolKindSpec) => {
      const required = Array.isArray(spec.config_schema?.['required'])
        ? (spec.config_schema['required'] as string[])
        : [];
      if (required.length === 0) {
        declareNow(
          spec.kind,
          suggestedToolName(spec.kind),
          defaultConfigForSchema(spec.config_schema),
        );
        return;
      }
      openConfigure(spec);
    },
    [declareNow, openConfigure],
  );

  const handleConfigureSubmit = React.useCallback(() => {
    if (stage.kind !== 'configure') return;
    const spec = stage.spec;
    const errors = requiredConfigErrors(spec.config_schema ?? null, config);
    setConfigErrors(errors);
    if (Object.keys(errors).length > 0) return;

    const targetValue =
      typeof config['function'] === 'string'
        ? (config['function'] as string)
        : typeof config['import'] === 'string'
          ? (config['import'] as string)
          : undefined;
    const base = toolName.trim() || suggestedToolName(spec.kind, targetValue);
    const explicit = toolName.trim().length > 0;
    const existing = useAgentEditorStore.getState().ast?.tools ?? [];
    const name = explicit ? base : uniqueToolName(base, existing);
    const ok = useAgentEditorStore.getState().declareTool(spec.kind, name, config);
    if (!ok) {
      setNameError('Tool name already exists');
      return;
    }
    finishWith({ kind: spec.kind, name, config });
  }, [stage, config, toolName, finishWith]);

  const ctaLabel = CTA_LABEL[intent];

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-db-navy-900/30 backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          className="db-root fixed left-1/2 top-1/2 z-50 flex max-h-[82vh] w-full max-w-[680px] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-db-lg border border-db-gray-lines bg-white font-db-sans shadow-db-xl focus:outline-none"
          aria-describedby={undefined}
        >
          <div className="border-b border-db-gray-lines px-[22px] py-[18px]">
            <div className="flex items-center gap-2.5">
              <Wrench size={18} className="text-db-navy-800" />
              <DialogPrimitive.Title className="text-[18px] font-medium text-db-navy-800">
                Add tool
              </DialogPrimitive.Title>
              <DialogPrimitive.Close
                className="ml-auto rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
                aria-label="Close"
              >
                <CloseIcon size={14} />
              </DialogPrimitive.Close>
            </div>
            <p className="mt-1.5 text-[13px] text-db-gray-text">
              Search functions, indexes, spaces and built-in tools, or pick an existing
              workflow tool.
            </p>
          </div>

          {stage.kind === 'search' && (
            <>
              <div className="border-b border-db-gray-lines px-[22px] py-3">
                <div className="relative">
                  <Search
                    size={14}
                    className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-db-gray-text"
                  />
                  <input
                    aria-label="Search tools"
                    autoFocus
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search tools, UC functions, indexes… or paste catalog.schema.function"
                    className="w-full rounded-db-md border border-db-gray-lines bg-white py-1.5 pl-8 pr-2.5 text-[13px] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus"
                  />
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(['all', ...availableFamilies] as const).map((f) => (
                    <button
                      key={f}
                      type="button"
                      aria-pressed={family === f}
                      onClick={() => setFamily(f)}
                      className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                        family === f
                          ? 'border-db-navy-800 bg-db-navy-800 text-white'
                          : 'border-db-gray-lines bg-white text-db-navy-800 hover:bg-db-oat-medium'
                      }`}
                    >
                      {f === 'all' ? 'All' : TOOL_FAMILY_LABELS[f]}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto px-[22px] pb-4">
                {matchingExisting.length > 0 && (
                  <>
                    <div className={GROUP_HEADING}>Existing workflow tools</div>
                    <ul>
                      {matchingExisting.map((tool) => (
                        <li key={tool.name}>
                          <button
                            type="button"
                            className={ROW_BUTTON}
                            onClick={() => finishWith(tool)}
                          >
                            <span className="min-w-0 flex-1">
                              <span className="block truncate font-db-mono text-[12px] font-medium text-db-navy-800">
                                {tool.name}
                              </span>
                              <span className="block truncate text-[10px] text-db-gray-text">
                                {tool.kind}
                              </span>
                            </span>
                            <span className="shrink-0 text-[10px] uppercase tracking-[0.05em] text-db-gray-text">
                              in workflow
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                {ucEnabled && (
                  <>
                    <div className={GROUP_HEADING}>Unity Catalog functions</div>
                    <div className="mb-1.5 flex items-center gap-2">
                      <input
                        aria-label="Catalog to search"
                        list="tool-picker-catalogs"
                        value={scopeCatalog}
                        onChange={(event) => setScopeCatalog(event.target.value)}
                        placeholder="catalog to search"
                        className="w-[200px] rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus"
                      />
                      <datalist id="tool-picker-catalogs">
                        {(catalogsQuery.data?.resources ?? []).map((resource) => (
                          <option key={resource.name} value={resource.name} />
                        ))}
                      </datalist>
                      <button
                        type="button"
                        onClick={() => setStage({ kind: 'uc_browse' })}
                        className="rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
                      >
                        Browse Unity Catalog…
                      </button>
                    </div>

                    {pastedFqn && (
                      <button
                        type="button"
                        className={ROW_BUTTON}
                        disabled={declaringFqn !== null}
                        onClick={() => void declareUcFunction(pastedFqn)}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-db-mono text-[12px] font-medium text-db-navy-800">
                            Use {pastedFqn}
                          </span>
                          <span className="block text-[10px] text-db-gray-text">
                            {declaringFqn === pastedFqn
                              ? 'Fetching signature…'
                              : 'Declare this Unity Catalog function'}
                          </span>
                        </span>
                      </button>
                    )}

                    {ucParent ? (
                      <>
                        <ul>
                          {(ucSearch.data?.resources ?? [])
                            .filter((r) => r.full_name && r.full_name !== pastedFqn)
                            .slice(0, 25)
                            .map((resource) => (
                              <li key={resource.full_name}>
                                <button
                                  type="button"
                                  className={ROW_BUTTON}
                                  disabled={declaringFqn !== null}
                                  onClick={() =>
                                    void declareUcFunction(resource.full_name ?? '')
                                  }
                                >
                                  <span className="min-w-0 flex-1">
                                    <span className="block truncate font-db-mono text-[12px] font-medium text-db-navy-800">
                                      {resource.name}
                                    </span>
                                    <span className="block truncate font-db-mono text-[10px] text-db-gray-text">
                                      {declaringFqn === resource.full_name
                                        ? 'Fetching signature…'
                                        : resource.full_name}
                                    </span>
                                  </span>
                                </button>
                              </li>
                            ))}
                        </ul>
                        {ucSearch.isFetching && (
                          <p className={HINT}>Searching {ucParent}…</p>
                        )}
                        {!ucSearch.isFetching &&
                          (ucSearch.data?.resources ?? []).length === 0 &&
                          !pastedFqn && (
                            <p className={HINT}>
                              No matching functions in {ucParent}. Paste a full
                              catalog.schema.function to add it manually.
                            </p>
                          )}
                        {ucSearch.data?.warning && (
                          <p className={HINT} role="status">
                            {ucSearch.data.warning}
                          </p>
                        )}
                        {ucSearch.data?.error && (
                          <p className={HINT} role="status">
                            {ucSearch.data.error.code === 'permission'
                              ? 'You lack BROWSE on this scope — paste a full function name instead.'
                              : ucSearch.data.error.message}
                          </p>
                        )}
                      </>
                    ) : (
                      !pastedFqn && (
                        <p className={HINT}>
                          Pick a catalog to search functions, paste a full
                          catalog.schema.function, or browse.
                        </p>
                      )
                    )}
                  </>
                )}

                {matchingKinds.length > 0 && (
                  <>
                    <div className={GROUP_HEADING}>Tool kinds</div>
                    <ul>
                      {matchingKinds.map((spec) => (
                        <li key={spec.kind}>
                          <button
                            type="button"
                            className={ROW_BUTTON}
                            onClick={() => handleKindClick(spec)}
                          >
                            <LayerChip layer={spec.layer ?? 'A'} />
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-[12px] font-medium text-db-navy-800">
                                {spec.label}
                              </span>
                              <span className="block truncate font-db-mono text-[10px] text-db-gray-text">
                                {spec.kind}
                              </span>
                            </span>
                            <span className="shrink-0 text-[10px] text-db-gray-text">
                              {TOOL_FAMILY_LABELS[familyForToolKind(spec)]}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                {matchingExisting.length === 0 &&
                  matchingKinds.length === 0 &&
                  !ucEnabled && (
                    <p className={`${HINT} mt-4`}>No tools match “{query}”.</p>
                  )}
              </div>
            </>
          )}

          {stage.kind === 'configure' && (
            <>
              <div className="flex-1 overflow-y-auto px-[22px] py-4">
                <div className="mb-3 flex items-center gap-2">
                  <button
                    type="button"
                    aria-label="Back to search"
                    onClick={() => setStage({ kind: 'search' })}
                    className="rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
                  >
                    <ArrowLeft size={14} />
                  </button>
                  <LayerChip layer={stage.spec.layer ?? 'A'} />
                  <div className="min-w-0">
                    <div className="truncate text-[12px] font-medium text-db-navy-800">
                      {stage.spec.label}
                    </div>
                    <div className="truncate font-db-mono text-[10px] text-db-gray-text">
                      {stage.spec.kind}
                    </div>
                  </div>
                </div>

                {Object.entries(schemaProperties(stage.spec.config_schema ?? null)).map(
                  ([fieldName, fieldSchema]) => (
                    <SchemaField
                      key={fieldName}
                      name={fieldName}
                      schema={fieldSchema}
                      value={config[fieldName]}
                      onChange={(v) => {
                        setConfig((prev) => ({ ...prev, [fieldName]: v }));
                        setConfigErrors((prev) => ({ ...prev, [fieldName]: [] }));
                      }}
                      errors={configErrors[fieldName] ?? []}
                    />
                  ),
                )}

                <details className="mt-3 rounded-db-md border border-db-gray-lines bg-white">
                  <summary className="cursor-pointer select-none px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-db-navy-700 hover:bg-db-oat-light">
                    Advanced
                  </summary>
                  <div className="border-t border-db-gray-lines px-3 py-3">
                    <label
                      htmlFor="tool-name-input"
                      className="mb-1 block text-[12px] font-medium text-db-navy-800"
                    >
                      Local tool name
                    </label>
                    <input
                      id="tool-name-input"
                      type="text"
                      value={toolName}
                      onChange={(event) => {
                        setToolName(event.target.value);
                        setNameError(null);
                      }}
                      placeholder={uniqueToolName(
                        suggestedToolName(
                          stage.spec.kind,
                          typeof config['function'] === 'string'
                            ? (config['function'] as string)
                            : typeof config['import'] === 'string'
                              ? (config['import'] as string)
                              : undefined,
                        ),
                        declaredTools,
                      )}
                      className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus"
                    />
                    <p className={HINT}>
                      How agents and tool steps refer to this tool. Leave blank for an
                      auto-generated name.
                    </p>
                    {nameError !== null && (
                      <p className="mt-1 text-[11px] text-db-lava-700" role="alert">
                        {nameError}
                      </p>
                    )}
                  </div>
                </details>
              </div>

              <div className="flex justify-end gap-2 border-t border-db-gray-lines px-[22px] py-3.5">
                <DialogPrimitive.Close asChild>
                  <button
                    type="button"
                    className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
                  >
                    Cancel
                  </button>
                </DialogPrimitive.Close>
                <button
                  type="button"
                  onClick={handleConfigureSubmit}
                  className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700"
                >
                  {ctaLabel}
                </button>
              </div>
            </>
          )}

          {stage.kind === 'uc_browse' && (
            <>
              <div className="flex-1 overflow-y-auto px-[22px] py-4">
                <div className="mb-3 flex items-center gap-2">
                  <button
                    type="button"
                    aria-label="Back to search"
                    onClick={() => setStage({ kind: 'search' })}
                    className="rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
                  >
                    <ArrowLeft size={14} />
                  </button>
                  <span className="text-[12px] font-medium text-db-navy-800">
                    Browse Unity Catalog
                  </span>
                </div>
                <UcFunctionPicker value={ucValue} onChange={setUcValue} />
              </div>
              <div className="flex justify-end gap-2 border-t border-db-gray-lines px-[22px] py-3.5">
                <DialogPrimitive.Close asChild>
                  <button
                    type="button"
                    className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
                  >
                    Cancel
                  </button>
                </DialogPrimitive.Close>
                <button
                  type="button"
                  disabled={!FQN.test(ucValue.function.trim().toLowerCase()) || declaringFqn !== null}
                  onClick={() =>
                    void declareUcFunction(ucValue.function.trim().toLowerCase(), {
                      params: ucValue.params,
                      returns_table: ucValue.returns_table,
                    })
                  }
                  className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {ctaLabel}
                </button>
              </div>
            </>
          )}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
