/**
 * FunctionToolNodeEditor — bespoke inspector for a deterministic `tool` node.
 *
 * A tool node references a tool *declaration* (ast.tools) by name and adds
 * per-node wiring (input_mapping / input_literals / output_key / …). A pure
 * SchemaField can't create the declaration, so this store-aware editor composes:
 *   • "Unity Catalog function" mode — UcFunctionPicker (catalog→schema→function);
 *     picking ensures a uc_function declaration exists (dedup by FQN) + points ref.
 *   • "Existing tool" mode — bind ref.name to any already-declared tool (builtin,
 *     mcp, python, registered, uc_function). This keeps non-UC tool nodes editable
 *     rather than forcing every node into the UC picker.
 * Both modes then map the signature params (state ref / literal) + set outputs.
 */

import * as React from 'react';

import type { UcFunctionParam } from '@/api/agentDesigner';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import type { Block } from '@/types/ast';

import { FunctionParamsEditor } from './FunctionParamsEditor';
import { UcFunctionPicker, type UcFunctionValue } from './UcFunctionPicker';

const LABEL = 'mb-1 block font-db-sans text-[12px] font-medium text-db-navy-800';
const SECTION =
  'mb-1.5 mt-4 font-db-sans text-[11px] font-semibold uppercase tracking-[0.06em] text-db-gray-text';
const INPUT =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus';
const SELECT =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2 py-1.5 font-db-sans text-[12px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus';
const CHECK = 'flex cursor-pointer items-center gap-2 font-db-sans text-[12px] text-db-navy-800';
const HINT = 'mt-1 text-[11px] text-db-gray-text';

interface ToolRef {
  type?: string;
  name?: string;
}
interface ToolNodeCfg {
  ref?: ToolRef | string;
  input_mapping?: Record<string, string>;
  input_literals?: Record<string, unknown>;
  output_key?: string;
  output_data_key?: string;
  fail_on_error?: boolean;
  enforce_output_schema?: boolean;
}

type Mode = 'uc_function' | 'existing';

function refName(ref: ToolRef | string | undefined): string {
  return typeof ref === 'string' ? ref : (ref?.name ?? '');
}

export function FunctionToolNodeEditor({
  block,
  selectedPath,
}: {
  block: Block;
  selectedPath: string;
}): React.ReactElement {
  const cfg = (block.config ?? {}) as ToolNodeCfg;
  const tools = useAgentEditorStore((s) => s.ast?.tools);
  const currentRefName = refName(cfg.ref);

  const decl = React.useMemo(
    () => (tools ?? []).find((t) => t.name === currentRefName) ?? null,
    [tools, currentRefName],
  );
  const declKind = decl?.kind;
  const declFn = (decl?.config?.['function'] as string | undefined) ?? '';
  const declParams = (decl?.config?.['params'] as UcFunctionParam[] | undefined) ?? [];

  // Initial mode: an existing binding to a non-uc_function declaration starts in
  // "existing" mode so it stays editable; everything else defaults to the picker.
  const [mode, setMode] = React.useState<Mode>(
    decl !== null && declKind !== 'uc_function' ? 'existing' : 'uc_function',
  );

  const updateConfig = React.useCallback(
    (patch: Partial<ToolNodeCfg>) => {
      const next = { ...(block.config as Record<string, unknown>), ...patch };
      useAgentEditorStore.getState().updateBlock(selectedPath, { config: next });
    },
    [block.config, selectedPath],
  );

  // Ensure a uc_function declaration exists for `fqn` (dedup by FQN); return its name.
  const ensureDeclaration = React.useCallback(
    (
      fqn: string,
      params: UcFunctionParam[] | undefined,
      returnsTable: boolean | undefined,
    ): string => {
      const state = useAgentEditorStore.getState();
      const all = state.ast?.tools ?? [];
      const rt = returnsTable !== undefined ? { returns_table: returnsTable } : {};
      const existing = all.find(
        (t) => t.kind === 'uc_function' && (t.config?.['function'] as string) === fqn,
      );
      if (existing) {
        state.updateTool(existing.name, {
          config: { ...existing.config, function: fqn, ...(params ? { params } : {}), ...rt },
        });
        return existing.name;
      }
      const base = fqn.split('.').pop() || 'uc_function';
      const taken = new Set(all.map((t) => t.name));
      let name = base;
      let i = 1;
      while (taken.has(name)) name = `${base}_${i++}`;
      state.declareTool('uc_function', name, {
        function: fqn,
        params: params ?? [],
        returns_table: returnsTable ?? false,
      });
      return name;
    },
    [],
  );

  const onPick = React.useCallback(
    (value: UcFunctionValue) => {
      if (!value.function) {
        updateConfig({ ref: { name: '' } });
        return;
      }
      const name = ensureDeclaration(value.function, value.params, value.returns_table);
      updateConfig({ ref: { name } });
    },
    [ensureDeclaration, updateConfig],
  );

  const onParamsChange = React.useCallback(
    (next: { inputMapping: Record<string, string>; inputLiterals: Record<string, unknown> }) => {
      updateConfig({ input_mapping: next.inputMapping, input_literals: next.inputLiterals });
    },
    [updateConfig],
  );

  const pickerValue: UcFunctionValue = {
    function: declFn,
    params: declParams,
    returns_table: decl?.config?.['returns_table'] as boolean | undefined,
  };
  const allTools = tools ?? [];

  return (
    <div>
      <p className={SECTION} style={{ marginTop: 0 }}>
        Tool source
      </p>
      <select
        className={SELECT}
        value={mode}
        onChange={(e) => setMode(e.target.value as Mode)}
        aria-label="Tool source"
      >
        <option value="uc_function">Unity Catalog function</option>
        <option value="existing">Existing tool declaration</option>
      </select>

      {mode === 'uc_function' ? (
        <div className="mt-2.5">
          <UcFunctionPicker value={pickerValue} onChange={onPick} />
        </div>
      ) : (
        <div className="mt-2.5">
          <label className={LABEL}>Tool</label>
          <select
            className={SELECT}
            value={currentRefName}
            onChange={(e) => updateConfig({ ref: { name: e.target.value } })}
            aria-label="Existing tool"
          >
            <option value="">— select a declared tool —</option>
            {allTools.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name} ({t.kind})
              </option>
            ))}
          </select>
          {allTools.length === 0 && (
            <p className={HINT}>
              No tools declared yet. Add one from an agent’s Tools tab, or switch to
              “Unity Catalog function” above.
            </p>
          )}
        </div>
      )}

      {currentRefName !== '' && (
        <>
          <p className={SECTION}>Parameters</p>
          {declParams.length === 0 ? (
            <p className={HINT}>
              This tool declares no mappable parameters.
            </p>
          ) : (
            <FunctionParamsEditor
              params={declParams}
              mode="node"
              inputMapping={cfg.input_mapping ?? {}}
              inputLiterals={cfg.input_literals ?? {}}
              onChange={onParamsChange}
            />
          )}
        </>
      )}

      <p className={SECTION}>Output</p>
      <div className="mb-2.5">
        <label className={LABEL}>Output key</label>
        <input
          type="text"
          className={INPUT}
          value={cfg.output_key ?? ''}
          placeholder="tool_result"
          onChange={(e) => updateConfig({ output_key: e.target.value })}
          spellCheck={false}
        />
      </div>
      <div className="mb-2.5">
        <label className={LABEL}>Output data key</label>
        <input
          type="text"
          className={INPUT}
          value={cfg.output_data_key ?? ''}
          placeholder="(optional) structured data + success/error"
          onChange={(e) =>
            updateConfig({ output_data_key: e.target.value === '' ? undefined : e.target.value })
          }
          spellCheck={false}
        />
      </div>
      <label className={`${CHECK} mb-1.5`}>
        <input
          type="checkbox"
          checked={Boolean(cfg.fail_on_error)}
          onChange={(e) => updateConfig({ fail_on_error: e.target.checked })}
          className="h-4 w-4 rounded-sm border-db-gray-lines text-db-lava-600"
        />
        Fail the workflow if this tool errors
      </label>
      <label className={CHECK}>
        <input
          type="checkbox"
          checked={Boolean(cfg.enforce_output_schema)}
          onChange={(e) => updateConfig({ enforce_output_schema: e.target.checked })}
          className="h-4 w-4 rounded-sm border-db-gray-lines text-db-lava-600"
        />
        Enforce output schema (required-keys check)
      </label>
    </div>
  );
}
