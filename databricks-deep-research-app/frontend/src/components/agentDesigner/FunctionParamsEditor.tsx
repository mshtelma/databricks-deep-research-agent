/**
 * FunctionParamsEditor — renders a function's signature parameters as rows.
 *
 * Node mode (deterministic tool node): each parameter binds to exactly ONE
 * source — a workflow-state reference (→ input_mapping) OR a literal constant
 * (→ input_literals). The two dicts are kept disjoint (mirrors the backend
 * ToolNodeConfig._no_arg_collisions validator). Readonly mode just lists the
 * signature (used on agent/declaration surfaces where the LLM fills args).
 */

import * as React from 'react';

import type { UcFunctionParam } from '@/api/agentDesigner';

const LABEL = 'font-db-sans text-[12px] font-medium text-db-navy-800';
const TYPE_BADGE =
  'ml-1 rounded-sm bg-db-oat-medium px-1 py-0.5 font-db-mono text-[10px] text-db-gray-text';
const REQUIRED = 'ml-0.5 text-db-lava-600';
const SELECT =
  'rounded-db-md border border-db-gray-lines bg-white px-2 py-1 font-db-sans text-[12px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus';
const INPUT =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

export type ParamSource = 'state' | 'literal';

export interface FunctionParamsEditorProps {
  params: UcFunctionParam[];
  mode?: 'node' | 'readonly';
  inputMapping?: Record<string, string>;
  inputLiterals?: Record<string, unknown>;
  onChange?: (next: {
    inputMapping: Record<string, string>;
    inputLiterals: Record<string, unknown>;
  }) => void;
}

function sourceOf(name: string, literals: Record<string, unknown>): ParamSource {
  // Default 'state'; an arg present in input_literals is a literal. The two dicts
  // are kept disjoint, so membership in literals is the only disambiguation needed.
  return name in literals ? 'literal' : 'state';
}

export function FunctionParamsEditor({
  params,
  mode = 'node',
  inputMapping = {},
  inputLiterals = {},
  onChange,
}: FunctionParamsEditorProps): React.ReactElement {
  if (params.length === 0) {
    return <p className="text-[11px] italic text-db-gray-text">No parameters.</p>;
  }

  if (mode === 'readonly' || !onChange) {
    return (
      <ul className="space-y-1">
        {params.map((p) => (
          <li key={p.name} className="text-[12px] text-db-navy-800">
            <span className="font-db-mono">{p.name}</span>
            <span className={TYPE_BADGE}>{p.type}</span>
            {p.required && <span className={REQUIRED}>*</span>}
          </li>
        ))}
      </ul>
    );
  }

  const setParam = (
    name: string,
    source: ParamSource,
    value: string,
  ): void => {
    // Keep the two dicts disjoint: an arg lives in exactly one (or neither).
    const nextMapping = { ...inputMapping };
    const nextLiterals = { ...inputLiterals };
    delete nextMapping[name];
    delete nextLiterals[name];
    if (value.trim() !== '') {
      if (source === 'state') nextMapping[name] = value;
      else nextLiterals[name] = value;
    }
    onChange({ inputMapping: nextMapping, inputLiterals: nextLiterals });
  };

  return (
    <div className="space-y-2">
      {params.map((p) => {
        const source = sourceOf(p.name, inputLiterals);
        const value =
          source === 'literal'
            ? String(inputLiterals[p.name] ?? '')
            : (inputMapping[p.name] ?? '');
        return (
          <div key={p.name} className="rounded-db-md border border-db-gray-lines bg-db-oat-light p-2">
            <div className="mb-1 flex items-center">
              <span className={`${LABEL} font-db-mono`}>{p.name}</span>
              <span className={TYPE_BADGE}>{p.type}</span>
              {p.required && <span className={REQUIRED}>*</span>}
            </div>
            <div className="flex items-center gap-2">
              <select
                className={SELECT}
                value={source}
                onChange={(e) => setParam(p.name, e.target.value as ParamSource, value)}
                aria-label={`${p.name} source`}
              >
                <option value="state">state ref</option>
                <option value="literal">literal</option>
              </select>
              <input
                type="text"
                className={INPUT}
                value={value}
                placeholder={
                  source === 'state' ? 'workflow state key (e.g. planner.tickers)' : 'constant value'
                }
                onChange={(e) => setParam(p.name, source, e.target.value)}
                spellCheck={false}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
