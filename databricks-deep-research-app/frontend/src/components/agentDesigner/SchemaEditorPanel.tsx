/**
 * First-class, editable Schema panel for the Designer.
 *
 * Presents an agent's input fields and output slots (the `definition.surface`
 * derived via `schemaModel`) as a structured, editable form. Edits are applied
 * live to the AST (`setAst`), so the Preview tab reflects them and the existing
 * Save flow (with `validate_surface` as the write gate) persists them.
 *
 * Mounted only while the Schema tab is active, so it re-derives from the current
 * surface on each entry (picking up edits made in the Edit tab / co-pilot).
 */
import * as React from 'react';

import { useAgentEditorStore } from '@/stores/agentEditorStore';
import {
  surfaceToSchema,
  applySchemaToSurface,
  type EditableSchema,
  type EditableSlot,
  type SlotKind,
  type ColumnType,
  type InputKind,
} from '@/lib/schemaModel';
import { extractSurfaceFromAgentDefinition } from '@/lib/agentSurface';
import type { AST } from '@/types/ast';
import type { Surface, SurfaceControlPolicy, SurfaceRuntimeControls } from '@/types/surface';

const INPUT_CLASS =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-sans text-[13px] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';
const LABEL_CLASS = 'mb-1 block text-[12px] font-medium text-db-navy-800';
const SECTION_CLASS =
  'rounded-db-md border border-db-gray-lines bg-white p-4';
const GHOST_BTN =
  'rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium';
const REMOVE_BTN =
  'rounded px-2 py-1 text-[11px] font-medium text-db-lava-700 hover:bg-db-lava-50';

const INPUT_KINDS: InputKind[] = ['TextField', 'TextArea', 'Select', 'Checkbox'];
const SLOT_KINDS: { value: SlotKind; label: string }[] = [
  { value: 'table', label: 'Table' },
  { value: 'metrics', label: 'Metric grid' },
  { value: 'findings', label: 'Findings list' },
];
const COLUMN_TYPES: ColumnType[] = ['string', 'number', 'date'];
const CONTROL_POLICIES: SurfaceControlPolicy[] = ['show', 'advanced', 'locked', 'hide'];
const RUN_CONTROL_LABELS: Record<keyof SurfaceRuntimeControls, string> = {
  effort: 'Effort',
  sources: 'Sources',
  verify_sources: 'Verify citations',
  plan_review: 'Plan review',
  report_style: 'Report style',
  cross_session_memory: 'Memory',
  live_search: 'Live search',
};

function surfaceOf(ast: AST | null): Surface | null {
  return extractSurfaceFromAgentDefinition(ast);
}

function nextName(prefix: string, taken: Set<string>): string {
  let i = 1;
  while (taken.has(`${prefix}${i}`)) i += 1;
  return `${prefix}${i}`;
}

export function SchemaEditorPanel(): React.ReactElement {
  const ast = useAgentEditorStore((s) => s.ast);
  const surface = React.useMemo(() => surfaceOf(ast), [ast]);
  const derivedSchema = React.useMemo(() => surfaceToSchema(surface), [surface]);
  const [schema, setSchema] = React.useState<EditableSchema>(derivedSchema);

  React.useEffect(() => {
    setSchema(derivedSchema);
  }, [derivedSchema]);

  // Apply the full edited schema against the LATEST surface each change — this
  // is idempotent (reconciles by slot/pointer) so repeated commits never
  // duplicate components, and it stays correct even as the store updates.
  const commit = React.useCallback((next: EditableSchema) => {
    setSchema(next);
    const cur = useAgentEditorStore.getState().ast;
    if (!cur) return;
    const curSurface =
      surfaceOf(cur) ?? ({ version: 1, components: [], data_model: {}, bindings: [] } as unknown as Surface);
    const nextSurface = applySchemaToSurface(curSurface, next);
    useAgentEditorStore.getState().setAst({ ...cur, surface: nextSurface });
  }, []);

  const hasSurface = surface !== null && !!schema.action;
  const hasDataCapability =
    (ast?.tools?.length ?? 0) > 0 || (ast?.sources?.length ?? 0) > 0;

  if (!hasSurface) {
    return (
      <div className={SECTION_CLASS}>
        <p className="text-[13px] text-db-gray-text">
          This agent has no UI surface yet. Add one from the co-pilot ("give this
          agent a UI") or the Edit tab.
        </p>
      </div>
    );
  }

  const slotNames = new Set(schema.slots.map((s) => s.name));
  const inputKeys = new Set(schema.inputs.map((i) => i.key));

  return (
    <div className="flex flex-col gap-5">
      {schema.slots.length > 0 && !hasDataCapability && (
        <div className="rounded-db-md border border-amber-200 bg-amber-50 px-4 py-2 text-[12px] text-amber-800">
          This agent has output sections but no research tool or data source, so
          they may stay empty. Add a tool/source (Edit tab) so the agent can
          gather the data to fill them.
        </div>
      )}

      {/* Inputs -------------------------------------------------------------- */}
      <section className={SECTION_CLASS}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-[13px] font-semibold text-db-navy-800">Inputs</h3>
          <button
            type="button"
            className={GHOST_BTN}
            onClick={() => {
              const key = nextName('field_', inputKeys);
              commit({
                ...schema,
                inputs: [
                  ...schema.inputs,
                  { id: `field_${key}`, component: 'TextField', label: 'New field', key },
                ],
              });
            }}
          >
            + Add input
          </button>
        </div>
        {schema.inputs.length === 0 && (
          <p className="text-[12px] text-db-gray-text">No input fields.</p>
        )}
        <div className="flex flex-col gap-2">
          {schema.inputs.map((input, i) => (
            <div key={input.id} className="flex items-end gap-2">
              <div className="flex-1">
                <label className={LABEL_CLASS}>Label</label>
                <input
                  className={INPUT_CLASS}
                  value={input.label}
                  onChange={(e) =>
                    commit({
                      ...schema,
                      inputs: schema.inputs.map((x, j) =>
                        j === i ? { ...x, label: e.target.value } : x,
                      ),
                    })
                  }
                />
              </div>
              <div className="w-40">
                <label className={LABEL_CLASS}>Key</label>
                <input
                  className={INPUT_CLASS}
                  value={input.key}
                  onChange={(e) =>
                    commit({
                      ...schema,
                      inputs: schema.inputs.map((x, j) =>
                        j === i ? { ...x, key: e.target.value } : x,
                      ),
                    })
                  }
                />
              </div>
              <div className="w-36">
                <label className={LABEL_CLASS}>Type</label>
                <select
                  className={INPUT_CLASS}
                  value={input.component}
                  onChange={(e) =>
                    commit({
                      ...schema,
                      inputs: schema.inputs.map((x, j) =>
                        j === i ? { ...x, component: e.target.value as InputKind } : x,
                      ),
                    })
                  }
                >
                  {INPUT_KINDS.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
              </div>
              <button
                type="button"
                className={REMOVE_BTN}
                aria-label={`Remove input ${input.label}`}
                onClick={() =>
                  commit({ ...schema, inputs: schema.inputs.filter((_, j) => j !== i) })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* Run controls ------------------------------------------------------ */}
      <section className={SECTION_CLASS}>
        <div className="mb-3">
          <h3 className="text-[13px] font-semibold text-db-navy-800">
            Run controls
          </h3>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {(Object.keys(RUN_CONTROL_LABELS) as Array<keyof SurfaceRuntimeControls>).map(
            (key) => (
              <label key={key} className="block">
                <span className={LABEL_CLASS}>{RUN_CONTROL_LABELS[key]}</span>
                <select
                  className={INPUT_CLASS}
                  value={schema.runControls[key] ?? 'show'}
                  onChange={(e) =>
                    commit({
                      ...schema,
                      runControls: {
                        ...schema.runControls,
                        [key]: e.target.value as SurfaceControlPolicy,
                      },
                    })
                  }
                >
                  {CONTROL_POLICIES.map((policy) => (
                    <option key={policy} value={policy}>
                      {policy}
                    </option>
                  ))}
                </select>
              </label>
            ),
          )}
        </div>
      </section>

      {/* Actions ----------------------------------------------------------- */}
      <section className={SECTION_CLASS}>
        <div className="mb-3">
          <h3 className="text-[13px] font-semibold text-db-navy-800">Actions</h3>
        </div>
        {schema.actions.length === 0 && (
          <p className="text-[12px] text-db-gray-text">No actions.</p>
        )}
        <div className="flex flex-col gap-2">
          {schema.actions.map((action, i) => (
            <div key={action.action} className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
              <label>
                <span className={LABEL_CLASS}>Label</span>
                <input
                  className={INPUT_CLASS}
                  value={action.label}
                  onChange={(e) =>
                    commit({
                      ...schema,
                      actions: schema.actions.map((x, j) =>
                        j === i ? { ...x, label: e.target.value } : x,
                      ),
                    })
                  }
                />
              </label>
              <label>
                <span className={LABEL_CLASS}>Output target</span>
                <input
                  className={INPUT_CLASS}
                  value={action.target}
                  onChange={(e) =>
                    commit({
                      ...schema,
                      target: action.action === schema.action ? e.target.value : schema.target,
                      actions: schema.actions.map((x, j) =>
                        j === i ? { ...x, target: e.target.value } : x,
                      ),
                    })
                  }
                />
              </label>
              <div className="self-end pb-1 font-db-mono text-[11px] text-db-gray-text">
                {action.action}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Output sections ---------------------------------------------------- */}
      <section className={SECTION_CLASS}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-[13px] font-semibold text-db-navy-800">
            Output sections
          </h3>
          <button
            type="button"
            className={GHOST_BTN}
            onClick={() => {
              const name = nextName('section_', slotNames);
              const slot: EditableSlot = {
                id: `new_${name}`,
                name,
                kind: 'findings',
                columns: [],
                componentIds: [],
                hasChart: false,
              };
              commit({ ...schema, slots: [...schema.slots, slot] });
            }}
          >
            + Add section
          </button>
        </div>
        {schema.slots.length === 0 && (
          <p className="text-[12px] text-db-gray-text">
            No output sections. Add one so the agent produces structured results.
          </p>
        )}
        <div className="flex flex-col gap-3">
          {schema.slots.map((slot, i) => (
            <SlotCard
              key={slot.id ?? slot.componentIds[0] ?? `new-${i}`}
              slot={slot}
              onChange={(patch) =>
                commit({
                  ...schema,
                  slots: schema.slots.map((x, j) => (j === i ? { ...x, ...patch } : x)),
                })
              }
              onRemove={() =>
                commit({ ...schema, slots: schema.slots.filter((_, j) => j !== i) })
              }
            />
          ))}
        </div>
      </section>
    </div>
  );
}

function SlotCard({
  slot,
  onChange,
  onRemove,
}: {
  slot: EditableSlot;
  onChange: (patch: Partial<EditableSlot>) => void;
  onRemove: () => void;
}): React.ReactElement {
  return (
    <div className="rounded-db-md border border-db-gray-lines bg-db-oat-light/40 p-3">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <label className={LABEL_CLASS}>Section name</label>
          <input
            className={INPUT_CLASS}
            value={slot.name}
            onChange={(e) => onChange({ name: e.target.value })}
          />
        </div>
        <div className="w-40">
          <label className={LABEL_CLASS}>Kind</label>
          <select
            className={INPUT_CLASS}
            value={slot.kind}
            onChange={(e) => onChange({ kind: e.target.value as SlotKind })}
          >
            {SLOT_KINDS.map((k) => (
              <option key={k.value} value={k.value}>
                {k.label}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className={REMOVE_BTN}
          aria-label={`Remove section ${slot.name}`}
          onClick={onRemove}
        >
          Remove
        </button>
      </div>

      {slot.hasChart && (
        <p className="mt-2 text-[11px] text-db-gray-text">
          This section also has a chart (edit it from the co-pilot).
        </p>
      )}

      {slot.kind === 'table' && (
        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between">
            <label className={LABEL_CLASS}>Columns</label>
            <button
              type="button"
              className={GHOST_BTN}
              onClick={() =>
                onChange({
                  columns: [
                    ...slot.columns,
                    { key: `column${slot.columns.length + 1}`, label: 'Column', type: 'string' },
                  ],
                })
              }
            >
              + Add column
            </button>
          </div>
          {slot.columns.length === 0 && (
            <p className="text-[11px] text-db-gray-text">No columns yet.</p>
          )}
          <div className="flex flex-col gap-2">
            {slot.columns.map((col, ci) => (
              <div key={ci} className="flex items-center gap-2">
                <input
                  className={INPUT_CLASS}
                  placeholder="key"
                  value={col.key}
                  onChange={(e) =>
                    onChange({
                      columns: slot.columns.map((c, j) =>
                        j === ci ? { ...c, key: e.target.value } : c,
                      ),
                    })
                  }
                />
                <input
                  className={INPUT_CLASS}
                  placeholder="label"
                  value={col.label}
                  onChange={(e) =>
                    onChange({
                      columns: slot.columns.map((c, j) =>
                        j === ci ? { ...c, label: e.target.value } : c,
                      ),
                    })
                  }
                />
                <select
                  className={`${INPUT_CLASS} w-32`}
                  value={col.type}
                  onChange={(e) =>
                    onChange({
                      columns: slot.columns.map((c, j) =>
                        j === ci ? { ...c, type: e.target.value as ColumnType } : c,
                      ),
                    })
                  }
                >
                  {COLUMN_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className={REMOVE_BTN}
                  aria-label={`Remove column ${col.key}`}
                  onClick={() =>
                    onChange({ columns: slot.columns.filter((_, j) => j !== ci) })
                  }
                >
                  X
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
