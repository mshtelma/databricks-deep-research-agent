/**
 * Editable schema model — a first-class, round-trippable view of an agent's
 * input fields and output slots, derived from (and written back to) the
 * `definition.surface` component tree.
 *
 * The surface has no separate schema object: inputs are INPUT_COMPONENTS bound
 * to `/form/<key>`, and output slots are OUTPUT components whose source/items
 * pointer is `<binding.output.target>/data/<slot>` (see the backend
 * `surface/output_schema.py` grammar). This module ports that mapping to the FE
 * so the Designer can present + edit the schema, then reconstructs the surface
 * (surgically — non-schema components like layout/report regions are preserved)
 * for the existing save + `validate_surface` write gate.
 */

import type {
  Surface,
  SurfaceComponent,
  ActionBinding,
  DynamicValue,
  SurfaceRuntimeControls,
} from '@/types/surface';

import { INPUT_COMPONENTS } from './surfaceComponents';

/** Output component → the prop that carries its slot pointer. */
const OUTPUT_POINTER_PROP: Record<string, string> = {
  Table: 'source',
  MetricGrid: 'source',
  KeyFindings: 'source',
  Chart: 'source',
  List: 'items',
};

/** Editor-facing slot kinds we can author (Chart/List are preserved on
 * round-trip but not created here). */
export type SlotKind = 'table' | 'metrics' | 'findings';
export type ColumnType = 'string' | 'number' | 'date';
export type InputKind = 'TextField' | 'TextArea' | 'Select' | 'Checkbox';

export interface EditableColumn {
  key: string;
  label: string;
  type: ColumnType;
}

export interface EditableInput {
  /** Component id (stable across edits). */
  id: string;
  component: InputKind;
  label: string;
  /** Workflow input key; the pointer is `/form/<key>`. */
  key: string;
}

export interface EditableSlot {
  /** Stable render key for the primary output component, when one exists. */
  id?: string;
  /** Slot name (the segment after `/data/`). */
  name: string;
  kind: SlotKind;
  /** Table columns (empty for metrics/findings). */
  columns: EditableColumn[];
  /** Component ids currently bound to this slot (for surgical round-trip). */
  componentIds: string[];
  /** True when the slot also has a Chart component (preserved, not edited here). */
  hasChart: boolean;
}

export type EditableRunControls = SurfaceRuntimeControls;

export interface EditableAction {
  action: string;
  label: string;
  target: string;
  buttonId?: string;
}

export interface EditableSchema {
  /** Binding action (e.g. "run"); empty when the surface has no binding. */
  action: string;
  /** Binding output target (e.g. "/results/run"). */
  target: string;
  runControls: EditableRunControls;
  actions: EditableAction[];
  inputs: EditableInput[];
  slots: EditableSlot[];
}

function pointerOf(value: unknown): string | null {
  if (value && typeof value === 'object' && 'path' in value) {
    const p = (value as { path?: unknown }).path;
    return typeof p === 'string' ? p : null;
  }
  return null;
}

/** Extract (action, slot) when `pointer` is `<target>/data/<slot>` (exactly one
 * segment after /data/); mirrors backend `split_slot_pointer`. */
function splitSlotPointer(
  pointer: string,
  targets: Record<string, string>,
): [string, string] | null {
  for (const [action, target] of Object.entries(targets)) {
    const prefix = `${target}/data/`;
    if (pointer.startsWith(prefix)) {
      const rest = pointer.slice(prefix.length);
      if (rest && !rest.includes('/')) return [action, rest];
    }
  }
  return null;
}

function slotKindFor(component: string): SlotKind {
  if (component === 'MetricGrid') return 'metrics';
  if (component === 'KeyFindings' || component === 'List') return 'findings';
  return 'table'; // Table / Chart
}

function outputPointerProp(component: string): string {
  return OUTPUT_POINTER_PROP[component] ?? 'source';
}

function componentCompatibleWithKind(component: string, kind: SlotKind): boolean {
  if (kind === 'table') return component === 'Table';
  if (kind === 'metrics') return component === 'MetricGrid';
  return component === 'KeyFindings' || component === 'List';
}

function columnsOf(props: Record<string, unknown>): EditableColumn[] {
  const raw = props['columns'];
  if (!Array.isArray(raw)) return [];
  const out: EditableColumn[] = [];
  for (const c of raw) {
    if (c && typeof c === 'object') {
      const key = (c as Record<string, unknown>)['key'];
      const label = (c as Record<string, unknown>)['label'];
      const type = (c as Record<string, unknown>)['type'];
      if (typeof key === 'string') {
        out.push({
          key,
          label: typeof label === 'string' ? label : key,
          type: type === 'number' || type === 'date' ? type : 'string',
        });
      }
    }
  }
  return out;
}

/** Derive the editable schema from a surface (the primary binding, or the one
 * matching `preferredAction`). Returns empty inputs/slots when absent. */
export function surfaceToSchema(
  surface: Surface | null | undefined,
  preferredAction?: string,
): EditableSchema {
  const empty: EditableSchema = {
    action: '',
    target: '',
    runControls: {},
    actions: [],
    inputs: [],
    slots: [],
  };
  if (!surface || !Array.isArray(surface.components)) return empty;

  const bindings: ActionBinding[] = surface.bindings ?? [];
  const binding =
    bindings.find((b) => b.action === preferredAction) ?? bindings[0] ?? null;
  const action = binding?.action ?? '';
  const target = binding?.output?.target ?? '';
  const targets: Record<string, string> = {};
  for (const b of bindings) targets[b.action] = b.output.target;

  const inputs: EditableInput[] = [];
  for (const comp of surface.components) {
    if (!INPUT_COMPONENTS.has(comp.component)) continue;
    const pointer = pointerOf(comp.props?.['value']);
    const key = pointer?.startsWith('/form/')
      ? pointer.slice('/form/'.length)
      : (pointer ?? comp.id);
    const label = comp.props?.['label'];
    inputs.push({
      id: comp.id,
      component: comp.component as InputKind,
      label: typeof label === 'string' ? label : key,
      key,
    });
  }

  const actions: EditableAction[] = bindings.map((b) => {
    const button = surface.components.find(
      (comp) => comp.component === 'Button' && comp.props?.['action'] === b.action,
    );
    const label = button?.props?.['label'];
    return {
      action: b.action,
      label: typeof label === 'string' ? label : b.action,
      target: b.output.target,
      buttonId: button?.id,
    };
  });

  // Group output components by slot (a slot may be shared by Table + Chart).
  const bySlot = new Map<string, EditableSlot>();
  for (const comp of surface.components) {
    const prop = OUTPUT_POINTER_PROP[comp.component];
    if (!prop) continue;
    const pointer = pointerOf(comp.props?.[prop]);
    if (pointer === null) continue;
    const split = splitSlotPointer(pointer, targets);
    if (split === null) continue;
    const [, slotName] = split;
    let slot = bySlot.get(slotName);
    if (!slot) {
      slot = {
        id: comp.component === 'Chart' ? undefined : comp.id,
        name: slotName,
        kind: slotKindFor(comp.component),
        columns: [],
        componentIds: [],
        hasChart: false,
      };
      bySlot.set(slotName, slot);
    }
    if (comp.component !== 'Chart') slot.id = comp.id;
    slot.componentIds.push(comp.id);
    if (comp.component === 'Chart') slot.hasChart = true;
    if (comp.component === 'Table') {
      slot.kind = 'table';
      slot.columns = columnsOf(comp.props ?? {});
    } else if (comp.component === 'MetricGrid') {
      slot.kind = 'metrics';
    } else if (comp.component === 'KeyFindings' || comp.component === 'List') {
      slot.kind = 'findings';
    }
  }

  return {
    action,
    target,
    runControls: surface.runtime_controls ?? {},
    actions,
    inputs,
    slots: [...bySlot.values()],
  };
}

function uniqueId(base: string, taken: Set<string>): string {
  if (!taken.has(base)) {
    taken.add(base);
    return base;
  }
  let i = 2;
  while (taken.has(`${base}_${i}`)) i += 1;
  const id = `${base}_${i}`;
  taken.add(id);
  return id;
}

const OUTPUT_COMPONENT_FOR: Record<SlotKind, string> = {
  table: 'Table',
  metrics: 'MetricGrid',
  findings: 'KeyFindings',
};

/** Build/update props for an output component while preserving non-pointer props. */
function outputPropsForComponent(
  component: string,
  pointer: string,
  columns: EditableColumn[],
  existing: Record<string, unknown> = {},
): Record<string, unknown> {
  const next: Record<string, unknown> = { ...existing };
  for (const prop of Object.values(OUTPUT_POINTER_PROP)) delete next[prop];
  if (component !== 'Table') delete next['columns'];

  next[outputPointerProp(component)] = { path: pointer };
  if (component === 'Table') {
    next['columns'] = columns.map((c) => ({
      key: c.key,
      label: c.label,
      type: c.type,
    }));
  }
  return next;
}

/** Build the props for a freshly-created output component of `kind`. */
function outputPropsFor(
  kind: SlotKind,
  pointer: string,
  columns: EditableColumn[],
): Record<string, unknown> {
  const component = OUTPUT_COMPONENT_FOR[kind];
  if (component === 'Table') {
    return {
      source: { path: pointer },
      columns: columns.map((c) => ({ key: c.key, label: c.label, type: c.type })),
    };
  }
  return outputPropsForComponent(component, pointer, columns);
}

/**
 * Reconstruct a surface from an edited schema, preserving every component that
 * is NOT a schema input/output-slot (root, layout, report regions, buttons,
 * static text). Inputs and output slots are reconciled by key/slot:
 * updated in place, created under the container that holds the existing ones
 * (fallback: root), and removed (with their children references + data_model +
 * binding inputs) when dropped. Chart/List components on a surviving slot are
 * preserved. The result is validated by the server write gate on save.
 */
export function applySchemaToSurface(
  surface: Surface,
  schema: EditableSchema,
): Surface {
  const next: Surface = JSON.parse(JSON.stringify(surface));
  next.runtime_controls = schema.runControls;
  const components = next.components;
  const byId = new Map<string, SurfaceComponent>();
  for (const c of components) byId.set(c.id, c);
  const takenIds = new Set(byId.keys());

  const parentOf = new Map<string, string>();
  for (const c of components) for (const ch of c.children) parentOf.set(ch, c.id);

  const target = schema.target || '/results/run';

  // --- Inputs -------------------------------------------------------------
  const existingInputs = components.filter((c) => INPUT_COMPONENTS.has(c.component));
  const formParent = existingInputs.length
    ? (parentOf.get(existingInputs[0]!.id) ?? 'root')
    : 'root';
  const keepInputIds = new Set<string>();
  const dataModel = (next.data_model ?? {}) as Record<string, unknown>;
  const form = (dataModel['form'] && typeof dataModel['form'] === 'object'
    ? (dataModel['form'] as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  const bindingInputs: Record<string, DynamicValue> = {};

  for (const input of schema.inputs) {
    const pointer = `/form/${input.key}`;
    let comp = byId.get(input.id);
    if (comp && INPUT_COMPONENTS.has(comp.component)) {
      comp.component = input.component;
      comp.props = { ...comp.props, label: input.label, value: { path: pointer } };
    } else {
      const id = uniqueId(`field_${input.key}`, takenIds);
      comp = {
        id,
        component: input.component,
        props: { label: input.label, value: { path: pointer } },
        children: [],
      };
      components.push(comp);
      byId.set(id, comp);
      const parent = byId.get(formParent);
      if (parent && !parent.children.includes(id)) parent.children.push(id);
    }
    keepInputIds.add(comp.id);
    if (form[input.key] === undefined) form[input.key] = '';
    bindingInputs[input.key] = { path: pointer };
  }

  // --- Output slots -------------------------------------------------------
  const existingOutputs = components.filter(
    (c) => OUTPUT_POINTER_PROP[c.component] !== undefined,
  );
  const resultsParent = existingOutputs.length
    ? (parentOf.get(existingOutputs[0]!.id) ?? 'root')
    : 'root';
  const keepSlotComponentIds = new Set<string>();
  const isSlotComp = (c: SurfaceComponent): boolean =>
    OUTPUT_POINTER_PROP[c.component] !== undefined;
  const pointerOfComp = (c: SurfaceComponent): string | null =>
    pointerOf(c.props?.[OUTPUT_POINTER_PROP[c.component]!]);

  for (const slot of schema.slots) {
    const pointer = `${target}/data/${slot.name}`;
    // Find the primary (non-Chart) component: by tracked id first (survives a
    // rename), else by current pointer (survives re-applying a new slot, so we
    // never create a duplicate on repeated commits).
    let primary: SurfaceComponent | undefined;
    for (const id of slot.componentIds) {
      const c = byId.get(id);
      if (c && isSlotComp(c) && c.component !== 'Chart') {
        primary = c;
        break;
      }
    }
    if (!primary) {
      primary = components.find(
        (c) => isSlotComp(c) && c.component !== 'Chart' && pointerOfComp(c) === pointer,
      );
    }
    if (primary) {
      const nextComponent = componentCompatibleWithKind(primary.component, slot.kind)
        ? primary.component
        : OUTPUT_COMPONENT_FOR[slot.kind];
      primary.component = nextComponent;
      primary.props = outputPropsForComponent(
        nextComponent,
        pointer,
        slot.columns,
        primary.props ?? {},
      );
      keepSlotComponentIds.add(primary.id);
    } else {
      const id = uniqueId(`slot_${slot.name}`, takenIds);
      const comp: SurfaceComponent = {
        id,
        component: OUTPUT_COMPONENT_FOR[slot.kind],
        props: outputPropsFor(slot.kind, pointer, slot.columns),
        children: [],
      };
      components.push(comp);
      byId.set(id, comp);
      const parent = byId.get(resultsParent);
      if (parent && !parent.children.includes(id)) parent.children.push(id);
      keepSlotComponentIds.add(id);
    }
    // Preserve a Chart bound to this slot (by tracked id or pointer) — the
    // schema editor doesn't author charts, but must not drop existing ones.
    for (const c of components) {
      if (
        c.component === 'Chart' &&
        (slot.componentIds.includes(c.id) || pointerOfComp(c) === pointer)
      ) {
        c.props = outputPropsForComponent(
          'Chart',
          pointer,
          slot.columns,
          c.props ?? {},
        );
        keepSlotComponentIds.add(c.id);
      }
    }
  }

  // --- Prune removed inputs / output slots --------------------------------
  const removeIds = new Set<string>();
  for (const c of components) {
    if (INPUT_COMPONENTS.has(c.component) && !keepInputIds.has(c.id)) {
      removeIds.add(c.id);
    } else if (
      OUTPUT_POINTER_PROP[c.component] !== undefined &&
      !keepSlotComponentIds.has(c.id) &&
      // only prune output components that actually bind a slot under our target
      splitSlotPointer(
        pointerOf(c.props?.[OUTPUT_POINTER_PROP[c.component]!]) ?? '',
        { [schema.action || 'run']: target },
      ) !== null
    ) {
      removeIds.add(c.id);
    }
  }
  if (removeIds.size) {
    next.components = components.filter((c) => !removeIds.has(c.id));
    for (const c of next.components) {
      c.children = c.children.filter((ch) => !removeIds.has(ch));
    }
  }

  // --- data_model + binding inputs ---------------------------------------
  const keepFormKeys = new Set(schema.inputs.map((i) => i.key));
  for (const k of Object.keys(form)) {
    if (!keepFormKeys.has(k)) delete form[k];
  }
  dataModel['form'] = form;
  next.data_model = dataModel;

  if (schema.action) {
    const binding = (next.bindings ?? []).find((b) => b.action === schema.action);
    if (binding) binding.inputs = bindingInputs;
  }

  for (const action of schema.actions) {
    const binding = (next.bindings ?? []).find((b) => b.action === action.action);
    if (binding && action.target) {
      binding.output.target = action.target;
    }
    const button =
      (action.buttonId ? byId.get(action.buttonId) : undefined) ??
      next.components.find(
        (c) => c.component === 'Button' && c.props?.['action'] === action.action,
      );
    if (button) {
      button.props = { ...button.props, label: action.label || action.action };
    }
  }

  return next;
}
