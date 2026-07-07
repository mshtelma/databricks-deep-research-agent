/**
 * ConfigPanel — Inspector aside (340px) on the right side of the designer.
 *
 * Two faces:
 *   1. Nothing selected → Workspace tools view (registry of declared tools
 *      available to any agent in this workflow). Add / remove tool decls.
 *   2. Block selected → header pill + tabs (Configure / Tools / Approvals).
 *      Configure renders the schema-driven SchemaField form (preserves all
 *      existing AJV validation logic). Tools binds workflow tools to the
 *      selected agent. Approvals lists gated tools + broker config.
 *
 * The Configure tab keeps the original AJV schema validation logic so existing
 * field rendering, error mapping, and dirty tracking continue to work.
 */

import * as React from 'react';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import {
  Wrench,
  Plus,
  X,
  Lock,
  Check,
  Info,
  Box,
  Boxes,
  RefreshCw,
  FlaskConical,
  Pencil,
  Save,
  Sparkles,
  ChevronRight,
  Link,
} from 'lucide-react';
import type { RegistryResponse, ToolKindSpec, DesignerResource } from '@/types/agentDesigner';
import type { ToolDecl } from '@/types/ast';
import {
  refreshCatalog,
  probeTools,
  listDesignerResources,
  getDesignerCapabilities,
  type ProbeSample,
  type DesignerCapabilities,
} from '@/api/agentDesigner';
import { resolveBlock } from '@/lib/blockPath';
import { requiredConfigErrors, schemaProperties } from '@/lib/jsonSchema';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { TypePill, LayerChip } from './atoms';
import { SchemaField } from './SchemaField';
import { AddToolDialog } from './AddToolDialog';
import { WorkflowSettingsPanel } from './WorkflowSettingsPanel';
import { ChatPanel } from './ChatPanel';

// ---------------------------------------------------------------------------
// AJV factory
// ---------------------------------------------------------------------------

/** A SchemaField ``x-enumOptions`` entry (dropdown choice). */
interface EnumOption {
  value: string;
  label: string;
}

/** Inject discovery-populated ``x-enumOptions`` into an array field's item schema. */
function withItemOptions(
  fieldSchema: Record<string, unknown>,
  options: EnumOption[],
): Record<string, unknown> {
  if (options.length === 0) return fieldSchema;
  const baseItems =
    fieldSchema['items'] && typeof fieldSchema['items'] === 'object'
      ? (fieldSchema['items'] as Record<string, unknown>)
      : { type: 'string' };
  return { ...fieldSchema, items: { ...baseItems, 'x-enumOptions': options } };
}

function makeValidator(schema: Record<string, unknown>): ReturnType<Ajv['compile']> | null {
  try {
    const instance = new Ajv({ allErrors: true, strict: false });
    addFormats(instance);
    return instance.compile(schema);
  } catch {
    return null;
  }
}

function buildErrorMap(
  errors: import('ajv').ErrorObject[] | null | undefined,
): Record<string, string[]> {
  if (!errors) return {};
  const map: Record<string, string[]> = {};
  for (const err of errors) {
    let key: string;
    if (err.keyword === 'required' && err.params && 'missingProperty' in err.params) {
      key = String((err.params as Record<string, unknown>)['missingProperty'] ?? '');
    } else {
      const raw = err.instancePath || '';
      key = raw.startsWith('/') ? (raw.slice(1).split('/')[0] ?? '') : raw;
    }
    if (!map[key]) map[key] = [];
    map[key]!.push(err.message ?? 'Invalid value');
  }
  return map;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function findToolKind(registry: RegistryResponse, kind: string | undefined): ToolKindSpec | null {
  if (!kind) return null;
  return registry.tool_kinds.find((k) => k.kind === kind) ?? null;
}

/** Returns true if a tool declaration's kind opts into HITL approval gating. */
function toolRequiresApproval(decl: ToolDecl): boolean {
  return decl.config?.['requires_approval'] === true;
}

function toolSummary(decl: ToolDecl): string {
  const config = decl.config ?? {};
  const primary =
    config['index_name'] ??
    config['space_id'] ??
    config['endpoint_name'] ??
    config['function_name'] ??
    config['tool_name'] ??
    config['import'] ??
    config['max_results'] ??
    config['num_results'];
  return primary === undefined || primary === null || primary === '' ? 'Not configured' : String(primary);
}

/**
 * Layer letter → human group label, matching the LayerChip colour taxonomy
 * (atoms.tsx LAYER_CMAP): A=Web, B=Knowledge, C=Data, D=Filesystem, E=MCP/Custom.
 * Used to bucket an agent's declared tools into the design's "Built-in & custom"
 * grouped checklist. Generic tool-category vocabulary — no domain coupling.
 */
const LAYER_GROUP_LABEL: Record<string, string> = {
  A: 'Web',
  B: 'Knowledge',
  C: 'Data',
  D: 'Files',
  E: 'Custom',
};
const LAYER_ORDER = ['A', 'B', 'C', 'D', 'E'];

/** Group declared tools by their kind's layer, ordered A→E (unknown layers last). */
function groupDeclaredToolsByLayer(
  tools: ToolDecl[],
  registry: RegistryResponse,
): Array<[string, ToolDecl[]]> {
  const byLayer = new Map<string, ToolDecl[]>();
  for (const decl of tools) {
    const layer = findToolKind(registry, decl.kind)?.layer ?? 'A';
    const bucket = byLayer.get(layer);
    if (bucket) bucket.push(decl);
    else byLayer.set(layer, [decl]);
  }
  const ordered: Array<[string, ToolDecl[]]> = [];
  for (const layer of LAYER_ORDER) {
    const arr = byLayer.get(layer);
    if (arr) {
      ordered.push([LAYER_GROUP_LABEL[layer] ?? layer, arr]);
      byLayer.delete(layer);
    }
  }
  // Any unknown layers (forward-compat) appended in insertion order.
  for (const [layer, arr] of byLayer) {
    ordered.push([LAYER_GROUP_LABEL[layer] ?? layer, arr]);
  }
  return ordered;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ConfigPanelProps {
  registry: RegistryResponse;
  /**
   * Session id for the embedded co-pilot tab (Direction 2 redesign — the
   * separate Designer Chat column folds into the inspector). Typically the
   * agent id; undefined for a brand-new draft.
   */
  chatSessionId?: string;
}

type AddToolRequest =
  | { mode: 'workspace' }
  | { mode: 'bind-agent'; blockPath: string }
  | { mode: 'select-tool-step'; blockPath: string };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ConfigPanel({ registry, chatSessionId }: ConfigPanelProps): React.ReactElement {
  const selectedPath = useAgentEditorStore((s) => s.selectedPath);
  const ast = useAgentEditorStore((s) => s.ast);

  const [addToolRequest, setAddToolRequest] = React.useState<AddToolRequest | null>(null);
  const [selectedToolName, setSelectedToolName] = React.useState<string | null>(null);
  // No-selection view tab: the Workspace tool registry, or the workflow co-pilot
  // (which used to be a separate column; Direction 2 folds it into the inspector).
  const [noSelTab, setNoSelTab] = React.useState<'tools' | 'chat'>('tools');

  const block = React.useMemo(() => {
    if (!ast || !selectedPath) return null;
    return resolveBlock(ast, selectedPath);
  }, [ast, selectedPath]);

  const declaredTools: ToolDecl[] = React.useMemo(() => ast?.tools ?? [], [ast?.tools]);
  const selectedTool = declaredTools.find((tool) => tool.name === selectedToolName) ?? null;
  const showAddTool = addToolRequest !== null;

  const handleToolDeclared = React.useCallback(
    (tool: ToolDecl) => {
      const request = addToolRequest;
      if (!request) return;
      const store = useAgentEditorStore.getState();
      if (request.mode === 'bind-agent') {
        store.bindToolToBlock(request.blockPath, tool.name);
      } else if (request.mode === 'select-tool-step') {
        const latestAst = store.ast;
        const latestBlock = latestAst ? resolveBlock(latestAst, request.blockPath) : null;
        const currentConfig =
          latestBlock?.config && typeof latestBlock.config === 'object' ? latestBlock.config : {};
        store.updateBlock(request.blockPath, {
          config: { ...currentConfig, ref: { type: 'builtin', name: tool.name } },
        });
      }
    },
    [addToolRequest],
  );
  const addToolDialog = showAddTool ? (
    <AddToolDialog
      open={showAddTool}
      onOpenChange={(open) => setAddToolRequest(open ? addToolRequest : null)}
      onDeclared={handleToolDeclared}
      registry={registry}
    />
  ) : null;

  React.useEffect(() => {
    if (selectedToolName && !declaredTools.some((tool) => tool.name === selectedToolName)) {
      setSelectedToolName(null);
    }
  }, [declaredTools, selectedToolName]);

  // -------------------------------------------------------------------------
  // No selection → Workspace tools view
  // -------------------------------------------------------------------------

  if (!selectedPath || !block) {
    return (
      <>
        <aside className="db-root flex w-[430px] shrink-0 flex-col border-l border-db-gray-lines bg-white font-db-sans">
          <div className="border-b border-db-gray-lines px-4 pt-3.5">
            <div className="mb-3 flex items-center gap-2">
              <Wrench size={15} className="text-db-navy-800" />
              <span className="text-[13px] font-medium text-db-navy-800">Workspace</span>
              <span className="ml-auto font-db-mono text-[11px] text-db-gray-text">
                {declaredTools.length}
              </span>
              {noSelTab === 'tools' && (
                <button
                  type="button"
                  onClick={() => setAddToolRequest({ mode: 'workspace' })}
                  className="inline-flex items-center gap-1 rounded-db-md bg-db-lava-600 px-2.5 py-1 text-[12px] font-medium text-white transition-colors hover:bg-db-lava-700"
                >
                  <Plus size={11} /> Add
                </button>
              )}
            </div>
            <div className="flex gap-0">
              <InspectorTabButton
                active={noSelTab === 'tools'}
                onClick={() => setNoSelTab('tools')}
                label="Workspace tools"
              />
              <InspectorTabButton
                active={noSelTab === 'chat'}
                onClick={() => setNoSelTab('chat')}
                label="Co-pilot"
                icon={<Sparkles size={13} />}
              />
            </div>
          </div>
          {noSelTab === 'chat' ? (
            <div className="flex min-h-0 flex-1 flex-col">
              <ChatPanel embedded sessionId={chatSessionId} />
            </div>
          ) : (
            <>
              <div className="border-b border-db-gray-lines bg-db-oat-light px-3.5 py-2.5 text-[11px] leading-[1.5] text-db-gray-text">
                Tools available to any agent in this workflow. Select an agent block to bind tools
                to it.
              </div>
              <div className="min-h-0 flex-1 overflow-auto p-3">
                <ToolsRegistryList
                  tools={declaredTools}
                  registry={registry}
                  mode="workspace"
                  selectedName={selectedToolName}
                  onSelect={setSelectedToolName}
                />
                {selectedTool && (
                  <ToolDeclarationEditor
                    key={selectedTool.name}
                    tool={selectedTool}
                    registry={registry}
                    onRename={setSelectedToolName}
                    onClose={() => setSelectedToolName(null)}
                  />
                )}
                <WorkflowSettingsPanel />
              </div>
            </>
          )}
        </aside>
        {addToolDialog}
      </>
    );
  }

  // -------------------------------------------------------------------------
  // Block selected — render with tabs
  // -------------------------------------------------------------------------

  return (
    <>
      <SelectedInspector
        block={block}
        selectedPath={selectedPath}
        registry={registry}
        declaredTools={declaredTools}
        onShowAddTool={(request) => setAddToolRequest(request)}
        chatSessionId={chatSessionId}
      />
      {addToolDialog}
    </>
  );
}

// ---------------------------------------------------------------------------
// Selected-block inspector (with tabs)
// ---------------------------------------------------------------------------

interface SelectedInspectorProps {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  onShowAddTool: (request: AddToolRequest) => void;
  chatSessionId?: string;
}

// Direction 1 (Tabbed Inspector): the separate Designer Chat column is gone —
// the co-pilot is the 4th tab. The docked panel carries four tabs:
// Configure · Tools · Approvals · Co-pilot. Tools is a single grouped scroll
// (declared tools bucketed by layer + MCP servers as expandable cards), and
// Approvals is its own tab again (no longer folded into Tools).
type InspectorTab = 'config' | 'tools' | 'hitl' | 'chat';

function SelectedInspector({
  block,
  selectedPath,
  registry,
  declaredTools,
  onShowAddTool,
  chatSessionId,
}: SelectedInspectorProps): React.ReactElement {
  const [tab, setTab] = React.useState<InspectorTab>('config');

  // Reset to "config" when the selected block changes — UNLESS the user is in
  // the co-pilot tab, which is workflow-level and stays put across selections.
  const prevPathRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (selectedPath !== prevPathRef.current) {
      prevPathRef.current = selectedPath;
      setTab((t) => (t === 'chat' ? 'chat' : 'config'));
    }
  }, [selectedPath]);

  const isAgent = block.type === 'agent';
  const boundToolNames = Array.isArray(block.config.tools)
    ? (block.config.tools as string[])
    : [];

  return (
    <aside className="db-root flex w-[384px] shrink-0 flex-col border-l border-db-gray-lines bg-white font-db-sans">
      {/* Header */}
      <div className="border-b border-db-gray-lines px-4 pt-3.5">
        <div className="mb-3 flex items-center gap-2">
          <TypePill type={block.type} />
          <span className="truncate text-[14px] font-medium text-db-navy-800">{block.label}</span>
        </div>
        <div className="flex gap-0">
          <InspectorTabButton
            active={tab === 'config'}
            onClick={() => setTab('config')}
            label="Configure"
          />
          {isAgent && (
            <InspectorTabButton
              active={tab === 'tools'}
              onClick={() => setTab('tools')}
              label="Tools"
              count={boundToolNames.length}
            />
          )}
          {isAgent && (
            <InspectorTabButton
              active={tab === 'hitl'}
              onClick={() => setTab('hitl')}
              label="Approvals"
            />
          )}
          <InspectorTabButton
            active={tab === 'chat'}
            onClick={() => setTab('chat')}
            label="Co-pilot"
            icon={<Sparkles size={13} />}
          />
        </div>
      </div>

      {tab === 'chat' ? (
        // The embedded co-pilot manages its own scroll + composer, so give it
        // the full pane with no padding/overflow wrapper.
        <div className="flex min-h-0 flex-1 flex-col">
          <ChatPanel embedded sessionId={chatSessionId} />
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto p-4">
          {tab === 'config' && (
            <ConfigureForm
              block={block}
              selectedPath={selectedPath}
              registry={registry}
              declaredTools={declaredTools}
              onShowAddTool={onShowAddTool}
            />
          )}
          {tab === 'tools' && isAgent && (
            <ToolsPane
              block={block}
              selectedPath={selectedPath}
              registry={registry}
              declaredTools={declaredTools}
              boundToolNames={boundToolNames}
              onShowAddTool={onShowAddTool}
            />
          )}
          {tab === 'hitl' && isAgent && (
            <ApprovalsForm
              block={block}
              selectedPath={selectedPath}
              registry={registry}
              declaredTools={declaredTools}
              boundToolNames={boundToolNames}
            />
          )}
        </div>
      )}
    </aside>
  );
}

function InspectorTabButton({
  active,
  onClick,
  label,
  count,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count?: number;
  icon?: React.ReactNode;
}): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`-mb-px inline-flex items-center gap-1.5 border-b-2 px-3.5 py-2.5 font-db-sans text-[13px] font-medium transition-colors ${
        active
          ? 'border-db-lava-600 text-db-navy-800'
          : 'border-transparent text-db-gray-text hover:text-db-navy-800'
      }`}
    >
      {icon && (
        <span className={active ? 'text-db-lava-600' : 'text-db-gray-text'}>{icon}</span>
      )}
      {label}
      {count !== undefined && (
        <span className="text-[11px] text-db-gray-text">{count}</span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Configure tab — schema-driven form (preserves existing AJV logic)
// ---------------------------------------------------------------------------

interface ConfigureFormProps {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  onShowAddTool: (request: AddToolRequest) => void;
}

function ConfigureForm({
  block,
  selectedPath,
  registry,
  declaredTools,
  onShowAddTool,
}: ConfigureFormProps): React.ReactElement {
  const nodeSpec = registry.node_types.find((s) => s.type === block.type) ?? null;
  const configSchema = nodeSpec?.config_schema ?? null;

  const [formValue, setFormValue] = React.useState<Record<string, unknown>>(
    () => (block.config ? { ...(block.config as Record<string, unknown>) } : {}),
  );

  const prevPathRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (selectedPath !== prevPathRef.current) {
      prevPathRef.current = selectedPath;
      setFormValue(block.config ? { ...(block.config as Record<string, unknown>) } : {});
      setErrors(null);
    }
  }, [selectedPath, block]);

  const validate = React.useMemo(() => {
    if (!configSchema) return null;
    return makeValidator(configSchema);
  }, [configSchema]);

  const [ajvErrors, setErrors] = React.useState<Record<string, string[]> | null>(null);

  const handleFieldChange = React.useCallback(
    (fieldName: string, fieldValue: unknown) => {
      const newConfig = { ...formValue, [fieldName]: fieldValue };
      setFormValue(newConfig);
      if (validate) {
        const valid = validate(newConfig);
        setErrors(valid ? null : buildErrorMap(validate.errors));
      }
      if (selectedPath) {
        useAgentEditorStore.getState().updateBlock(selectedPath, { config: newConfig });
      }
    },
    [formValue, validate, selectedPath],
  );

  // Discovery-populated options for the skills / mcp_servers array fields. Fetched
  // once per agent inspector; on any error we keep the fields as free-text arrays
  // (graceful — authoring still works, just without a dropdown). No QueryClient
  // dependency (plain effect) so bare-rendered inspectors stay test-safe.
  const isAgentLike = block.type === 'agent' || block.type === 'plan_and_execute';
  const [skillOptions, setSkillOptions] = React.useState<EnumOption[]>([]);
  const [mcpOptions, setMcpOptions] = React.useState<EnumOption[]>([]);
  // Per-group collapse overrides (group name → user-set open state). Advanced
  // groups start collapsed; a group is force-opened when it contains an error.
  const [openGroups, setOpenGroups] = React.useState<Record<string, boolean>>({});
  // Global feature-gate states, so a gated knob set per-agent but disabled
  // globally is flagged (never-silent) instead of silently doing nothing.
  const [capabilities, setCapabilities] = React.useState<DesignerCapabilities | null>(null);
  React.useEffect(() => {
    if (!isAgentLike) return undefined;
    let cancelled = false;
    listDesignerResources(['skill', 'mcp_server'])
      .then((res) => {
        if (cancelled) return;
        const toOpts = (kind: string): EnumOption[] =>
          res.resources
            .filter((r) => r.kind === kind)
            .map((r) => ({ value: r.name, label: r.name }));
        setSkillOptions(toOpts('skill'));
        setMcpOptions(toOpts('mcp_server'));
      })
      .catch(() => {
        /* graceful: leave skills / mcp_servers as free-text arrays */
      });
    return () => {
      cancelled = true;
    };
  }, [isAgentLike, selectedPath]);

  // Fetch global gate states once (cached) so gated knobs can be annotated.
  React.useEffect(() => {
    if (!isAgentLike) return undefined;
    let cancelled = false;
    getDesignerCapabilities()
      .then((c) => {
        if (!cancelled) setCapabilities(c);
      })
      .catch(() => {
        /* graceful: no banner if capabilities are unavailable */
      });
    return () => {
      cancelled = true;
    };
  }, [isAgentLike]);

  if (!configSchema) {
    return (
      <p className="text-[12px] text-db-gray-text">No configurable properties for this block.</p>
    );
  }

  if (block.type === 'tool') {
    return (
      <ToolStepForm
        block={block}
        selectedPath={selectedPath}
        registry={registry}
        declaredTools={declaredTools}
        onShowAddTool={onShowAddTool}
      />
    );
  }

  const schemaProps =
    configSchema['properties'] &&
    typeof configSchema['properties'] === 'object' &&
    !Array.isArray(configSchema['properties'])
      ? (configSchema['properties'] as Record<string, Record<string, unknown>>)
      : {};
  const visibleSchemaProps =
    block.type === 'plan_and_execute'
      ? Object.fromEntries(
          Object.entries(schemaProps).filter(([fieldName]) => fieldName !== 'body'),
        )
      : schemaProps;

  const requiredKeys = Array.isArray(configSchema['required'])
    ? (configSchema['required'] as string[])
    : [];

  const errorMap = ajvErrors ?? {};

  // Turn the skills / mcp_servers free-text arrays into discovery-populated
  // dropdowns when options were fetched (otherwise the fields stay free-text).
  const enhancedSchemaProps: Record<string, Record<string, unknown>> = {
    ...visibleSchemaProps,
  };
  if (enhancedSchemaProps['skills']) {
    enhancedSchemaProps['skills'] = withItemOptions(
      enhancedSchemaProps['skills'],
      skillOptions,
    );
  }
  if (enhancedSchemaProps['mcp_servers']) {
    enhancedSchemaProps['mcp_servers'] = withItemOptions(
      enhancedSchemaProps['mcp_servers'],
      mcpOptions,
    );
  }

  // Never-silent: warn when a gated knob is enabled per-agent but its global
  // switch is off, so it doesn't quietly no-op at runtime.
  const skillScriptsGatedOff =
    capabilities !== null &&
    formValue['allow_skill_scripts'] === true &&
    !capabilities.skill_scripts_global;
  const gatedWarning = skillScriptsGatedOff ? (
    <div
      role="status"
      className="mb-3 flex items-start gap-2 rounded-db-md border border-db-gray-lines bg-db-oat-light px-3 py-2 text-[12px] leading-[1.5] text-db-navy-700"
    >
      <Info size={14} className="mt-0.5 shrink-0 text-db-gray-text" aria-hidden="true" />
      <span>
        <strong>allow_skill_scripts</strong> is on for this agent, but skill-script execution
        is disabled globally (<code className="font-db-mono">skills.allow_script_execution</code>)
        — it has no effect until an admin enables it.
      </span>
    </div>
  ) : null;

  const renderField = (
    fieldName: string,
    fieldSchema: Record<string, unknown>,
  ): React.ReactElement => (
    <SchemaField
      key={fieldName}
      name={fieldName}
      schema={fieldSchema}
      value={formValue[fieldName]}
      onChange={(v) => handleFieldChange(fieldName, v)}
      required={requiredKeys.includes(fieldName)}
      errors={errorMap[fieldName] ?? []}
    />
  );

  const propEntries = Object.entries(enhancedSchemaProps);
  const hasGroups = propEntries.some(
    ([, s]) => typeof s['x-group'] === 'string',
  );

  // Non-agent nodes (loop, conditional, tool, …) have no grouping metadata —
  // render them flat, exactly as before. Only the agent inspector's 40+ fields
  // get collapsible groups.
  if (!hasGroups) {
    return (
      <>
        {gatedWarning}
        {propEntries.map(([name, s]) => renderField(name, s))}
      </>
    );
  }

  // Partition into groups, preserving the backend's (group, order) ordering
  // (the registry emits properties already sorted, and Map keeps insertion order).
  const grouped = new Map<string, Array<[string, Record<string, unknown>]>>();
  for (const [fieldName, fieldSchema] of propEntries) {
    const group =
      typeof fieldSchema['x-group'] === 'string'
        ? (fieldSchema['x-group'] as string)
        : 'Settings';
    const bucket = grouped.get(group);
    if (bucket) bucket.push([fieldName, fieldSchema]);
    else grouped.set(group, [[fieldName, fieldSchema]]);
  }

  return (
    <>
      {gatedWarning}
      {Array.from(grouped.entries()).map(([groupName, entries]) => {
        const advanced = entries.every(([, s]) => s['x-advanced'] === true);
        const errorCount = entries.reduce(
          (n, [fieldName]) => n + (errorMap[fieldName]?.length ?? 0),
          0,
        );
        const userOpen = openGroups[groupName];
        // Default: expanded unless advanced. User toggle wins. An error always
        // force-opens the group so a hidden validation error can't be missed.
        const open = (userOpen !== undefined ? userOpen : !advanced) || errorCount > 0;
        return (
          <details
            key={groupName}
            open={open}
            className="mb-2 rounded-db-md border border-db-gray-lines bg-white"
          >
            <summary
              onClick={(e) => {
                e.preventDefault();
                setOpenGroups((prev) => ({ ...prev, [groupName]: !open }));
              }}
              className="flex cursor-pointer select-none items-center justify-between gap-2 rounded-db-md px-3 py-2 font-db-sans text-[11px] font-semibold uppercase tracking-[0.06em] text-db-navy-700 hover:bg-db-oat-light"
            >
              <span className="flex items-center gap-1.5">
                <ChevronRight
                  size={13}
                  className={`transition-transform ${open ? 'rotate-90' : ''}`}
                  aria-hidden="true"
                />
                {groupName}
              </span>
              {errorCount > 0 && (
                <span
                  className="rounded-full bg-db-lava-600 px-1.5 py-0.5 text-[10px] font-bold text-white"
                  aria-label={`${errorCount} validation ${errorCount === 1 ? 'error' : 'errors'}`}
                >
                  {errorCount}
                </span>
              )}
            </summary>
            <div className="border-t border-db-gray-lines px-3 pt-3">
              {entries.map(([name, s]) => renderField(name, s))}
            </div>
          </details>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// Tool Step form — declaration-first calls with advanced direct-ref compatibility
// ---------------------------------------------------------------------------

type DirectToolRefType = 'uc_function' | 'uc_tool' | 'enterprise';

const TOOL_STEP_INPUT_CLASS =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-sans text-[13px] leading-[1.4] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const TOOL_STEP_MONO_INPUT_CLASS =
  'w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] leading-[1.4] text-db-navy-800 outline-none transition-colors placeholder:text-db-navy-300 focus:border-db-navy-400 focus:shadow-db-focus';

const DIRECT_TOOL_REF_LABELS: Record<DirectToolRefType, {
  nameLabel: string;
  placeholder: string;
}> = {
  uc_function: {
    nameLabel: 'Function name',
    placeholder: 'catalog.schema.function',
  },
  uc_tool: {
    nameLabel: 'Tool name',
    placeholder: 'catalog.schema.tool',
  },
  enterprise: {
    nameLabel: 'Tool name',
    placeholder: 'tool_name',
  },
};

function normalizeToolRef(value: unknown): { type: string; name: string } {
  if (typeof value === 'string') {
    return { type: 'builtin', name: value };
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const raw = value as Record<string, unknown>;
    return {
      type: typeof raw['type'] === 'string' ? raw['type'] : 'builtin',
      name: typeof raw['name'] === 'string' ? raw['name'] : '',
    };
  }
  return { type: 'builtin', name: '' };
}

function isDirectToolRef(ref: { type: string }): boolean {
  if (ref.type === 'uc_function' || ref.type === 'uc_tool' || ref.type === 'enterprise') {
    return true;
  }
  return false;
}

function normalizeInputMapping(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }
  const out: Record<string, string> = {};
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    out[key] = typeof raw === 'string' ? raw : String(raw ?? '');
  }
  return out;
}

function uniqueMappingKey(mapping: Record<string, string>): string {
  const base = 'parameter';
  if (!(base in mapping)) return base;
  let index = 2;
  while (`${base}_${index}` in mapping) {
    index += 1;
  }
  return `${base}_${index}`;
}

function ToolStepForm({
  block,
  selectedPath,
  registry,
  declaredTools,
  onShowAddTool,
}: {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  onShowAddTool: (request: AddToolRequest) => void;
}): React.ReactElement {
  const config = block.config as Record<string, unknown>;
  const ref = normalizeToolRef(config['ref']);
  const directRef = isDirectToolRef(ref);
  const inputMapping = normalizeInputMapping(config['input_mapping']);
  const outputKey = typeof config['output_key'] === 'string' ? config['output_key'] : 'tool_result';
  const selectedDecl =
    !directRef
      ? declaredTools.find((tool) => tool.name === ref.name) ?? null
      : null;
  const unresolvedLocalRef = !directRef && ref.name.length > 0 && selectedDecl === null;

  const updateConfig = React.useCallback(
    (patch: Record<string, unknown>) => {
      useAgentEditorStore.getState().updateBlock(selectedPath, {
        config: { ...config, ...patch },
      });
    },
    [config, selectedPath],
  );

  const updateRef = React.useCallback(
    (nextRef: { type: string; name: string }) => {
      updateConfig({ ref: nextRef });
    },
    [updateConfig],
  );

  const updateMappingEntries = React.useCallback(
    (entries: Array<[string, string]>) => {
      updateConfig({ input_mapping: Object.fromEntries(entries) });
    },
    [updateConfig],
  );

  const mappingEntries = Object.entries(inputMapping);
  const directType: DirectToolRefType =
    ref.type === 'uc_tool' || ref.type === 'enterprise' ? ref.type : 'uc_function';
  const directLabels = DIRECT_TOOL_REF_LABELS[directType];

  return (
    <div>
      <FieldShell label="Workflow tool">
        {declaredTools.length === 0 ? (
          <div className="rounded-db-md border border-dashed border-db-gray-lines p-3 text-center text-[12px] leading-[1.5] text-db-gray-text">
            No workflow tools.
            <button
              type="button"
              onClick={() => onShowAddTool({ mode: 'select-tool-step', blockPath: selectedPath })}
              className="ml-1 font-medium text-db-navy-800 underline"
            >
              Add to workflow
            </button>
          </div>
        ) : (
          <div className="flex gap-2">
            <select
              aria-label="Workflow tool"
              value={directRef ? '' : ref.name}
              onChange={(event) => updateRef({ type: 'builtin', name: event.target.value })}
              className={`${TOOL_STEP_INPUT_CLASS} min-w-0 flex-1`}
            >
              <option value="">{unresolvedLocalRef ? 'Unresolved tool' : 'Select tool'}</option>
              {declaredTools.map((decl) => {
                const kind = findToolKind(registry, decl.kind);
                return (
                  <option key={decl.name} value={decl.name}>
                    {kind?.label ? `${decl.name} · ${kind.label}` : decl.name}
                  </option>
                );
              })}
            </select>
            <button
              type="button"
              onClick={() => onShowAddTool({ mode: 'select-tool-step', blockPath: selectedPath })}
              className="inline-flex shrink-0 items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
            >
              <Plus size={11} /> Add
            </button>
          </div>
        )}
      </FieldShell>

      {unresolvedLocalRef && (
        <div className="mb-3.5 rounded-db-md border border-db-yellow-500 bg-db-yellow-100 px-2.5 py-2 text-[11px] leading-[1.45] text-db-navy-800">
          This step references undeclared workflow tool{' '}
          <code className="font-db-mono">{ref.name}</code>. Select a declared tool or add it to
          the workflow.
        </div>
      )}

      {selectedDecl && (
        <div className="mb-3.5 rounded-db-md border border-db-gray-lines bg-db-oat-light px-2.5 py-2">
          <div className="flex items-center gap-2">
            <LayerChip layer={findToolKind(registry, selectedDecl.kind)?.layer ?? 'D'} />
            <div className="min-w-0 flex-1">
              <div className="truncate font-db-mono text-[12px] font-medium text-db-navy-800">
                {selectedDecl.name}
              </div>
              <div className="truncate text-[10px] text-db-gray-text">
                {findToolKind(registry, selectedDecl.kind)?.label ?? selectedDecl.kind}
                {' · '}
                {toolSummary(selectedDecl)}
              </div>
            </div>
          </div>
          {selectedDecl.description && (
            <p className="mt-1.5 text-[11px] leading-[1.45] text-db-gray-text">
              {selectedDecl.description}
            </p>
          )}
        </div>
      )}

      {directRef && (
        <div className="mb-3.5 rounded-db-md border border-db-yellow-500 bg-db-yellow-100 px-2.5 py-2 text-[11px] leading-[1.45] text-db-navy-800">
          This step uses an imported direct reference. Prefer declaring it as a workflow tool
          when editing this workflow.
        </div>
      )}

      <details
        open={directRef}
        className="mb-3.5 rounded-db-md border border-db-gray-lines bg-white"
      >
        <summary className="cursor-pointer select-none px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.06em] text-db-navy-700 hover:bg-db-oat-light">
          Advanced direct reference
        </summary>
        <div className="border-t border-db-gray-lines px-3 pt-3">
          <FieldShell label="Direct reference type">
            <select
              aria-label="Direct reference type"
              value={directType}
              onChange={(event) => {
                const nextType = event.target.value as DirectToolRefType;
                updateRef({ type: nextType, name: directRef ? ref.name : '' });
              }}
              className={TOOL_STEP_INPUT_CLASS}
            >
              <option value="uc_function">Unity Catalog function</option>
              <option value="uc_tool">UC tool</option>
              <option value="enterprise">Enterprise tool</option>
            </select>
          </FieldShell>
          <FieldShell label={directLabels.nameLabel}>
            <input
              aria-label={directLabels.nameLabel}
              value={directRef ? ref.name : ''}
              onChange={(event) => updateRef({ type: directType, name: event.target.value })}
              placeholder={directLabels.placeholder}
              className={TOOL_STEP_MONO_INPUT_CLASS}
            />
          </FieldShell>
          <button
            type="button"
            onClick={() => onShowAddTool({ mode: 'select-tool-step', blockPath: selectedPath })}
            className="mb-3 inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
          >
            <Plus size={11} /> Convert to workflow tool
          </button>
        </div>
      </details>

      <FieldShell label="Parameters">
        <div className="space-y-2 rounded-db-md border border-db-gray-lines bg-db-oat-light p-2.5">
          {mappingEntries.length === 0 && (
            <p className="text-[11px] italic text-db-gray-text">No parameter mappings.</p>
          )}
          {mappingEntries.map(([argName, stateKey], index) => (
            <div key={`${argName}-${index}`} className="grid grid-cols-[1fr_1fr_auto] gap-2">
              <input
                aria-label={`Parameter ${index + 1}`}
                value={argName}
                onChange={(event) => {
                  const next = [...mappingEntries];
                  next[index] = [event.target.value, stateKey];
                  updateMappingEntries(next);
                }}
                placeholder="parameter"
                className={TOOL_STEP_MONO_INPUT_CLASS}
              />
              <input
                aria-label={`State key for ${argName || `parameter ${index + 1}`}`}
                value={stateKey}
                onChange={(event) => {
                  const next = [...mappingEntries];
                  next[index] = [argName, event.target.value];
                  updateMappingEntries(next);
                }}
                placeholder="state_key"
                className={TOOL_STEP_MONO_INPUT_CLASS}
              />
              <button
                type="button"
                aria-label={`Remove ${argName || `parameter ${index + 1}`}`}
                onClick={() => {
                  updateMappingEntries(mappingEntries.filter((_, i) => i !== index));
                }}
                className="inline-flex h-7 w-7 items-center justify-center rounded-db-md border border-transparent text-db-gray-text transition-colors hover:border-db-lava-400 hover:bg-db-lava-300 hover:text-db-lava-800"
              >
                <X size={13} />
              </button>
            </div>
          ))}
          <button
            type="button"
            aria-label="Add parameter mapping"
            onClick={() => {
              updateConfig({
                input_mapping: {
                  ...inputMapping,
                  [uniqueMappingKey(inputMapping)]: '',
                },
              });
            }}
            className="inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
          >
            <Plus size={11} /> Add parameter
          </button>
        </div>
      </FieldShell>

      <FieldShell label="Output key">
        <input
          aria-label="Output key"
          value={outputKey}
          onChange={(event) => updateConfig({ output_key: event.target.value })}
          className={TOOL_STEP_MONO_INPUT_CLASS}
        />
      </FieldShell>

      <SchemaField
        name="output_schema"
        schema={{ type: 'object', title: 'Output schema', 'x-widget': 'json' }}
        value={config['output_schema']}
        onChange={(value) => updateConfig({ output_schema: value })}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tools tab (agent only) — bind/unbind workflow tools
// ---------------------------------------------------------------------------

interface ToolsBindingFormProps {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  boundToolNames: string[];
  onShowAddTool: (request: AddToolRequest) => void;
}

function ToolsBindingForm({
  block: _block,
  selectedPath,
  registry,
  declaredTools,
  boundToolNames,
  onShowAddTool,
}: ToolsBindingFormProps): React.ReactElement {
  const ast = useAgentEditorStore((s) => s.ast);
  const agentId = useAgentEditorStore((s) => s.agentId);
  const setAst = useAgentEditorStore((s) => s.setAst);
  const [catalogBusy, setCatalogBusy] = React.useState<'refresh' | 'probe' | null>(null);
  const [catalogEditing, setCatalogEditing] = React.useState(false);
  const [catalogDraft, setCatalogDraft] = React.useState('');
  const [probeSamples, setProbeSamples] = React.useState<ProbeSample[]>([]);
  const [catalogError, setCatalogError] = React.useState<string | null>(null);
  // Which declared tool (if any) is expanded for inline config editing in this tab.
  const [editingToolName, setEditingToolName] = React.useState<string | null>(null);
  const editingTool = declaredTools.find((tool) => tool.name === editingToolName) ?? null;
  const extras =
    _block.config && typeof _block.config['extras'] === 'object' && _block.config['extras'] !== null
      ? (_block.config['extras'] as Record<string, unknown>)
      : {};
  const catalogText = typeof extras['_framework_tool_catalog'] === 'string'
    ? extras['_framework_tool_catalog']
    : '';
  const renderError = typeof extras['_framework_tool_catalog_render_error'] === 'string'
    ? extras['_framework_tool_catalog_render_error']
    : '';

  React.useEffect(() => {
    if (!catalogEditing) {
      setCatalogDraft(catalogText);
    }
  }, [catalogEditing, catalogText]);

  // Close the inline tool editor if its tool was removed/renamed elsewhere, or
  // when a different node is selected. ConfigPanel reads selection from the
  // store (not props), so this component is NOT remounted on node switch —
  // local state must be reset deliberately.
  React.useEffect(() => {
    if (editingToolName && !declaredTools.some((tool) => tool.name === editingToolName)) {
      setEditingToolName(null);
    }
  }, [declaredTools, editingToolName]);
  React.useEffect(() => {
    setEditingToolName(null);
  }, [selectedPath]);

  const toggleBinding = (name: string) => {
    const store = useAgentEditorStore.getState();
    if (boundToolNames.includes(name)) {
      // Unbind by editing block.config.tools directly
      const next = boundToolNames.filter((n) => n !== name);
      store.updateBlock(selectedPath, { config: { ..._block.config, tools: next } });
    } else {
      store.bindToolToBlock(selectedPath, name);
    }
  };

  const handleRefreshCatalog = async () => {
    if (!ast) return;
    setCatalogError(null);
    setCatalogBusy('refresh');
    try {
      const result = await refreshCatalog({ definition: ast, agentId, forceRegen: true });
      setAst(result.definition);
    } catch (err) {
      setCatalogError(err instanceof Error ? err.message : 'Catalog refresh failed');
    } finally {
      setCatalogBusy(null);
    }
  };

  const handleProbeTools = async () => {
    if (!ast || boundToolNames.length === 0) return;
    setCatalogError(null);
    setCatalogBusy('probe');
    try {
      const result = await probeTools({
        definition: ast,
        agentId,
        toolNames: boundToolNames,
        persist: false,
      });
      setProbeSamples(result.samples);
    } catch (err) {
      setCatalogError(err instanceof Error ? err.message : 'Tool probe failed');
    } finally {
      setCatalogBusy(null);
    }
  };

  const handleSaveCatalogEdit = () => {
    const nextExtras = {
      ...extras,
      '_framework_tool_catalog': catalogDraft,
      '_framework_tool_catalog_user_edited': true,
      '_framework_tool_catalog_render_error': null,
      '_framework_tool_catalog_injection_enabled': true,
    };
    useAgentEditorStore.getState().updateBlock(selectedPath, {
      config: { ..._block.config, extras: nextExtras },
    });
    setCatalogEditing(false);
  };

  // One bind row (a <div> holding two sibling buttons: the bind toggle and the
  // inline-config edit toggle). Rendered grouped by layer below.
  const renderToolRow = (decl: ToolDecl): React.ReactElement => {
    const bound = boundToolNames.includes(decl.name);
    const kind = findToolKind(registry, decl.kind);
    const requiresApproval = toolRequiresApproval(decl);
    const editing = editingToolName === decl.name;
    return (
      <div
        key={decl.name}
        className={`flex items-center gap-1 rounded-db-md border pr-1 transition-colors ${
          bound || editing
            ? 'border-db-navy-300 bg-db-oat-medium'
            : 'border-transparent hover:bg-db-oat-light'
        }`}
      >
        <button
          type="button"
          onClick={() => toggleBinding(decl.name)}
          className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left"
        >
          <LayerChip layer={kind?.layer ?? 'A'} />
          <Wrench size={14} className="text-db-navy-800" />
          <div className="min-w-0 flex-1">
            <div className="truncate font-db-mono text-[12px] font-medium text-db-navy-800">
              {decl.name}
            </div>
            {kind?.label && (
              <div className="truncate text-[10px] text-db-gray-text">{kind.label}</div>
            )}
          </div>
          {requiresApproval && <Lock size={11} className="text-db-yellow-700" />}
          {bound && <Check size={12} className="text-db-green-700" strokeWidth={2.5} />}
        </button>
        <button
          type="button"
          aria-label="Edit tool config"
          aria-expanded={editing}
          title="Edit tool config"
          onClick={() =>
            setEditingToolName((prev) => (prev === decl.name ? null : decl.name))
          }
          className={`rounded p-1 transition-colors hover:bg-db-oat-light ${
            editing ? 'text-db-navy-800' : 'text-db-gray-text'
          }`}
        >
          <Pencil size={12} />
        </button>
      </div>
    );
  };

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <span className="font-db-sans text-[10px] font-semibold uppercase tracking-[0.06em] text-db-navy-400">
          Built-in &amp; custom
        </span>
        <span className="ml-auto font-db-mono text-[11px] text-db-gray-text">
          {boundToolNames.length} on
        </span>
      </div>
      <div className="mb-2.5 text-[11px] leading-[1.5] text-db-gray-text">
        Bind tools to this agent. Editing a tool&apos;s config changes it for every agent bound
        to it.
      </div>
      {declaredTools.length === 0 ? (
        <div className="rounded-db-md border border-dashed border-db-gray-lines p-4 text-center text-[12px] leading-[1.55] text-db-gray-text">
          No tools yet.
          <br />
          Click{' '}
          <button
            type="button"
            onClick={() => onShowAddTool({ mode: 'bind-agent', blockPath: selectedPath })}
            className="font-medium text-db-navy-800 underline"
          >
            + Add to workflow
          </button>{' '}
          to wire a builtin, MCP server, or @tool function.
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {groupDeclaredToolsByLayer(declaredTools, registry).map(([groupLabel, decls]) => (
            <div key={groupLabel}>
              <div className="px-1 pb-1 font-db-mono text-[10.5px] text-db-gray-text">
                {groupLabel}
              </div>
              <div className="flex flex-col gap-1">{decls.map((decl) => renderToolRow(decl))}</div>
            </div>
          ))}
        </div>
      )}
      {editingTool && (
        <ToolDeclarationEditor
          key={editingTool.name}
          tool={editingTool}
          registry={registry}
          onRename={setEditingToolName}
          onClose={() => setEditingToolName(null)}
        />
      )}
      <button
        type="button"
        onClick={() => onShowAddTool({ mode: 'bind-agent', blockPath: selectedPath })}
        className="mt-3 inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
      >
        <Plus size={11} /> Add to workflow
      </button>
      {boundToolNames.length > 0 && (
        <div className="mt-4 border-t border-db-gray-lines pt-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="text-[12px] font-medium text-db-navy-800">Catalog Preview</h3>
            <div className="flex gap-1.5">
              {catalogEditing ? (
                <>
                  <button
                    type="button"
                    onClick={() => {
                      setCatalogDraft(catalogText);
                      setCatalogEditing(false);
                    }}
                    disabled={catalogBusy !== null}
                    className="inline-flex h-7 items-center gap-1 rounded-db-md border border-db-gray-lines bg-white px-2 text-[11px] font-medium text-db-navy-800 hover:bg-db-oat-light disabled:opacity-60"
                  >
                    <X size={12} /> Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleSaveCatalogEdit}
                    disabled={catalogBusy !== null}
                    className="inline-flex h-7 items-center gap-1 rounded-db-md border border-db-gray-lines bg-white px-2 text-[11px] font-medium text-db-navy-800 hover:bg-db-oat-light disabled:opacity-60"
                  >
                    <Save size={12} /> Save
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => {
                    setCatalogDraft(catalogText);
                    setCatalogEditing(true);
                  }}
                  disabled={catalogBusy !== null}
                  className="inline-flex h-7 items-center gap-1 rounded-db-md border border-db-gray-lines bg-white px-2 text-[11px] font-medium text-db-navy-800 hover:bg-db-oat-light disabled:opacity-60"
                >
                  <Pencil size={12} /> Edit
                </button>
              )}
              <button
                type="button"
                onClick={handleRefreshCatalog}
                disabled={catalogBusy !== null}
                className="inline-flex h-7 items-center gap-1 rounded-db-md border border-db-gray-lines bg-white px-2 text-[11px] font-medium text-db-navy-800 hover:bg-db-oat-light disabled:opacity-60"
              >
                <RefreshCw size={12} /> Refresh
              </button>
              <button
                type="button"
                onClick={handleProbeTools}
                disabled={catalogBusy !== null}
                className="inline-flex h-7 items-center gap-1 rounded-db-md border border-db-gray-lines bg-white px-2 text-[11px] font-medium text-db-navy-800 hover:bg-db-oat-light disabled:opacity-60"
              >
                <FlaskConical size={12} /> Probe
              </button>
            </div>
          </div>
          {(renderError || catalogError) && (
            <div className="mb-2 rounded-db-md border border-db-lava-200 bg-db-lava-50 px-2 py-1.5 text-[11px] text-db-lava-700">
              {catalogError || renderError}
            </div>
          )}
          {catalogEditing ? (
            <textarea
              value={catalogDraft}
              onChange={(event) => setCatalogDraft(event.target.value)}
              className="min-h-44 w-full resize-y rounded-db-md border border-db-gray-lines bg-white p-2 font-db-mono text-[11px] leading-[1.45] text-db-navy-800 outline-none focus:border-db-navy-300"
            />
          ) : (
            <pre className="max-h-56 overflow-auto rounded-db-md border border-db-gray-lines bg-db-oat-light p-2 font-db-mono text-[11px] leading-[1.45] text-db-navy-800">
              {catalogText || 'Catalog will be generated on save.'}
            </pre>
          )}
          {probeSamples.length > 0 && (
            <div className="mt-2 flex flex-col gap-1">
              {probeSamples.map((sample, idx) => {
                const toolName = boundToolNames[idx] ?? `tool_${idx + 1}`;
                return (
                  <div
                    key={`${toolName}-${sample.status}-${idx}`}
                    className="rounded-db-md border border-db-gray-lines px-2 py-1.5 text-[11px] text-db-gray-text"
                  >
                    <span className="font-db-mono font-medium text-db-navy-800">
                      {toolName}
                    </span>
                    <span> - {sample.status}</span>
                    {sample.reason ? <span> - {sample.reason}</span> : null}
                    {sample.sample_output ? (
                      <div className="mt-1 truncate font-db-mono">{sample.sample_output}</div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tools tab (Direction 1, agent only) — a single grouped scroll. Declared tools
// are bucketed by layer (Web / Knowledge / Data / Files / Custom) as a bind
// checklist (reusing ToolsBindingForm), and discovered MCP servers render as
// expandable cards bound at server granularity (config.mcp_servers). Approvals
// are no longer folded in here — they have their own tab again.
// ---------------------------------------------------------------------------

interface ToolsPaneProps {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  boundToolNames: string[];
  onShowAddTool: (request: AddToolRequest) => void;
}

function ToolsPane({
  block,
  selectedPath,
  registry,
  declaredTools,
  boundToolNames,
  onShowAddTool,
}: ToolsPaneProps): React.ReactElement {
  const boundMcpServers = Array.isArray(block.config['mcp_servers'])
    ? (block.config['mcp_servers'] as string[])
    : [];

  // Discover MCP servers (graceful: empty on error → the MCP section just shows
  // the connect affordance).
  const [mcpServers, setMcpServers] = React.useState<DesignerResource[]>([]);
  React.useEffect(() => {
    let cancelled = false;
    listDesignerResources(['mcp_server'])
      .then((res) => {
        if (!cancelled) {
          setMcpServers(res.resources.filter((r) => r.kind === 'mcp_server'));
        }
      })
      .catch(() => {
        /* graceful: no MCP cards */
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPath]);

  const toggleMcpServer = (name: string) => {
    const next = boundMcpServers.includes(name)
      ? boundMcpServers.filter((n) => n !== name)
      : [...boundMcpServers, name];
    useAgentEditorStore.getState().updateBlock(selectedPath, {
      config: { ...block.config, mcp_servers: next },
    });
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Built-in & custom tools — grouped bind checklist */}
      <ToolsBindingForm
        block={block}
        selectedPath={selectedPath}
        registry={registry}
        declaredTools={declaredTools}
        boundToolNames={boundToolNames}
        onShowAddTool={onShowAddTool}
      />

      {/* MCP servers — expandable cards, bound at server granularity */}
      <div>
        <div className="mb-2 flex items-center gap-2">
          <span className="font-db-sans text-[10px] font-semibold uppercase tracking-[0.06em] text-db-navy-400">
            MCP servers
          </span>
          {mcpServers.length > 0 && (
            <span className="ml-auto font-db-mono text-[11px] text-db-gray-text">
              {boundMcpServers.length}/{mcpServers.length}
            </span>
          )}
        </div>
        {mcpServers.length === 0 ? (
          <div className="rounded-db-md border border-dashed border-db-gray-lines p-3 text-center text-[11px] leading-[1.5] text-db-gray-text">
            No MCP servers discovered. Connect one to expose its tools to this agent.
          </div>
        ) : (
          mcpServers.map((srv) => (
            <McpServerCard
              key={srv.name}
              server={srv}
              bound={boundMcpServers.includes(srv.name)}
              onToggle={() => toggleMcpServer(srv.name)}
            />
          ))
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MCP server card — expandable. The header carries identity + status + a master
// bind/unbind toggle; expanding reveals connection details. MCP binds at SERVER
// granularity (config.mcp_servers), so there are no per-tool checkboxes — most
// servers expose a single tool and the backend binds the whole server.
// ---------------------------------------------------------------------------

function McpServerCard({
  server,
  bound,
  onToggle,
}: {
  server: DesignerResource;
  bound: boolean;
  onToggle: () => void;
}): React.ReactElement {
  const connected = server.status === 'connected' || server.status === 'ready';
  const [open, setOpen] = React.useState(connected);
  const metaUrl = typeof server.metadata['url'] === 'string' ? server.metadata['url'] : '';
  const url = metaUrl || (server.full_name ?? '');
  const statusLabel = server.status ?? 'available';

  return (
    <div className="mb-2 overflow-hidden rounded-db-md border border-db-gray-lines bg-white">
      <div className="flex items-center bg-db-oat-light">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-2 text-left"
        >
          <ChevronRight
            size={13}
            className={`shrink-0 text-db-gray-text transition-transform ${open ? 'rotate-90' : ''}`}
            aria-hidden="true"
          />
          <Boxes size={15} className="shrink-0 text-db-navy-800" />
          <span className="min-w-0 flex-1 truncate font-db-mono text-[13px] font-medium text-db-navy-800">
            {server.name}
          </span>
          <span
            className={`inline-flex shrink-0 items-center gap-1 rounded-db-pill px-2 py-0.5 text-[10px] ${
              connected ? 'bg-db-green-300 text-db-green-800' : 'bg-db-oat-medium text-db-gray-text'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-db-green-700' : 'bg-db-navy-300'}`}
            />
            {statusLabel}
          </span>
        </button>
        <button
          type="button"
          onClick={onToggle}
          aria-label={bound ? `Unbind ${server.name}` : `Bind ${server.name}`}
          className={`mr-2 inline-flex shrink-0 items-center gap-1 rounded-db-md border px-2 py-1 text-[11px] font-medium transition-colors ${
            bound
              ? 'border-db-navy-300 bg-db-oat-medium text-db-navy-800 hover:bg-white'
              : 'border-db-navy-800 bg-db-navy-800 text-white hover:bg-db-navy-900'
          }`}
        >
          {bound ? (
            <>
              <Check size={12} strokeWidth={2.5} /> Bound
            </>
          ) : (
            <>
              <Plus size={12} /> Bind
            </>
          )}
        </button>
      </div>
      {open && (
        <div className="border-t border-db-gray-lines px-3 py-2.5">
          {url && (
            <div className="flex items-center gap-1.5 font-db-mono text-[10.5px] text-db-gray-text">
              <Link size={11} className="shrink-0" aria-hidden="true" />
              <span className="truncate">{url}</span>
            </div>
          )}
          {server.description && (
            <p className="mt-1 text-[11px] leading-[1.5] text-db-gray-text">{server.description}</p>
          )}
          <p className="mt-1.5 text-[11px] leading-[1.5] text-db-gray-text">
            Binding exposes this server&apos;s tools to the agent during its ReAct loop. MCP
            servers are bound at server granularity.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Approvals tab — HITL config (presentational; values persist via updateBlock)
// ---------------------------------------------------------------------------

interface ApprovalsFormProps {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  boundToolNames: string[];
}

function ApprovalsForm({
  block,
  selectedPath,
  registry,
  declaredTools,
  boundToolNames,
}: ApprovalsFormProps): React.ReactElement {
  const cfg = block.config as Record<string, unknown>;
  const timeout = (cfg['approval_timeout_seconds'] as number | undefined) ?? 300;
  const broker = (cfg['approval_broker'] as string | undefined) ?? 'InProcessApprovalBroker';

  const updateConfig = (patch: Record<string, unknown>) => {
    useAgentEditorStore.getState().updateBlock(selectedPath, {
      config: { ...cfg, ...patch },
    });
  };

  const gatedTools = declaredTools.filter(
    (t) => boundToolNames.includes(t.name) && toolRequiresApproval(t),
  );

  return (
    <div>
      <div className="mb-3 text-[12px] leading-[1.55] text-db-gray-text">
        Tools marked{' '}
        <code className="rounded bg-db-oat-medium px-1 py-px font-db-mono text-[11px] text-db-navy-800">
          requires_approval
        </code>{' '}
        pause the ReAct loop and emit a{' '}
        <code className="rounded bg-db-oat-medium px-1 py-px font-db-mono text-[11px] text-db-navy-800">
          GateWaitingEvent
        </code>
        . Only the originating user may resolve.
      </div>

      <FieldShell label="Approval timeout" hint="Seconds before the gate auto-denies.">
        <input
          type="number"
          value={timeout}
          onChange={(e) => updateConfig({ approval_timeout_seconds: parseInt(e.target.value, 10) || 0 })}
          className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus"
        />
      </FieldShell>

      <FieldShell label="Approval broker">
        <select
          value={broker}
          onChange={(e) => updateConfig({ approval_broker: e.target.value })}
          className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[13px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus"
        >
          <option value="InProcessApprovalBroker">InProcessApprovalBroker</option>
          <option value="DeltaApprovalBroker">DeltaApprovalBroker</option>
        </select>
      </FieldShell>

      <div className="mt-3.5 font-db-sans text-[11px] font-medium uppercase tracking-[0.06em] text-db-navy-800">
        Gated tools on this agent
      </div>
      <div className="mt-2 flex flex-col gap-1.5">
        {gatedTools.length === 0 && (
          <div className="rounded-db-md bg-db-oat-light px-2.5 py-2 text-[12px] text-db-gray-text">
            No gated tools bound.
          </div>
        )}
        {gatedTools.map((decl) => {
          const reason =
            (decl.config?.['approval_reason'] as string | undefined) ??
            'Requires human approval before execution.';
          const kind = findToolKind(registry, decl.kind);
          return (
            <div
              key={decl.name}
              className="flex items-center gap-2 rounded-db-md border border-db-yellow-300 bg-db-yellow-300/40 px-2.5 py-2"
            >
              <Lock size={12} className="text-db-yellow-800" />
              <div className="min-w-0 flex-1">
                <div className="truncate font-db-mono text-[12px] text-db-navy-800">
                  {decl.name}
                </div>
                <div className="truncate text-[10px] text-db-yellow-800">
                  {reason}
                  {kind?.label ? ` · ${kind.label}` : ''}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tools registry list (workspace tools view)
// ---------------------------------------------------------------------------

interface ToolsRegistryListProps {
  tools: ToolDecl[];
  registry: RegistryResponse;
  mode: 'workspace' | 'binding';
  onRemove?: (name: string) => void;
  selectedName?: string | null;
  onSelect?: (name: string) => void;
}

function ToolsRegistryList({
  tools,
  registry,
  mode,
  onRemove,
  selectedName,
  onSelect,
}: ToolsRegistryListProps): React.ReactElement {
  if (tools.length === 0) {
    return (
      <div className="rounded-db-md border border-dashed border-db-gray-lines p-4 text-center text-[12px] leading-[1.55] text-db-gray-text">
        No tools yet.
        <br />
        Click <strong className="font-medium text-db-navy-800">+ Add</strong> to wire a builtin,
        MCP server, or @tool function.
      </div>
    );
  }
  const removeTool = useAgentEditorStore.getState().removeTool;
  const handleRemove = onRemove ?? ((name: string) => removeTool(name));
  return (
    <div className="flex flex-col gap-1">
      {tools.map((decl) => {
        const kind = findToolKind(registry, decl.kind);
        const requiresApproval = toolRequiresApproval(decl);
        const selected = selectedName === decl.name;
        return (
          <div
            key={decl.name}
            role={mode === 'workspace' ? 'button' : undefined}
            tabIndex={mode === 'workspace' ? 0 : undefined}
            onClick={() => onSelect?.(decl.name)}
            onKeyDown={(event) => {
              if (mode === 'workspace' && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault();
                onSelect?.(decl.name);
              }
            }}
            className={`flex items-center gap-2 rounded-db-md px-2 py-1.5 ${
              selected ? 'bg-db-oat-medium ring-1 ring-db-navy-300' : 'hover:bg-db-oat-light'
            }`}
          >
            <LayerChip layer={kind?.layer ?? 'A'} />
            <Wrench size={14} className="text-db-gray-text" />
            <div className="min-w-0 flex-1">
              <div className="truncate font-db-mono text-[12px] text-db-navy-800">
                {decl.name}
              </div>
              {kind?.label && (
                <div className="truncate text-[10px] text-db-gray-text">
                  {kind.label} · {toolSummary(decl)}
                </div>
              )}
            </div>
            {requiresApproval && <Lock size={11} className="text-db-yellow-700" />}
            {mode === 'workspace' && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove(decl.name);
                }}
                title="Remove tool from workflow"
                className="rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
              >
                <X size={11} />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tool declaration editor
// ---------------------------------------------------------------------------

function ToolDeclarationEditor({
  tool,
  registry,
  onRename,
  onClose,
}: {
  tool: ToolDecl;
  registry: RegistryResponse;
  onRename: (name: string) => void;
  onClose: () => void;
}): React.ReactElement {
  const kind = findToolKind(registry, tool.kind);
  const schema = kind?.config_schema ?? null;
  const schemaProps = schemaProperties(schema);
  const [name, setName] = React.useState(tool.name);
  const [nameError, setNameError] = React.useState<string | null>(null);
  const [configErrors, setConfigErrors] = React.useState<Record<string, string[]>>({});

  React.useEffect(() => {
    setName(tool.name);
    setConfigErrors(requiredConfigErrors(schema, tool.config ?? {}));
  }, [schema, tool.config, tool.name]);

  const applyName = React.useCallback(() => {
    const nextName = name.trim();
    if (!nextName) {
      setNameError('Tool name is required');
      return;
    }
    if (nextName === tool.name) {
      setNameError(null);
      return;
    }
    const ok = useAgentEditorStore.getState().updateTool(tool.name, { name: nextName });
    if (!ok) {
      setNameError('Tool name already exists');
      return;
    }
    onRename(nextName);
    setNameError(null);
  }, [name, onRename, tool.name]);

  const updateConfig = (fieldName: string, fieldValue: unknown) => {
    const nextConfig = { ...(tool.config ?? {}), [fieldName]: fieldValue };
    const errors = requiredConfigErrors(schema, nextConfig);
    setConfigErrors(errors);
    useAgentEditorStore.getState().updateTool(tool.name, { config: nextConfig });
  };

  return (
    <div className="mt-3 border-t border-db-gray-lines pt-3">
      <div className="mb-2 flex items-center gap-2">
        <Wrench size={14} className="text-db-navy-800" />
        <span className="text-[13px] font-medium text-db-navy-800">Configure tool</span>
        <button
          type="button"
          onClick={onClose}
          className="ml-auto rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium"
          aria-label="Close tool editor"
        >
          <X size={12} />
        </button>
      </div>

      <FieldShell label="Tool name" required>
        <input
          value={name}
          onChange={(event) => {
            setName(event.target.value);
            setNameError(null);
          }}
          onBlur={applyName}
          className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus"
        />
        {nameError && <p className="mt-1 text-[11px] text-db-lava-700">{nameError}</p>}
      </FieldShell>

      <FieldShell label="Kind">
        <input
          value={kind?.label ? `${kind.label} (${tool.kind})` : tool.kind}
          readOnly
          className="w-full rounded-db-md border border-db-gray-lines bg-db-oat-light px-2.5 py-1.5 font-db-mono text-[12px] text-db-gray-text"
        />
      </FieldShell>

      <FieldShell label="Description">
        <textarea
          value={tool.description ?? ''}
          onChange={(event) => {
            useAgentEditorStore.getState().updateTool(tool.name, {
              description: event.target.value,
            });
          }}
          rows={2}
          className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[12px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus"
        />
      </FieldShell>

      {Object.entries(schemaProps).map(([fieldName, fieldSchema]) => (
        <SchemaField
          key={fieldName}
          name={fieldName}
          schema={fieldSchema}
          value={tool.config?.[fieldName]}
          onChange={(value) => updateConfig(fieldName, value)}
          required={Array.isArray(schema?.['required']) && (schema['required'] as string[]).includes(fieldName)}
          errors={configErrors[fieldName] ?? []}
        />
      ))}

      <button
        type="button"
        onClick={() => {
          useAgentEditorStore.getState().removeTool(tool.name);
          onClose();
        }}
        title="Remove this tool from the workflow"
        className="mt-2 inline-flex items-center gap-1 rounded-db-md bg-db-lava-300 px-2 py-1 text-[11px] font-medium text-db-lava-800 transition-colors hover:bg-db-lava-400"
      >
        <X size={11} /> Remove
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Field shell (label + hint)
// ---------------------------------------------------------------------------

function FieldShell({
  label,
  hint,
  required,
  children,
}: {
  label: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <div className="mb-3.5">
      <label className="mb-1 flex items-center gap-1 text-[12px] font-medium text-db-navy-800">
        {label}
        {required && <span className="text-db-lava-600">*</span>}
      </label>
      {children}
      {hint && (
        <div className="mt-1 flex items-center gap-1 text-[11px] text-db-gray-text">
          <Info size={11} /> {hint}
        </div>
      )}
    </div>
  );
}

// Suppress unused-import warning when Box icon is referenced only for typing
void Box;
