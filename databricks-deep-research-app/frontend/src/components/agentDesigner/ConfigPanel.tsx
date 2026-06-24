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
  RefreshCw,
  FlaskConical,
  Pencil,
  Save,
  ChevronRight,
} from 'lucide-react';
import type { RegistryResponse, ToolKindSpec } from '@/types/agentDesigner';
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
    config['max_results'] ??
    config['num_results'];
  return primary === undefined || primary === null || primary === '' ? 'Not configured' : String(primary);
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ConfigPanelProps {
  registry: RegistryResponse;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ConfigPanel({ registry }: ConfigPanelProps): React.ReactElement {
  const selectedPath = useAgentEditorStore((s) => s.selectedPath);
  const ast = useAgentEditorStore((s) => s.ast);

  const [showAddTool, setShowAddTool] = React.useState(false);
  const [selectedToolName, setSelectedToolName] = React.useState<string | null>(null);

  const block = React.useMemo(() => {
    if (!ast || !selectedPath) return null;
    return resolveBlock(ast, selectedPath);
  }, [ast, selectedPath]);

  const declaredTools: ToolDecl[] = React.useMemo(() => ast?.tools ?? [], [ast?.tools]);
  const selectedTool = declaredTools.find((tool) => tool.name === selectedToolName) ?? null;

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
        <aside className="db-root flex w-[340px] shrink-0 flex-col border-l border-db-gray-lines bg-white font-db-sans">
          <div className="flex items-center gap-2 border-b border-db-gray-lines px-4 py-3.5">
            <Wrench size={15} className="text-db-navy-800" />
            <span className="text-[13px] font-medium text-db-navy-800">Workspace tools</span>
            <span className="ml-auto font-db-mono text-[11px] text-db-gray-text">
              {declaredTools.length}
            </span>
            <button
              type="button"
              onClick={() => setShowAddTool(true)}
              className="inline-flex items-center gap-1 rounded-db-md bg-db-lava-600 px-2.5 py-1 text-[12px] font-medium text-white transition-colors hover:bg-db-lava-700"
            >
              <Plus size={11} /> Add
            </button>
          </div>
          <div className="border-b border-db-gray-lines bg-db-oat-light px-3.5 py-2.5 text-[11px] leading-[1.5] text-db-gray-text">
            Tools available to any agent in this workflow. Select an agent block to bind tools to
            it.
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
        </aside>
        {showAddTool && (
          <AddToolDialog open={showAddTool} onOpenChange={setShowAddTool} registry={registry} />
        )}
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
        onShowAddTool={() => setShowAddTool(true)}
      />
      {showAddTool && (
        <AddToolDialog open={showAddTool} onOpenChange={setShowAddTool} registry={registry} />
      )}
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
  onShowAddTool: () => void;
}

type InspectorTab = 'config' | 'tools' | 'hitl';

function SelectedInspector({
  block,
  selectedPath,
  registry,
  declaredTools,
  onShowAddTool,
}: SelectedInspectorProps): React.ReactElement {
  const [tab, setTab] = React.useState<InspectorTab>('config');

  // Reset to "config" tab whenever the selected block changes
  const prevPathRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (selectedPath !== prevPathRef.current) {
      prevPathRef.current = selectedPath;
      setTab('config');
    }
  }, [selectedPath]);

  const isAgent = block.type === 'agent';
  const boundToolNames = Array.isArray(block.config.tools)
    ? (block.config.tools as string[])
    : [];

  return (
    <aside className="db-root flex w-[340px] shrink-0 flex-col border-l border-db-gray-lines bg-white font-db-sans">
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
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        {tab === 'config' && <ConfigureForm block={block} selectedPath={selectedPath} registry={registry} />}
        {tab === 'tools' && isAgent && (
          <ToolsBindingForm
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
    </aside>
  );
}

function InspectorTabButton({
  active,
  onClick,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count?: number;
}): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`-mb-px border-b-2 px-4 py-2.5 font-db-sans text-[13px] font-medium transition-colors ${
        active
          ? 'border-db-lava-600 text-db-navy-800'
          : 'border-transparent text-db-gray-text hover:text-db-navy-800'
      }`}
    >
      {label}
      {count !== undefined && (
        <span className="ml-1 text-[11px] text-db-gray-text">{count}</span>
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
}

function ConfigureForm({ block, selectedPath, registry }: ConfigureFormProps): React.ReactElement {
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
// Tools tab (agent only) — bind/unbind workflow tools
// ---------------------------------------------------------------------------

interface ToolsBindingFormProps {
  block: import('@/types/ast').Block;
  selectedPath: string;
  registry: RegistryResponse;
  declaredTools: ToolDecl[];
  boundToolNames: string[];
  onShowAddTool: () => void;
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

  return (
    <div>
      <div className="mb-2.5 text-[11px] text-db-gray-text">
        Tools the agent can call during its ReAct loop. Click a tool to bind/unbind, or the
        pencil to edit its config. Edits apply to the shared workflow tool — every agent bound
        to it sees the change.
      </div>
      {declaredTools.length === 0 ? (
        <div className="rounded-db-md border border-dashed border-db-gray-lines p-4 text-center text-[12px] leading-[1.55] text-db-gray-text">
          No tools yet.
          <br />
          Click{' '}
          <button
            type="button"
            onClick={onShowAddTool}
            className="font-medium text-db-navy-800 underline"
          >
            + Add to workflow
          </button>{' '}
          to wire a builtin, MCP server, or @tool function.
        </div>
      ) : (
        <div className="flex flex-col gap-1">
          {declaredTools.map((decl) => {
            const bound = boundToolNames.includes(decl.name);
            const kind = findToolKind(registry, decl.kind);
            const requiresApproval = toolRequiresApproval(decl);
            const editing = editingToolName === decl.name;
            return (
              // Row is a <div> (not a <button>) so it can hold two sibling
              // buttons: the bind toggle and the inline-config edit toggle.
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
                  {bound && (
                    <Check size={12} className="text-db-green-700" strokeWidth={2.5} />
                  )}
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
          })}
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
        onClick={onShowAddTool}
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
