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
import { Wrench, Plus, X, Lock, Check, Info, Box } from 'lucide-react';
import type { RegistryResponse, ToolKindSpec } from '@/types/agentDesigner';
import type { ToolDecl } from '@/types/ast';
import { resolveBlock } from '@/lib/blockPath';
import { requiredConfigErrors, schemaProperties } from '@/lib/jsonSchema';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { TypePill, LayerChip } from './atoms';
import { SchemaField } from './SchemaField';
import { AddToolDialog } from './AddToolDialog';

// ---------------------------------------------------------------------------
// AJV factory
// ---------------------------------------------------------------------------

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

  return (
    <>
      {Object.entries(visibleSchemaProps).map(([fieldName, fieldSchema]) => (
        <SchemaField
          key={fieldName}
          name={fieldName}
          schema={fieldSchema}
          value={formValue[fieldName]}
          onChange={(v) => handleFieldChange(fieldName, v)}
          required={requiredKeys.includes(fieldName)}
          errors={errorMap[fieldName] ?? []}
        />
      ))}
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

  return (
    <div>
      <div className="mb-2.5 text-[11px] text-db-gray-text">
        Tools the agent can call during its ReAct loop. Click to bind/unbind.
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
            return (
              <button
                type="button"
                key={decl.name}
                onClick={() => toggleBinding(decl.name)}
                className={`flex items-center gap-2 rounded-db-md border px-2 py-1.5 text-left transition-colors ${
                  bound
                    ? 'border-db-navy-300 bg-db-oat-medium'
                    : 'border-transparent hover:bg-db-oat-light'
                }`}
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
            );
          })}
        </div>
      )}
      <button
        type="button"
        onClick={onShowAddTool}
        className="mt-3 inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[12px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
      >
        <Plus size={11} /> Add to workflow
      </button>
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
