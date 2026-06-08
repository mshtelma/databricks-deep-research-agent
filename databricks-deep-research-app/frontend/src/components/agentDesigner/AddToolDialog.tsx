/**
 * AddToolDialog — Radix Dialog for declaring a new tool in the workflow.
 *
 * Layer colour mapping (used for kind palette grouping headings):
 *   A → blue, B → green, C → purple, D → amber
 *
 * Config form: one SchemaField per top-level property of ToolKindSpec.config_schema.
 */

import * as React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { Wrench, X as CloseIcon } from 'lucide-react';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { defaultConfigForSchema, requiredConfigErrors, schemaProperties } from '@/lib/jsonSchema';
import { SchemaField } from './SchemaField';
import { LayerChip } from './atoms';
import type { RegistryResponse, ToolKindSpec } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Layer helpers
// ---------------------------------------------------------------------------

type Layer = 'A' | 'B' | 'C' | 'D';

function layerForIndex(index: number): Layer {
  if (index < 3) return 'A';
  if (index < 6) return 'B';
  if (index < 9) return 'C';
  return 'D';
}

function layerForSpec(spec: ToolKindSpec, index: number): Layer {
  const layer = spec.layer;
  return layer === 'A' || layer === 'B' || layer === 'C' || layer === 'D'
    ? layer
    : layerForIndex(index);
}

const LAYER_LABELS: Record<Layer, string> = {
  A: 'Web',
  B: 'Knowledge',
  C: 'Data',
  D: 'Filesystem',
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface AddToolDialogProps {
  registry: RegistryResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AddToolDialog({
  registry,
  open,
  onOpenChange,
}: AddToolDialogProps): React.ReactElement {
  const [selectedKind, setSelectedKind] = React.useState<string | null>(null);
  const [toolName, setToolName] = React.useState('');
  const [config, setConfig] = React.useState<Record<string, unknown>>({});
  const [configErrors, setConfigErrors] = React.useState<Record<string, string[]>>({});
  const [nameError, setNameError] = React.useState<string | null>(null);
  const selectedSpec = registry.tool_kinds.find((tk) => tk.kind === selectedKind);
  const configSchema = selectedSpec?.config_schema ?? null;

  // Reset form state whenever dialog opens
  React.useEffect(() => {
    if (open) {
      setSelectedKind(null);
      setToolName('');
      setConfig({});
      setConfigErrors({});
      setNameError(null);
    }
  }, [open]);

  // Auto-fill name when kind changes
  const handleKindSelect = React.useCallback((kind: string) => {
    const spec = registry.tool_kinds.find((tk) => tk.kind === kind);
    setSelectedKind(kind);
    setToolName(kind);
    setConfig(defaultConfigForSchema(spec?.config_schema));
    setConfigErrors({});
    setNameError(null);
  }, [registry.tool_kinds]);

  const handleSubmit = React.useCallback(() => {
    if (!selectedKind) return;

    const name = toolName.trim();
    if (!name) {
      setNameError('Tool name is required');
      return;
    }

    const errors = requiredConfigErrors(configSchema, config);
    setConfigErrors(errors);
    if (Object.keys(errors).length > 0) {
      return;
    }

    const ok = useAgentEditorStore.getState().declareTool(selectedKind, name, config);
    if (!ok) {
      setNameError('Tool name already exists');
      return;
    }

    onOpenChange(false);
  }, [selectedKind, toolName, config, configSchema, onOpenChange]);

  // Group tool_kinds by derived layer
  const grouped = React.useMemo(() => {
    const map = new Map<Layer, Array<{ spec: ToolKindSpec; layer: Layer }>>([
      ['A', []],
      ['B', []],
      ['C', []],
      ['D', []],
    ]);
    registry.tool_kinds.forEach((spec, idx) => {
      const layer = layerForSpec(spec, idx);
      map.get(layer)!.push({ spec, layer });
    });
    return map;
  }, [registry.tool_kinds]);

  const schemaProps = schemaProperties(configSchema);

  const LAYERS: Layer[] = ['A', 'B', 'C', 'D'];

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-db-navy-900/30 backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          className="db-root fixed left-1/2 top-1/2 z-50 flex max-h-[80vh] w-full max-w-[640px] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-db-lg border border-db-gray-lines bg-white font-db-sans shadow-db-xl focus:outline-none"
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
              Pick a builtin, MCP server, or @tool function to expose to your agents.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto px-[22px] py-3">
            {LAYERS.map((layer) => {
              const items = grouped.get(layer) ?? [];
              if (items.length === 0) return null;
              return (
                <div key={layer} className="mt-3.5 first:mt-2">
                  <div className="mb-2 flex items-center gap-2">
                    <LayerChip layer={layer} />
                    <span className="font-db-sans text-[11px] font-medium uppercase tracking-[0.06em] text-db-navy-800">
                      Layer {layer} · {LAYER_LABELS[layer]}
                    </span>
                    <div className="h-px flex-1 bg-db-gray-lines" />
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {items.map(({ spec }) => {
                      const isSelected = selectedKind === spec.kind;
                      return (
                        <button
                          key={spec.kind}
                          type="button"
                          onClick={() => handleKindSelect(spec.kind)}
                          className={[
                            'flex flex-col gap-1 rounded-db-md border bg-white p-2.5 text-left transition-all',
                            isSelected
                              ? 'border-db-navy-400 shadow-db-sm'
                              : 'border-db-gray-lines hover:border-db-navy-300 hover:shadow-db-xs',
                          ].join(' ')}
                        >
                          <div className="flex items-center gap-1.5">
                            <Wrench size={13} className="text-db-navy-800" />
                            <span className="font-db-mono text-[12px] font-medium text-db-navy-800">
                              {spec.label}
                            </span>
                          </div>
                          <div className="font-db-mono text-[10px] text-db-gray-text">
                            {spec.kind}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {/* Name + config form — shown once a kind is selected */}
            {selectedKind !== null && (
              <div className="mt-4 border-t border-db-gray-lines pt-4">
                <div className="mb-3">
                  <label
                    htmlFor="tool-name-input"
                    className="mb-1 flex items-center gap-1 text-[12px] font-medium text-db-navy-800"
                  >
                    Tool name <span className="text-db-lava-600">*</span>
                  </label>
                  <input
                    id="tool-name-input"
                    type="text"
                    value={toolName}
                    onChange={(e) => {
                      setToolName(e.target.value);
                      setNameError(null);
                    }}
                    className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 font-db-mono text-[12px] text-db-navy-800 outline-none transition-colors focus:border-db-navy-400 focus:shadow-db-focus"
                  />
                  {nameError !== null && (
                    <p className="mt-1 text-[11px] text-db-lava-700" role="alert">
                      {nameError}
                    </p>
                  )}
                </div>

                {Object.entries(schemaProps).map(([fieldName, fieldSchema]) => (
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
                ))}
              </div>
            )}
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
              onClick={handleSubmit}
              disabled={selectedKind === null}
              className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:cursor-not-allowed disabled:opacity-55"
            >
              Add tool
            </button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
