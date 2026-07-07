import * as React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { Wrench, X as CloseIcon } from 'lucide-react';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { defaultConfigForSchema, requiredConfigErrors, schemaProperties } from '@/lib/jsonSchema';
import type { ToolDecl } from '@/types/ast';
import type { RegistryResponse, ToolKindSpec } from '@/types/agentDesigner';
import { LayerChip } from '../atoms';
import { SchemaField } from '../SchemaField';
import {
  familyForToolKind,
  TOOL_FAMILY_LABELS,
  TOOL_FAMILY_ORDER,
  type ToolFamily,
} from './toolKindFamilies';

export interface ToolDeclarationDialogProps {
  registry: RegistryResponse;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeclared?: (tool: ToolDecl) => void;
}

function firstKindForFamily(registry: RegistryResponse, family: ToolFamily): ToolKindSpec | null {
  return registry.tool_kinds.find((spec) => familyForToolKind(spec) === family) ?? null;
}

function defaultFamily(registry: RegistryResponse): ToolFamily {
  return TOOL_FAMILY_ORDER.find((family) => firstKindForFamily(registry, family)) ?? 'other';
}

export function ToolDeclarationDialog({
  registry,
  open,
  onOpenChange,
  onDeclared,
}: ToolDeclarationDialogProps): React.ReactElement {
  const [selectedFamily, setSelectedFamily] = React.useState<ToolFamily>(() =>
    defaultFamily(registry),
  );
  const [selectedKind, setSelectedKind] = React.useState<string>('');
  const [toolName, setToolName] = React.useState('');
  const [config, setConfig] = React.useState<Record<string, unknown>>({});
  const [configErrors, setConfigErrors] = React.useState<Record<string, string[]>>({});
  const [nameError, setNameError] = React.useState<string | null>(null);

  const familyKinds = React.useMemo(
    () => registry.tool_kinds.filter((spec) => familyForToolKind(spec) === selectedFamily),
    [registry.tool_kinds, selectedFamily],
  );
  const selectedSpec = registry.tool_kinds.find((tk) => tk.kind === selectedKind) ?? null;
  const configSchema = selectedSpec?.config_schema ?? null;
  const schemaProps = schemaProperties(configSchema);

  const setKind = React.useCallback(
    (kind: string) => {
      const spec = registry.tool_kinds.find((tk) => tk.kind === kind) ?? null;
      setSelectedKind(kind);
      setToolName(kind);
      setConfig(defaultConfigForSchema(spec?.config_schema));
      setConfigErrors({});
      setNameError(null);
    },
    [registry.tool_kinds],
  );

  React.useEffect(() => {
    if (!open) return;
    const family = defaultFamily(registry);
    const first = firstKindForFamily(registry, family);
    setSelectedFamily(family);
    setKind(first?.kind ?? '');
  }, [open, registry, setKind]);

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

    onDeclared?.({ kind: selectedKind, name, config });
    onOpenChange(false);
  }, [selectedKind, toolName, config, configSchema, onDeclared, onOpenChange]);

  const availableFamilies = TOOL_FAMILY_ORDER.filter((family) =>
    registry.tool_kinds.some((spec) => familyForToolKind(spec) === family),
  );

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
              Declare a workflow tool, then bind it to agents or call it from tool steps.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto px-[22px] py-4">
            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="mb-1 block text-[12px] font-medium text-db-navy-800">
                  Family/source
                </span>
                <select
                  aria-label="Tool family"
                  value={selectedFamily}
                  onChange={(event) => {
                    const family = event.target.value as ToolFamily;
                    const first = firstKindForFamily(registry, family);
                    setSelectedFamily(family);
                    setKind(first?.kind ?? '');
                  }}
                  className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[13px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus"
                >
                  {availableFamilies.map((family) => (
                    <option key={family} value={family}>
                      {TOOL_FAMILY_LABELS[family]}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-[12px] font-medium text-db-navy-800">
                  Tool kind
                </span>
                <select
                  aria-label="Tool kind"
                  value={selectedKind}
                  onChange={(event) => setKind(event.target.value)}
                  className="w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[13px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus"
                >
                  {familyKinds.map((spec) => (
                    <option key={spec.kind} value={spec.kind}>
                      {spec.label} ({spec.kind})
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {selectedSpec && (
              <div className="mt-3 flex items-center gap-2 rounded-db-md border border-db-gray-lines bg-db-oat-light px-2.5 py-2">
                <LayerChip layer={selectedSpec.layer ?? 'A'} />
                <div className="min-w-0">
                  <div className="truncate text-[12px] font-medium text-db-navy-800">
                    {selectedSpec.label}
                  </div>
                  <div className="truncate font-db-mono text-[10px] text-db-gray-text">
                    {selectedSpec.kind}
                  </div>
                </div>
              </div>
            )}

            {selectedKind && (
              <div className="mt-4 border-t border-db-gray-lines pt-4">
                <div className="mb-3">
                  <label
                    htmlFor="tool-name-input"
                    className="mb-1 flex items-center gap-1 text-[12px] font-medium text-db-navy-800"
                  >
                    Local tool name <span className="text-db-lava-600">*</span>
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
              disabled={!selectedKind}
              className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Add tool
            </button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

