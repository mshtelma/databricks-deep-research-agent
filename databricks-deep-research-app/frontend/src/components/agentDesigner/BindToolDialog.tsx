/**
 * BindToolDialog — Radix Dialog for binding/unbinding declared tools to an agent block.
 *
 * Design choice: we use `updateBlock(blockPath, { config: { tools: <selected names> } })` to persist
 * the full selected set in a single call. `bindToolToBlock` is not used here because it
 * only handles additions (one at a time) and the store's updateBlock is sufficient to
 * represent both additions and removals atomically.
 */

import * as React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { Wrench, X as CloseIcon } from 'lucide-react';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { resolveBlock } from '@/lib/blockPath';
import type { BlockPath } from '@/types/ast';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface BindToolDialogProps {
  blockPath: BlockPath;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function BindToolDialog({
  blockPath,
  open,
  onOpenChange,
}: BindToolDialogProps): React.ReactElement {
  const ast = useAgentEditorStore((s) => s.ast);

  // Resolve block's current bound tools
  const block = React.useMemo(() => {
    if (!ast) return null;
    return resolveBlock(ast, blockPath);
  }, [ast, blockPath]);

  const allTools = ast?.tools ?? [];
  const boundTools = React.useMemo(
    () => Array.isArray(block?.config.tools) ? block.config.tools as string[] : [],
    [block],
  );
  const initialChecked: Set<string> = new Set(boundTools);

  const [checked, setChecked] = React.useState<Set<string>>(initialChecked);

  // Re-sync checked state when dialog opens or block changes
  React.useEffect(() => {
    if (open) {
      setChecked(new Set(boundTools));
    }
  }, [open, boundTools]);

  const handleToggle = React.useCallback((name: string) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }, []);

  const handleSubmit = React.useCallback(() => {
    const selectedNames = Array.from(checked);
    useAgentEditorStore.getState().updateBlock(blockPath, {
      config: {
        ...(block?.config ?? {}),
        tools: selectedNames,
      },
    });
    onOpenChange(false);
  }, [checked, blockPath, block, onOpenChange]);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-db-navy-900/30 backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          className="db-root fixed left-1/2 top-1/2 z-50 flex max-h-[80vh] w-full max-w-md -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-db-lg border border-db-gray-lines bg-white font-db-sans shadow-db-xl focus:outline-none"
          aria-describedby={undefined}
        >
          <div className="border-b border-db-gray-lines px-[22px] py-[18px]">
            <div className="flex items-center gap-2.5">
              <Wrench size={16} className="text-db-navy-800" />
              <DialogPrimitive.Title className="text-[16px] font-medium text-db-navy-800">
                Bind tools to agent
              </DialogPrimitive.Title>
              <DialogPrimitive.Close
                className="ml-auto rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
                aria-label="Close"
              >
                <CloseIcon size={14} />
              </DialogPrimitive.Close>
            </div>
            <p className="mt-1 text-[12px] text-db-gray-text">
              Select which workflow tools this agent can call during its ReAct loop.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto px-[22px] py-3.5">
            {allTools.length === 0 ? (
              <div className="rounded-db-md border border-dashed border-db-gray-lines p-4 text-center text-[12px] leading-[1.55] text-db-gray-text">
                No tools declared yet. Add tools to the workflow first.
              </div>
            ) : (
              <ul className="flex flex-col gap-1">
                {allTools.map((tool) => {
                  const isChecked = checked.has(tool.name);
                  return (
                    <li
                      key={tool.name}
                      className={`flex items-center gap-2.5 rounded-db-md border px-2.5 py-2 transition-colors ${
                        isChecked
                          ? 'border-db-navy-300 bg-db-oat-medium'
                          : 'border-transparent hover:bg-db-oat-light'
                      }`}
                    >
                      <input
                        id={`bind-tool-${tool.name}`}
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => handleToggle(tool.name)}
                        className="h-3.5 w-3.5 rounded border-db-gray-lines accent-db-lava-600"
                      />
                      <label
                        htmlFor={`bind-tool-${tool.name}`}
                        className="flex-1 cursor-pointer"
                      >
                        <div className="font-db-mono text-[12px] font-medium text-db-navy-800">
                          {tool.name}
                        </div>
                        <div className="font-db-mono text-[10px] text-db-gray-text">
                          {tool.kind}
                        </div>
                      </label>
                    </li>
                  );
                })}
              </ul>
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
              className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700"
            >
              Apply
            </button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
