/**
 * ChatAttachmentsSelector — attach Skills to a chat query (E1).
 *
 * Reuses the Designer resource-discovery endpoint (`listDesignerResources`,
 * OBO-scoped) to list the skills available to the user, lets the user toggle
 * which to attach to THIS query, and supports registering a new
 * skill folder inline (persisted via `skillFoldersApi`; applied to the user's
 * next research run by the runtime skill store).
 *
 * The selection is lifted to the parent (MessageInput) and threaded into the
 * QuerySubmission as `enabledSkills`; the backend merges them into the run.
 *
 * Overlay: a portaled, edge-flipping Radix Popover (the `DeployDropdown`
 * pattern). The composer is pinned to the bottom of an `h-screen` column, so a
 * naive `absolute mt-1` panel opened downward and spilled past the viewport,
 * forcing the toolbar to reflow. Portaling out of the flex flow + `side="top"`
 * + a height capped to the available space keeps the panel on-screen.
 */

import * as React from 'react';
import * as Popover from '@radix-ui/react-popover';
import { listDesignerResources } from '@/api/agentDesigner';
import { skillFoldersApi } from '@/api/client';

export interface ChatAttachmentsSelectorProps {
  selectedSkills: string[];
  onChange: (nextSkills: string[]) => void;
  disabled?: boolean;
}

function toggle(list: string[], value: string): string[] {
  return list.includes(value)
    ? list.filter((v) => v !== value)
    : [...list, value];
}

export function ChatAttachmentsSelector({
  selectedSkills,
  onChange,
  disabled = false,
}: ChatAttachmentsSelectorProps): React.ReactElement | null {
  const [open, setOpen] = React.useState(false);
  const [skills, setSkills] = React.useState<string[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [folderPath, setFolderPath] = React.useState('');
  const [folderBusy, setFolderBusy] = React.useState(false);
  const [folderError, setFolderError] = React.useState<string | null>(null);

  const load = React.useCallback(() => {
    setLoading(true);
    listDesignerResources(['skill'])
      .then((res) => {
        setSkills(
          res.resources.filter((r) => r.kind === 'skill').map((r) => r.name),
        );
      })
      .catch(() => {
        /* discovery is best-effort; leave lists empty */
      })
      .finally(() => setLoading(false));
  }, []);

  React.useEffect(() => {
    if (open && !loading && skills.length === 0) {
      load();
    }
  }, [open, load, loading, skills.length]);

  const selectedCount = selectedSkills.length;

  const handleAddFolder = async () => {
    const path = folderPath.trim();
    if (!path) return;
    setFolderBusy(true);
    setFolderError(null);
    try {
      const kind = path.startsWith('/Volumes/') ? 'volume' : 'workspace';
      await skillFoldersApi.add(path, kind);
      setFolderPath('');
      load(); // refresh — skills under the new folder apply to the next run
    } catch (err) {
      setFolderError(
        err instanceof Error ? err.message : 'Could not add skill folder',
      );
    } finally {
      setFolderBusy(false);
    }
  };

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          disabled={disabled}
          className="inline-flex items-center gap-1 rounded-db-md border border-db-gray-lines px-2 py-1 text-[12px] text-db-navy-800 hover:bg-db-oat-light disabled:opacity-50"
        >
          Skills
          {selectedCount > 0 && (
            <span className="ml-1 rounded-full bg-db-navy-800 px-1.5 text-[10px] text-white">
              {selectedCount}
            </span>
          )}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side="top"
          align="start"
          sideOffset={6}
          collisionPadding={8}
          style={{ maxHeight: 'var(--radix-popover-content-available-height)' }}
          className="z-50 w-72 overflow-auto rounded-db-md border border-db-gray-lines bg-white p-3 shadow-lg"
        >
          {loading && <p className="text-[11px] text-db-gray-text">Loading…</p>}

          <p className="mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
            Skills
          </p>
          {!loading && skills.length === 0 && (
            <p className="mb-2 text-[11px] italic text-db-gray-text">
              No skills found.
            </p>
          )}
          {skills.map((name) => (
            <label
              key={`skill-${name}`}
              className="flex items-center gap-2 py-0.5 text-[12px] text-db-navy-800"
            >
              <input
                type="checkbox"
                checked={selectedSkills.includes(name)}
                onChange={() => onChange(toggle(selectedSkills, name))}
              />
              {name}
            </label>
          ))}

          <div className="mt-3 border-t border-db-gray-lines pt-2">
            <p className="mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
              Add skill folder
            </p>
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="/Workspace/Users/you/.skills"
                aria-label="Skill folder path"
                className="min-w-0 flex-1 rounded-db-md border border-db-gray-lines px-2 py-1 text-[11px]"
              />
              <button
                type="button"
                onClick={handleAddFolder}
                disabled={folderBusy || !folderPath.trim()}
                className="rounded-db-md bg-db-navy-800 px-2 py-1 text-[11px] text-white disabled:opacity-50"
              >
                Add
              </button>
            </div>
            {folderError && (
              <p className="mt-1 text-[11px] text-red-600">{folderError}</p>
            )}
            <p className="mt-1 text-[10px] italic text-db-gray-text">
              Folders apply to your next research run.
            </p>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
