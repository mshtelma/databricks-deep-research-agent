/**
 * SkillCompilePicker — pick skill(s) to COMPILE into the workflow the Designer
 * chat is about to draft (Skill -> Workflow, P5).
 *
 * The selection is threaded to the chat request as `skill_names` (separate from
 * resource `assets`); the backend summarizes each skill (OBO, scanned) into a
 * bounded brief that drives a deterministic blueprint + per-node prompts. The
 * dropdown opens upward since the composer sits at the bottom of the panel.
 */

import * as React from 'react';

import { listDesignerResources } from '@/api/agentDesigner';

export interface SkillCompilePickerProps {
  selected: string[];
  onChange: (next: string[]) => void;
  disabled?: boolean;
}

export function SkillCompilePicker({
  selected,
  onChange,
  disabled = false,
}: SkillCompilePickerProps): React.ReactElement {
  const [open, setOpen] = React.useState(false);
  const [skills, setSkills] = React.useState<string[]>([]);
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    if (!open || skills.length > 0) return;
    setLoading(true);
    listDesignerResources(['skill'])
      .then((res) =>
        setSkills(res.resources.filter((r) => r.kind === 'skill').map((r) => r.name)),
      )
      .catch(() => {
        /* discovery is best-effort; leave the list empty */
      })
      .finally(() => setLoading(false));
  }, [open, skills.length]);

  const toggle = (name: string): void =>
    onChange(
      selected.includes(name)
        ? selected.filter((s) => s !== name)
        : [...selected, name],
    );

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded-db-md border border-db-gray-lines px-2 py-1 text-[11px] text-db-navy-800 hover:bg-db-oat-medium disabled:opacity-50"
        aria-haspopup="true"
        aria-expanded={open}
      >
        Compile skill{selected.length > 0 ? ` (${selected.length})` : ''}
      </button>
      {open && (
        <div
          className="absolute bottom-full z-20 mb-1 max-h-64 w-64 overflow-auto rounded-db-md border border-db-gray-lines bg-white p-2 shadow-lg"
          role="menu"
        >
          <p className="mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
            Skills to compile into the workflow
          </p>
          {loading && <p className="text-[11px] text-db-gray-text">Loading…</p>}
          {!loading && skills.length === 0 && (
            <p className="text-[11px] italic text-db-gray-text">No skills found.</p>
          )}
          {skills.map((name) => (
            <label
              key={name}
              className="flex items-center gap-2 py-0.5 text-[12px] text-db-navy-800"
            >
              <input
                type="checkbox"
                checked={selected.includes(name)}
                onChange={() => toggle(name)}
              />
              {name}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
