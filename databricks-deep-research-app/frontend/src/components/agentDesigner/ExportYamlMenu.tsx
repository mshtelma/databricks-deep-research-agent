/**
 * ExportYamlMenu — a small dropdown that exports a workflow as YAML.
 *
 * Offers "Download .yaml" and "Copy to clipboard", both driven by a lazily
 * evaluated `getYaml` (so the export endpoint is hit only on click). Used in
 * the agent editor top bar; the agents-list per-agent menu reuses the same
 * `useYamlExport` hook directly as inline menu items.
 */
import * as React from 'react'
import { ChevronDown, Copy, Download } from 'lucide-react'
import { useYamlExport } from '@/hooks/useYamlExport'

interface ExportYamlMenuProps {
  /** Produce the YAML text to export (called on each action). */
  getYaml: () => Promise<string>
  /** Download filename, e.g. `my-agent.yaml`. */
  filename: string
  /** Disable the trigger (e.g. no canvas loaded yet). */
  disabled?: boolean
}

export function ExportYamlMenu({ getYaml, filename, disabled }: ExportYamlMenuProps) {
  const [open, setOpen] = React.useState(false)
  const ref = React.useRef<HTMLDivElement>(null)
  const { download, copy, busy } = useYamlExport(getYaml, filename)

  React.useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  const choose = (action: () => Promise<void>) => {
    setOpen(false)
    void action()
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled || busy}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Export YAML"
        className="inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium disabled:cursor-not-allowed disabled:opacity-55"
      >
        <Download size={13} /> {busy ? 'Exporting…' : 'Export YAML'}
        <ChevronDown size={12} />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-20 mt-1 w-48 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-db-md"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => choose(download)}
            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
          >
            <Download size={13} /> Download .yaml
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => choose(copy)}
            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
          >
            <Copy size={13} /> Copy to clipboard
          </button>
        </div>
      )}
    </div>
  )
}

export default ExportYamlMenu
