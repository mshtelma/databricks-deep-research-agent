/**
 * ImportYamlDialog — create a new agent from a YAML workflow document.
 *
 * Flow (compose existing endpoints; no new persistence backend):
 *   1. source  — file upload / drag-drop / paste → POST /import-yaml
 *                (safe-parse + registry_version + structural validation)
 *   2. (same step) POST /validate on the returned definition — SEMANTIC parity
 *                with create, so a doc that would be rejected at save shows its
 *                errors BEFORE the confirm form (force can't bypass these).
 *   3. preview — workflow summary + editable name/description/visibility,
 *                prefilled from the YAML; run_as reset to "caller" (security).
 *   4. create  — POST /agents-v2; a critic verdict=fail (422) offers
 *                "Import anyway" (force=true, bypasses ONLY the critic).
 */
import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { importYaml, validateWorkflow, type ImportYamlResponse } from '@/api/agentDesigner'
import { YamlImportError, type YamlFieldError } from '@/api/client'
import { parseAgentCriticError, type CritiqueResult } from '@/api/agentsV2'
import { useCreateAgentV2 } from '@/hooks/useAgentsV2'
import type { AST } from '@/types/ast'

const MAX_YAML_BYTES = 256 * 1024
type Visibility = 'private' | 'workspace'

interface ImportYamlDialogProps {
  open: boolean
  onClose: () => void
}

function byteLength(text: string): number {
  return new TextEncoder().encode(text).length
}

/** Coerce a definition.name / description that may be a non-string. */
function asString(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

export function ImportYamlDialog({ open, onClose }: ImportYamlDialogProps) {
  const navigate = useNavigate()
  const createMutation = useCreateAgentV2()

  const [yamlText, setYamlText] = React.useState('')
  const [dragOver, setDragOver] = React.useState(false)
  const [validating, setValidating] = React.useState(false)
  const [errors, setErrors] = React.useState<YamlFieldError[] | null>(null)
  const [genericError, setGenericError] = React.useState<string | null>(null)
  const [result, setResult] = React.useState<ImportYamlResponse | null>(null)
  const [critique, setCritique] = React.useState<CritiqueResult | null>(null)
  // Coverage-only /validate failures don't block the import — the document can
  // still be created as a draft with force=true (mirrors the editor's
  // force-overridable coverage rule). Any other error keeps hard-blocking.
  const [coverageWarnings, setCoverageWarnings] = React.useState<YamlFieldError[] | null>(null)

  // Confirm-form fields
  const [name, setName] = React.useState('')
  const [description, setDescription] = React.useState('')
  const [visibility, setVisibility] = React.useState<Visibility>('private')

  const step: 'source' | 'preview' = result ? 'preview' : 'source'

  // Reset everything whenever the dialog is (re)opened.
  React.useEffect(() => {
    if (open) {
      setYamlText('')
      setDragOver(false)
      setValidating(false)
      setErrors(null)
      setGenericError(null)
      setResult(null)
      setCritique(null)
      setCoverageWarnings(null)
      setName('')
      setDescription('')
      setVisibility('private')
    }
  }, [open])

  if (!open) return null

  const readFile = async (file: File): Promise<void> => {
    setErrors(null)
    setGenericError(null)
    if (file.size > MAX_YAML_BYTES) {
      setGenericError(`File too large (max ${Math.floor(MAX_YAML_BYTES / 1024)} KiB).`)
      return
    }
    try {
      setYamlText(await file.text())
    } catch {
      setGenericError('Could not read the selected file.')
    }
  }

  const handleValidate = async (): Promise<void> => {
    setErrors(null)
    setGenericError(null)
    setCoverageWarnings(null)
    if (byteLength(yamlText) > MAX_YAML_BYTES) {
      setGenericError(`Document too large (max ${Math.floor(MAX_YAML_BYTES / 1024)} KiB).`)
      return
    }
    setValidating(true)
    try {
      const imported = await importYaml(yamlText)
      // Semantic parity with create: surface undeclared-tool / required-config
      // errors here rather than dead-ending at the create call.
      const semantic = await validateWorkflow(imported.definition)
      if (!semantic.valid) {
        const items = semantic.errors.map((e) => ({
          path: e.path,
          kind: e.kind,
          message: e.message,
        }))
        // Coverage-only failures are force-passable (the create endpoint's
        // coverage gate honors force=true) — now that imported documents keep
        // their designer metadata, a work-in-progress export must stay
        // importable as a draft rather than dead-ending here.
        if (items.every((e) => e.kind === 'coverage')) {
          setCoverageWarnings(items)
        } else {
          setErrors(items)
          return
        }
      }
      setName(asString(imported.definition.name).slice(0, 255))
      setDescription(asString(imported.definition.description))
      setResult(imported)
    } catch (err) {
      if (err instanceof YamlImportError) {
        setErrors(err.errors)
      } else {
        setGenericError(err instanceof Error ? err.message : 'Failed to validate YAML.')
      }
    } finally {
      setValidating(false)
    }
  }

  const handleCreate = (force: boolean): void => {
    if (!result) return
    const trimmed = name.trim()
    if (!trimmed) {
      setGenericError('Name is required.')
      return
    }
    setGenericError(null)
    setCritique(null)
    // Sync AST name to the entered name; reset run_as so an imported document
    // can never smuggle in a service-principal the importer didn't choose.
    const definition: AST = {
      ...(result.definition as Record<string, unknown>),
      name: trimmed,
      run_as: 'caller',
    } as AST
    createMutation.mutate(
      {
        name: trimmed,
        description: description.trim() || null,
        visibility,
        definition,
        force,
      },
      {
        onSuccess: ({ agent }) => {
          onClose()
          void navigate(`/designer/${agent.id}`)
        },
        onError: (err) => {
          const critic = parseAgentCriticError(err)
          if (critic) {
            setCritique(critic.critique)
          } else {
            setGenericError(err instanceof Error ? err.message : 'Failed to create agent.')
          }
        },
      },
    )
  }

  const summary = result?.workflow_summary
  const creating = createMutation.isPending

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-db-navy-900/30 p-4 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
      aria-label="Import agent from YAML"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-db-lg border border-db-gray-lines bg-white shadow-db-xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-db-gray-lines px-5 py-3.5">
          <div className="text-[15px] font-medium text-db-navy-800">
            {step === 'source' ? 'Import agent from YAML' : 'Review imported agent'}
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            className="rounded p-1 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-auto px-5 py-4">
          {step === 'source' ? (
            <>
              <label
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragOver(true)
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDragOver(false)
                  const file = e.dataTransfer.files?.[0]
                  if (file) void readFile(file)
                }}
                className={`flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-db-md border border-dashed px-4 py-6 text-center transition-colors ${
                  dragOver
                    ? 'border-db-navy-400 bg-db-oat-medium'
                    : 'border-db-gray-lines bg-db-oat-light hover:border-db-navy-300'
                }`}
              >
                <input
                  type="file"
                  accept=".yaml,.yml,text/yaml,application/x-yaml"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) void readFile(file)
                    e.target.value = ''
                  }}
                />
                <span className="text-[13px] font-medium text-db-navy-800">
                  Drop a <span className="font-db-mono">.yaml</span> file or click to browse
                </span>
                <span className="text-[12px] text-db-gray-text">or paste the document below</span>
              </label>

              <textarea
                value={yamlText}
                onChange={(e) => setYamlText(e.target.value)}
                placeholder="registry_version: '1.0.0'&#10;name: My Agent&#10;root: …"
                spellCheck={false}
                className="mt-3 h-44 w-full resize-y rounded-db-md border border-db-gray-lines bg-white p-2.5 font-db-mono text-[12px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus"
              />

              {errors && errors.length > 0 && (
                <div
                  role="alert"
                  className="mt-3 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] text-db-lava-700"
                >
                  <div className="mb-1 font-medium">This document can’t be imported:</div>
                  <ul className="space-y-1">
                    {errors.map((e, i) => (
                      <li key={i} className="leading-relaxed">
                        <span className="font-db-mono font-semibold">{e.kind}</span>
                        {e.path ? <span className="font-db-mono"> · {e.path}</span> : null}
                        {' — '}
                        {e.message}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {genericError && (
                <div
                  role="alert"
                  className="mt-3 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] text-db-lava-700"
                >
                  {genericError}
                </div>
              )}
            </>
          ) : (
            <>
              {summary && (
                <div className="mb-4 flex flex-wrap gap-2 text-[11px] text-db-gray-text">
                  <span className="rounded-db-pill bg-db-oat-medium px-2 py-0.5 font-db-mono">
                    {summary.node_count} blocks
                  </span>
                  <span className="rounded-db-pill bg-db-oat-medium px-2 py-0.5 font-db-mono">
                    {summary.tool_count} tools
                  </span>
                  <span className="rounded-db-pill bg-db-oat-medium px-2 py-0.5 font-db-mono">
                    {summary.source_count} sources
                  </span>
                </div>
              )}

              {result && (result.warnings?.length ?? 0) > 0 && (
                <div
                  role="alert"
                  className="mb-4 rounded-db-md border border-db-yellow-700 bg-db-yellow-300 px-3 py-2 text-[12px] text-db-yellow-800"
                >
                  <div className="font-medium">
                    Imported with warnings — some designer metadata was adjusted:
                  </div>
                  <ul className="mt-1 space-y-1">
                    {(result.warnings ?? []).map((w, i) => (
                      <li key={i} className="leading-relaxed">
                        <span className="font-db-mono font-semibold">{w.key}</span>
                        <span className="font-db-mono"> · {w.action}</span>
                        {' — '}
                        {w.message}
                        {w.detail && w.detail.length > 0 ? ` (${w.detail.join(', ')})` : null}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {coverageWarnings && coverageWarnings.length > 0 && (
                <div
                  role="alert"
                  className="mb-4 rounded-db-md border border-db-yellow-700 bg-db-yellow-300 px-3 py-2 text-[12px] text-db-yellow-800"
                >
                  <div className="font-medium">
                    This document doesn&rsquo;t cover every requested topic yet.
                  </div>
                  <ul className="mt-1 space-y-1">
                    {coverageWarnings.map((e, i) => (
                      <li key={i} className="leading-relaxed">
                        {e.message}
                      </li>
                    ))}
                  </ul>
                  <p className="mt-1">You can import it as a draft and finish it in the editor.</p>
                </div>
              )}

              <label className="mb-1 block text-[12px] font-medium text-db-navy-800">Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={255}
                className="mb-3 w-full rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[13px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus"
              />

              <label className="mb-1 block text-[12px] font-medium text-db-navy-800">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="mb-3 h-20 w-full resize-y rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[13px] text-db-navy-800 outline-none focus:border-db-navy-400 focus:shadow-db-focus"
              />

              <span className="mb-1 block text-[12px] font-medium text-db-navy-800">Visibility</span>
              <div className="flex gap-4">
                {(['private', 'workspace'] as const).map((v) => (
                  <label key={v} className="flex items-center gap-1.5 text-[13px] text-db-navy-800">
                    <input
                      type="radio"
                      name="visibility"
                      checked={visibility === v}
                      onChange={() => setVisibility(v)}
                    />
                    <span className="capitalize">{v}</span>
                  </label>
                ))}
              </div>

              {critique && (
                <div
                  role="alert"
                  className="mt-4 rounded-db-md border border-db-yellow-700 bg-db-yellow-300 px-3 py-2 text-[12px] text-db-yellow-800"
                >
                  <div className="font-medium">The workflow critic flagged this agent.</div>
                  {critique.summary && <p className="mt-1 leading-relaxed">{critique.summary}</p>}
                  <p className="mt-1">You can import it anyway and refine it in the editor.</p>
                </div>
              )}
              {genericError && (
                <div
                  role="alert"
                  className="mt-4 rounded-db-md border border-db-lava-300 bg-db-lava-100 px-3 py-2 text-[12px] text-db-lava-700"
                >
                  {genericError}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 border-t border-db-gray-lines px-5 py-3.5">
          {step === 'source' ? (
            <>
              <button
                type="button"
                onClick={onClose}
                className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleValidate()}
                disabled={!yamlText.trim() || validating}
                className="rounded-db-md bg-db-navy-800 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-navy-900 disabled:cursor-not-allowed disabled:opacity-55"
              >
                {validating ? 'Validating…' : 'Validate'}
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setResult(null)}
                disabled={creating}
                className="rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:bg-db-oat-medium disabled:opacity-55"
              >
                Back
              </button>
              {critique ? (
                <button
                  type="button"
                  onClick={() => handleCreate(true)}
                  disabled={creating || !name.trim()}
                  className="rounded-db-md bg-db-lava-600 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-lava-700 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {creating ? 'Importing…' : 'Import anyway'}
                </button>
              ) : coverageWarnings && coverageWarnings.length > 0 ? (
                <button
                  type="button"
                  onClick={() => handleCreate(true)}
                  disabled={creating || !name.trim()}
                  className="rounded-db-md bg-db-navy-800 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-navy-900 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {creating ? 'Importing…' : 'Import as draft'}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => handleCreate(false)}
                  disabled={creating || !name.trim()}
                  className="rounded-db-md bg-db-navy-800 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-navy-900 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {creating ? 'Creating…' : 'Create agent'}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  )
}

export default ImportYamlDialog
