/**
 * MermaidPreview — renders a Mermaid flowchart for an agent workflow.
 *
 * Server-mode flag
 * ----------------
 * When ``VITE_AGENT_DESIGNER_MERMAID_SERVER === '1'``, the component fetches
 * the Mermaid source from ``GET /api/v1/agents-v2/{id}/mermaid`` and renders
 * the returned text as an SVG via the ``mermaid`` npm package.
 *
 * Local-fallback mode (default)
 * ------------------------------
 * When the flag is absent or any value other than ``'1'``, the component
 * renders a placeholder without making any network requests.  Full local
 * Mermaid generation from the AST is deferred to a future story.
 *
 * Mermaid dependency
 * ------------------
 * The ``mermaid`` npm package is imported dynamically (``import()``) so that
 * it is code-split into its own chunk and does not inflate the main bundle.
 * Tree-shaking ensures unused Mermaid internals are dropped by Vite.
 */

import * as React from 'react'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE_URL = import.meta.env.VITE_API_URL ?? '/api/v1'
const SERVER_MODE = import.meta.env.VITE_AGENT_DESIGNER_MERMAID_SERVER === '1'

// Stable unique id counter for Mermaid diagram containers — avoids collisions
// when multiple <MermaidPreview> instances mount simultaneously.
let _idCounter = 0
function _nextId(): string {
  _idCounter += 1
  return `mermaid-preview-${_idCounter}`
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface MermaidPreviewProps {
  /** UUID of the agent to render. Required in server mode; ignored in local mode. */
  agentId?: string
  /** Optional CSS class name applied to the container <div>. */
  className?: string
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MermaidPreview({ agentId, className }: MermaidPreviewProps): React.ReactElement {
  const [source, setSource] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState<boolean>(false)

  // Stable container id so Mermaid can target the correct DOM node.
  const containerId = React.useRef<string>(_nextId()).current

  // -------------------------------------------------------------------------
  // Fetch Mermaid source from server (server mode only)
  // -------------------------------------------------------------------------

  React.useEffect(() => {
    if (!SERVER_MODE || !agentId) {
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetch(`${API_BASE_URL}/agents-v2/${agentId}/mermaid`)
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}: failed to fetch Mermaid source`)
        }
        return resp.text()
      })
      .then((text) => {
        if (!cancelled) {
          setSource(text)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [agentId])

  // -------------------------------------------------------------------------
  // Render Mermaid SVG whenever `source` changes (server mode)
  // -------------------------------------------------------------------------

  React.useEffect(() => {
    if (!source) return

    let cancelled = false

    // Dynamic import keeps Mermaid out of the critical chunk.
    import('mermaid')
      .then((mod) => {
        if (cancelled) return
        const mermaid = mod.default
        mermaid.initialize({ startOnLoad: false, theme: 'default' })
        return mermaid.render(containerId, source)
      })
      .then((result) => {
        if (cancelled || !result) return
        const el = document.getElementById(containerId)
        if (el) {
          el.innerHTML = result.svg
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Mermaid render failed')
        }
      })

    return () => {
      cancelled = true
    }
  }, [source, containerId])

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (!SERVER_MODE) {
    // Local-fallback placeholder — full local generation is out of V1 scope.
    return (
      <div
        data-testid="mermaid-preview-placeholder"
        className={`flex items-center justify-center rounded border border-dashed border-slate-300 p-6 text-sm text-slate-400 ${className ?? ''}`}
      >
        Mermaid preview (local generation not yet available)
      </div>
    )
  }

  if (loading) {
    return (
      <div
        data-testid="mermaid-preview-loading"
        className={`flex items-center justify-center p-6 text-sm text-slate-500 ${className ?? ''}`}
      >
        Loading diagram…
      </div>
    )
  }

  if (error) {
    return (
      <div
        role="alert"
        data-testid="mermaid-preview-error"
        className={`rounded border border-red-200 bg-red-50 p-4 text-sm text-red-700 ${className ?? ''}`}
      >
        {error}
      </div>
    )
  }

  return (
    <div
      id={containerId}
      data-testid="mermaid-preview-diagram"
      className={`overflow-auto ${className ?? ''}`}
    />
  )
}
