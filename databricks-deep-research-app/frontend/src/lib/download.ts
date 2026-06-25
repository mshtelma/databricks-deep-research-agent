/**
 * Shared browser download / clipboard helpers.
 *
 * Extracted so the YAML export menu, the chat message export menu, and any
 * future "download this text" feature share one implementation instead of
 * re-deriving the Blob + <a download> dance each time.
 */

/** Trigger a browser download of `content` as a file named `filename`. */
export function downloadTextFile(
  content: string,
  filename: string,
  mime = 'text/yaml',
): void {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Copy `text` to the clipboard.
 *
 * Returns `false` (never throws) when the Clipboard API is unavailable — e.g.
 * a non-secure (HTTP) context or a browser that withholds permission — so the
 * caller can fall back to "use Download" messaging.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // fall through to the unavailable result
  }
  return false
}

/**
 * Build a safe download filename from a human name.
 *
 * `"My Agent!"` + `"yaml"` → `"my-agent.yaml"`. Falls back to `"agent"` when
 * the name has no usable characters.
 */
export function slugifyFilename(name: string, ext: string): string {
  const base = (name || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
  return `${base || 'agent'}.${ext}`
}
