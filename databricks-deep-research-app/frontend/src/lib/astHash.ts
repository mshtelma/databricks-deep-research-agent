/**
 * Canonical AST hashing for the chat-driven mutation race detector (W14).
 *
 * `JSON.stringify` in JavaScript preserves insertion order — so two ASTs
 * with identical content but different key insertion order serialize to
 * different strings and would hash differently, producing spurious
 * "base diverged" prompts in the editor. Codex flagged this in the
 * Phase-4 plan review.
 *
 * `canonicalStringify` produces a deterministic UTF-8 string with keys
 * sorted recursively in every object. We then SHA-1 it to get a compact
 * fingerprint. SHA-1 is fine here: this is a divergence detector, not a
 * security boundary.
 */

function canonicalStringify(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'number') {
    return Number.isFinite(value) ? JSON.stringify(value) : 'null'
  }
  if (typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalStringify).join(',')}]`
  }
  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>
    const keys = Object.keys(obj).sort()
    const parts = keys.map(
      (k) => `${JSON.stringify(k)}:${canonicalStringify(obj[k])}`,
    )
    return `{${parts.join(',')}}`
  }
  // undefined / functions / bigints / symbols — JSON.stringify omits them
  // entirely in objects, so we mirror that and skip in the canonical form
  // by returning the empty string at the top level. Callers should never
  // hash these directly.
  return 'null'
}

/**
 * SHA-1 of the canonical serialization of `ast`.
 *
 * Async because Web Crypto's `subtle.digest` is async. In jsdom test
 * environments where `crypto.subtle` is missing, callers should use the
 * sync fallback (`canonicalStringifyForTest`) and a deterministic
 * stand-in hash.
 */
export async function astHash(ast: unknown): Promise<string> {
  const canon = canonicalStringify(ast)
  const bytes = new TextEncoder().encode(canon)
  const digest = await crypto.subtle.digest('SHA-1', bytes)
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

/** Exposed for unit tests that want to assert key-ordering insensitivity
 * without round-tripping through SubtleCrypto. */
export const canonicalStringifyForTest = canonicalStringify
