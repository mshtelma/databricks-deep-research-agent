/**
 * Tests for the canonical AST hash helper (W14).
 *
 * The whole point of the helper is determinism across insertion-order
 * differences: two ASTs with the same content but different key order
 * MUST hash to the same value. Otherwise the chat-mutation race detector
 * fires spurious "base diverged" prompts.
 */

import { describe, expect, it } from 'vitest'

import { astHash, canonicalStringifyForTest } from '@/lib/astHash'

describe('canonicalStringify', () => {
  it('produces identical output for objects with reordered keys', () => {
    const a = { a: 1, b: 2, c: { x: 10, y: 20 } }
    const b = { c: { y: 20, x: 10 }, b: 2, a: 1 }
    expect(canonicalStringifyForTest(a)).toBe(canonicalStringifyForTest(b))
  })

  it('preserves array order (arrays are NOT sorted)', () => {
    const a = [1, 2, 3]
    const b = [3, 2, 1]
    expect(canonicalStringifyForTest(a)).not.toBe(canonicalStringifyForTest(b))
  })

  it('handles nested arrays and objects', () => {
    const ast1 = {
      root: {
        type: 'sequence',
        children: [
          { id: 'a', type: 'agent', config: { endpoint: 'gpt-5' } },
        ],
      },
      tools: [],
    }
    const ast2 = {
      tools: [],
      root: {
        children: [
          { config: { endpoint: 'gpt-5' }, type: 'agent', id: 'a' },
        ],
        type: 'sequence',
      },
    }
    expect(canonicalStringifyForTest(ast1)).toBe(canonicalStringifyForTest(ast2))
  })

  it('handles primitives, null, and missing values', () => {
    expect(canonicalStringifyForTest(null)).toBe('null')
    expect(canonicalStringifyForTest(true)).toBe('true')
    expect(canonicalStringifyForTest(42)).toBe('42')
    expect(canonicalStringifyForTest('hello')).toBe('"hello"')
  })
})

describe('astHash', () => {
  it('returns a 40-char hex SHA-1', async () => {
    const h = await astHash({ root: { type: 'agent', id: 'r' } })
    expect(h).toMatch(/^[0-9a-f]{40}$/)
  })

  it('returns identical hashes for key-reordered equivalent ASTs', async () => {
    const a = await astHash({ a: 1, b: [{ x: 1, y: 2 }] })
    const b = await astHash({ b: [{ y: 2, x: 1 }], a: 1 })
    expect(a).toBe(b)
  })

  it('returns different hashes for differing content', async () => {
    const a = await astHash({ root: { type: 'agent', id: 'r1' } })
    const b = await astHash({ root: { type: 'agent', id: 'r2' } })
    expect(a).not.toBe(b)
  })
})
