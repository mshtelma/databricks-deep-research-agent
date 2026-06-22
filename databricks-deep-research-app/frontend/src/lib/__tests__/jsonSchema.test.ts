/**
 * Tests for defaultConfigForSchema's no-eager-bake behavior.
 *
 * AddToolDialog seeds a new tool's config from this helper. It must seed ONLY
 * fields with an explicit `default` or that are `required` — optional fields
 * without a default stay absent. This keeps "unset" distinguishable from a
 * zero-value so a blank `provider` inherits the workspace default and an absent
 * `resolve_redirects` is filled from app config at run time (not pinned false).
 */

import { describe, expect, it } from 'vitest'

import { defaultConfigForSchema } from '@/lib/jsonSchema'

describe('defaultConfigForSchema', () => {
  it('omits optional fields without an explicit default (web-provider fields)', () => {
    const schema = {
      type: 'object',
      properties: {
        provider: { type: 'string', enum: ['brave', 'jina', 'databricks'] },
        model: { type: 'string' },
        timeout_seconds: { type: 'number' },
        resolve_redirects: { type: 'boolean' },
      },
    }
    expect(defaultConfigForSchema(schema)).toEqual({})
  })

  it('bakes fields that declare an explicit default', () => {
    const schema = {
      type: 'object',
      properties: { max_results: { type: 'integer', default: 10 } },
    }
    expect(defaultConfigForSchema(schema)).toEqual({ max_results: 10 })
  })

  it('seeds required fields with a type zero-value', () => {
    const schema = {
      type: 'object',
      properties: { index_name: { type: 'string' } },
      required: ['index_name'],
    }
    expect(defaultConfigForSchema(schema)).toEqual({ index_name: '' })
  })

  it('does not bake an optional boolean as false', () => {
    const schema = {
      type: 'object',
      properties: { resolve_redirects: { type: 'boolean' } },
    }
    expect(defaultConfigForSchema(schema)).toEqual({})
  })

  it('mixes explicit-default, required, and optional correctly', () => {
    const schema = {
      type: 'object',
      properties: {
        provider: { type: 'string', enum: ['brave', 'databricks'] }, // optional, no default
        max_results: { type: 'integer', default: 5 }, // explicit default
        index_name: { type: 'string' }, // required
      },
      required: ['index_name'],
    }
    expect(defaultConfigForSchema(schema)).toEqual({ max_results: 5, index_name: '' })
  })
})
