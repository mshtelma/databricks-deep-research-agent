/**
 * Tests for statusStyles.ts — the single source of truth for the
 * status × mode action-label matrix. The behavior in this file backs the
 * Undeploy UI contract on every consumer (StatusPanel, DeploymentRow,
 * UndeployConfirmDialog). If you change the matrix here, expect to
 * update the matching table in plan imperative-wishing-lynx.md.
 */

import { describe, expect, it } from 'vitest'

import type {
  DeploymentMode,
  DeploymentResponse,
  DeploymentStatus,
} from '@/types/deployment'

import {
  getAction,
  getEffectiveStatusLabel,
  getImpactText,
  getResourceSummary,
  readResourceId,
} from '../statusStyles'

function makeRow(overrides: Partial<DeploymentResponse>): DeploymentResponse {
  return {
    id: 'dep-1',
    agent_id: 'agent-1',
    revision_id: 'rev-1',
    mode: 'in_app',
    status: 'pending',
    config: {},
    endpoint_name: null,
    model_name: null,
    external_resource_ids: null,
    error_message: null,
    cleanup_attempts: 0,
    cancel_requested: false,
    deployed_by: 'user-1',
    created_at: '2026-05-25T10:00:00Z',
    updated_at: '2026-05-25T10:00:00Z',
    deactivated_at: null,
    ...overrides,
  }
}

const MODES: DeploymentMode[] = ['in_app', 'shell_app', 'mlflow_agent', 'batch']

describe('getAction matrix', () => {
  // pending/deploying — every mode maps to Cancel.
  for (const status of ['pending', 'deploying'] as DeploymentStatus[]) {
    for (const mode of MODES) {
      it(`status=${status} mode=${mode} → Cancel`, () => {
        const a = getAction(makeRow({ status, mode }))
        expect(a).not.toBeNull()
        expect(a!.kind).toBe('cancel')
        expect(a!.label).toBe('Cancel')
      })
    }
  }

  // active — in_app maps to Unregister; others map to Undeploy.
  it('status=active mode=in_app → Unregister', () => {
    const a = getAction(makeRow({ status: 'active', mode: 'in_app' }))
    expect(a!.kind).toBe('unregister')
    expect(a!.label).toBe('Unregister')
  })

  for (const mode of ['shell_app', 'mlflow_agent', 'batch'] as DeploymentMode[]) {
    it(`status=active mode=${mode} → Undeploy`, () => {
      const a = getAction(makeRow({ status: 'active', mode }))
      expect(a!.kind).toBe('undeploy')
      expect(a!.label).toBe('Undeploy')
    })
  }

  // failed — every mode maps to Clean up.
  for (const mode of MODES) {
    it(`status=failed mode=${mode} → Clean up`, () => {
      const a = getAction(makeRow({ status: 'failed', mode }))
      expect(a!.kind).toBe('cleanup')
      expect(a!.label).toBe('Clean up')
    })
  }

  // cleanup_failed — every mode maps to Retry cleanup.
  for (const mode of MODES) {
    it(`status=cleanup_failed mode=${mode} → Retry cleanup`, () => {
      const a = getAction(makeRow({ status: 'cleanup_failed', mode }))
      expect(a!.kind).toBe('retry-cleanup')
      expect(a!.label).toBe('Retry cleanup')
    })
  }

  // deactivated — null (no action button).
  for (const mode of MODES) {
    it(`status=deactivated mode=${mode} → null`, () => {
      const a = getAction(makeRow({ status: 'deactivated', mode }))
      expect(a).toBeNull()
    })
  }
})

describe('getEffectiveStatusLabel', () => {
  it('returns Cancelling… for pending + cancel_requested', () => {
    const label = getEffectiveStatusLabel(
      makeRow({ status: 'pending', cancel_requested: true }),
    )
    expect(label).toBe('Cancelling…')
  })

  it('returns Cancelling… for deploying + cancel_requested', () => {
    const label = getEffectiveStatusLabel(
      makeRow({ status: 'deploying', cancel_requested: true }),
    )
    expect(label).toBe('Cancelling…')
  })

  it('returns plain status label when cancel_requested is false', () => {
    const label = getEffectiveStatusLabel(
      makeRow({ status: 'deploying', cancel_requested: false }),
    )
    expect(label).toBe('Deploying')
  })

  it('does not override active-status label even if cancel_requested somehow set', () => {
    // Defensive: cancel_requested only meaningful in pending/deploying.
    const label = getEffectiveStatusLabel(
      makeRow({ status: 'active', cancel_requested: true }),
    )
    expect(label).toBe('Active')
  })
})

describe('readResourceId fallback chain', () => {
  it('reads from external_resource_ids first', () => {
    const row = makeRow({
      external_resource_ids: { app_name: 'from-external' },
      config: { app_name: 'from-config' },
    })
    expect(readResourceId(row, 'app_name')).toBe('from-external')
  })

  it('falls back to config when external is absent', () => {
    const row = makeRow({
      external_resource_ids: null,
      config: { app_name: 'from-config' },
    })
    expect(readResourceId(row, 'app_name')).toBe('from-config')
  })

  it('returns null when both empty', () => {
    const row = makeRow({ external_resource_ids: null, config: {} })
    expect(readResourceId(row, 'app_name')).toBeNull()
  })

  it('returns null for non-string values', () => {
    const row = makeRow({ external_resource_ids: { app_name: 123 } })
    expect(readResourceId(row, 'app_name')).toBeNull()
  })

  it('returns null for empty-string values', () => {
    const row = makeRow({ external_resource_ids: { app_name: '' } })
    expect(readResourceId(row, 'app_name')).toBeNull()
  })
})

describe('getResourceSummary', () => {
  it('shell_app shows app_name', () => {
    const row = makeRow({
      mode: 'shell_app',
      external_resource_ids: { app_name: 'dr-shell-foo' },
    })
    expect(getResourceSummary(row)).toBe('dr-shell-foo')
  })

  it('shell_app falls back when app_name unset', () => {
    const row = makeRow({ mode: 'shell_app' })
    expect(getResourceSummary(row)).toBe('(app name pending)')
  })

  it('mlflow_agent prefers endpoint_name column', () => {
    const row = makeRow({
      mode: 'mlflow_agent',
      endpoint_name: 'dr-agent-foo',
      external_resource_ids: { endpoint_name: 'should-be-ignored' },
    })
    expect(getResourceSummary(row)).toBe('dr-agent-foo')
  })

  it('in_app shows generic Chat picker label', () => {
    const row = makeRow({ mode: 'in_app' })
    expect(getResourceSummary(row)).toBe('Chat picker')
  })

  it('batch labels itself a stub', () => {
    const row = makeRow({ mode: 'batch' })
    expect(getResourceSummary(row)).toBe('Lakeflow pipeline (stub)')
  })
})

describe('getImpactText', () => {
  it('pending/deploying shows in-flight phrasing for any mode', () => {
    for (const mode of MODES) {
      const text = getImpactText(makeRow({ status: 'deploying', mode }))
      expect(text).toMatch(/Cancel this in-flight deployment/)
    }
  })

  it('shell_app + active mentions app deletion and preservation', () => {
    const text = getImpactText(
      makeRow({
        status: 'active',
        mode: 'shell_app',
        external_resource_ids: { app_name: 'dr-shell-foo' },
      }),
    )
    expect(text).toMatch(/dr-shell-foo/)
    expect(text).toMatch(/deleted/)
    expect(text).toMatch(/agent.*preserved/i)
  })

  it('mlflow_agent + active mentions endpoint deletion', () => {
    const text = getImpactText(
      makeRow({
        status: 'active',
        mode: 'mlflow_agent',
        endpoint_name: 'dr-agent-foo',
        model_name: 'catalog.schema.model/1',
      }),
    )
    expect(text).toMatch(/dr-agent-foo/)
    expect(text).toMatch(/archived/)
  })

  it('in_app + active mentions chat picker', () => {
    const text = getImpactText(makeRow({ status: 'active', mode: 'in_app' }))
    expect(text).toMatch(/chat picker/i)
    expect(text).toMatch(/agent.*preserved/i)
  })

  it('batch + active honestly calls out the Phase-3 stub', () => {
    const text = getImpactText(makeRow({ status: 'active', mode: 'batch' }))
    expect(text).toMatch(/Phase 3/)
    expect(text).toMatch(/manually/)
  })
})
