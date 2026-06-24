import { describe, it, expect } from 'vitest'
import runStatusContract from '../../../../contracts/run_status_contract.json'
import {
  RUN_STATUS_LABELS,
  RUN_STATUSES,
  formatStatusLabel,
  formatActivityLabel,
} from '../activityLabels'

/**
 * Frontend half of the backend<->frontend status parity.
 *
 * The single shared fixture (contracts/run_status_contract.json) pins the
 * backend RunStatus enum (a framework pytest) AND this TS label map. If either
 * side drifts, exactly one of the two CI checks fails. This vitest is the
 * frontend gate.
 */
describe('run status contract parity', () => {
  it('TS label map keys == contract statuses', () => {
    expect(new Set(Object.keys(RUN_STATUS_LABELS))).toEqual(
      new Set(runStatusContract.statuses),
    )
  })

  it('exported RUN_STATUSES == contract statuses', () => {
    expect(new Set(RUN_STATUSES)).toEqual(new Set(runStatusContract.statuses))
  })

  it('every contract status has a non-empty label', () => {
    for (const status of runStatusContract.statuses) {
      const label = RUN_STATUS_LABELS[status]
      expect(typeof label).toBe('string')
      expect((label ?? '').length).toBeGreaterThan(0)
    }
  })
})

describe('formatStatusLabel', () => {
  it('renders a known status from the map', () => {
    expect(formatStatusLabel('completed')).toBe(RUN_STATUS_LABELS['completed'])
    expect(formatStatusLabel('safety_termination')).toBe(
      RUN_STATUS_LABELS['safety_termination'],
    )
  })

  it('falls back safely for an unknown status (no throw)', () => {
    expect(formatStatusLabel('not_a_real_status')).toBe('not_a_real_status')
    expect(formatStatusLabel(null)).toBe('Unknown')
    expect(formatStatusLabel(undefined)).toBe('Unknown')
    expect(formatStatusLabel('')).toBe('Unknown')
  })
})

describe('formatActivityLabel status-awareness', () => {
  it('renders from the status field when the event carries a known status', () => {
    const event = {
      eventType: 'node_completed',
      status: 'budget_exceeded',
    } as unknown as Parameters<typeof formatActivityLabel>[0]
    expect(formatActivityLabel(event)).toBe(RUN_STATUS_LABELS['budget_exceeded'])
  })

  it('ignores an unknown status and keeps existing eventType behavior', () => {
    const event = {
      eventType: 'research_started',
      status: 'bogus_status',
    } as unknown as Parameters<typeof formatActivityLabel>[0]
    expect(formatActivityLabel(event)).toBe('Research started')
  })

  it('keeps existing behavior for events without a status field', () => {
    const event = {
      eventType: 'research_started',
    } as unknown as Parameters<typeof formatActivityLabel>[0]
    expect(formatActivityLabel(event)).toBe('Research started')
  })
})
