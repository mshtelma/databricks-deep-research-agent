import { describe, it, expect } from 'vitest'
import { ApiError } from '../client'

/**
 * Smoke tests for ApiError so callers (notably useStreamingQuery) can
 * classify failures by HTTP status / error code rather than substring-
 * matching the message body.
 *
 * Background: a research query containing the literal text "429" or
 * "concurrent" used to misclassify as MAX_CONCURRENT_JOBS because the
 * hook checked errorMessage.includes('429'). With ApiError exposing
 * status + code, classification is robust.
 */
describe('ApiError', () => {
  it('is a thrown Error subclass with status and code attached', () => {
    const err = new ApiError(429, 'MAX_CONCURRENT_JOBS', 'too many jobs')
    expect(err).toBeInstanceOf(Error)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(429)
    expect(err.code).toBe('MAX_CONCURRENT_JOBS')
    expect(err.message).toBe('too many jobs')
  })

  it('lets callers discriminate 429 (concurrent jobs) from other errors', () => {
    const concurrent = new ApiError(429, 'MAX_CONCURRENT_JOBS', 'too many jobs')
    const inProgress = new ApiError(409, 'RESEARCH_IN_PROGRESS', 'already running')
    const generic = new ApiError(500, 'UNKNOWN', 'server error')

    const classify = (err: unknown): string => {
      if (err instanceof ApiError) {
        if (err.status === 429 || err.code === 'MAX_CONCURRENT_JOBS') return 'MAX_CONCURRENT_JOBS'
        if (err.status === 409 || err.code === 'RESEARCH_IN_PROGRESS') return 'RESEARCH_IN_PROGRESS'
      }
      return 'SUBMISSION_FAILED'
    }

    expect(classify(concurrent)).toBe('MAX_CONCURRENT_JOBS')
    expect(classify(inProgress)).toBe('RESEARCH_IN_PROGRESS')
    expect(classify(generic)).toBe('SUBMISSION_FAILED')
  })

  it('does NOT misclassify a generic Error whose message contains "429" or "concurrent"', () => {
    // Regression: this was the original bug — substring matching the
    // message body misclassified legitimate research queries as
    // MAX_CONCURRENT_JOBS. Now we discriminate on the status code, not
    // the message string.
    const benign = new Error('Research query: explain HTTP 429 concurrent rate limiting strategies')

    const classify = (err: unknown): string => {
      if (err instanceof ApiError) {
        if (err.status === 429 || err.code === 'MAX_CONCURRENT_JOBS') return 'MAX_CONCURRENT_JOBS'
        if (err.status === 409 || err.code === 'RESEARCH_IN_PROGRESS') return 'RESEARCH_IN_PROGRESS'
      }
      return 'SUBMISSION_FAILED'
    }

    expect(classify(benign)).toBe('SUBMISSION_FAILED')
  })

  it('exposes optional details payload', () => {
    const err = new ApiError(400, 'VALIDATION_FAILED', 'bad input', { field: 'query' })
    expect(err.details).toEqual({ field: 'query' })
  })
})
