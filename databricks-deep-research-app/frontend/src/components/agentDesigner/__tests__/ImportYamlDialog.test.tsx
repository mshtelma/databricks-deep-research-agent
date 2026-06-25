import '@testing-library/jest-dom/vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

// --- Hoisted mocks -----------------------------------------------------------
const { importYaml, validateWorkflow, mutate } = vi.hoisted(() => ({
  importYaml: vi.fn(),
  validateWorkflow: vi.fn(),
  mutate: vi.fn(),
}))

vi.mock('@/api/agentDesigner', () => ({ importYaml, validateWorkflow }))
vi.mock('@/hooks/useAgentsV2', () => ({
  useCreateAgentV2: () => ({ mutate, isPending: false }),
}))
vi.mock('@/api/agentsV2', () => ({
  // Recognise the critic error by its `critique` marker (real impl is equivalent).
  parseAgentCriticError: (e: unknown) =>
    e && typeof e === 'object' && 'critique' in (e as object) ? e : null,
}))

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return { ...actual, useNavigate: () => mockNavigate }
})

import { YamlImportError } from '@/api/client'
import { ImportYamlDialog } from '../ImportYamlDialog'

const OK_IMPORT = {
  definition: { id: 'wf', name: 'Imported Agent', description: 'desc', root: {}, run_as: 'caller' },
  workflow_summary: { node_count: 2, tool_count: 1, source_count: 0 },
}
const OK_VALIDATE = { valid: true, errors: [], workflow_summary: OK_IMPORT.workflow_summary }

function typeYaml() {
  fireEvent.change(screen.getByPlaceholderText(/registry_version/), {
    target: { value: "registry_version: '1.0.0'\nname: Imported Agent\n" },
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  mutate.mockImplementation((_vars: unknown, opts: { onSuccess: (r: unknown) => void }) =>
    opts.onSuccess({ agent: { id: 'new-agent-1' }, etag: '"v1"' }),
  )
})

describe('ImportYamlDialog', () => {
  it('validates → preview → creates → navigates to the new agent', async () => {
    importYaml.mockResolvedValue(OK_IMPORT)
    validateWorkflow.mockResolvedValue(OK_VALIDATE)
    const onClose = vi.fn()
    render(<ImportYamlDialog open onClose={onClose} />)

    typeYaml()
    fireEvent.click(screen.getByRole('button', { name: 'Validate' }))

    // Preview: name prefilled from definition.name
    await waitFor(() => expect(screen.getByDisplayValue('Imported Agent')).toBeInTheDocument())
    expect(screen.getByText('2 blocks')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Create agent' }))

    expect(mutate).toHaveBeenCalledTimes(1)
    const payload = mutate.mock.calls[0][0] as {
      name: string
      visibility: string
      force: boolean
      definition: Record<string, unknown>
    }
    expect(payload.name).toBe('Imported Agent')
    expect(payload.force).toBe(false)
    // run_as reset to caller; AST name synced to the entered name
    expect(payload.definition.run_as).toBe('caller')
    expect(payload.definition.name).toBe('Imported Agent')
    expect(onClose).toHaveBeenCalled()
    expect(mockNavigate).toHaveBeenCalledWith('/designer/new-agent-1')
  })

  it('renders structured import errors and stays on the source step', async () => {
    importYaml.mockRejectedValue(
      new YamlImportError(
        [{ path: null, kind: 'registry_version_mismatch', message: 'expected 1.0.0, received 1.0' }],
        400,
      ),
    )
    render(<ImportYamlDialog open onClose={vi.fn()} />)

    typeYaml()
    fireEvent.click(screen.getByRole('button', { name: 'Validate' }))

    await waitFor(() => expect(screen.getByText(/registry_version_mismatch/)).toBeInTheDocument())
    // No confirm form appeared.
    expect(screen.queryByRole('button', { name: 'Create agent' })).toBeNull()
    expect(validateWorkflow).not.toHaveBeenCalled()
  })

  it('surfaces semantic-validation errors (would be rejected at save)', async () => {
    importYaml.mockResolvedValue(OK_IMPORT)
    validateWorkflow.mockResolvedValue({
      valid: false,
      errors: [{ path: 'root.children.0', kind: 'validation', message: 'undeclared tool "web"', line: null }],
      workflow_summary: null,
    })
    render(<ImportYamlDialog open onClose={vi.fn()} />)

    typeYaml()
    fireEvent.click(screen.getByRole('button', { name: 'Validate' }))

    await waitFor(() => expect(screen.getByText(/undeclared tool/)).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: 'Create agent' })).toBeNull()
  })

  it('offers "Import anyway" on a critic verdict=fail and retries with force', async () => {
    importYaml.mockResolvedValue(OK_IMPORT)
    validateWorkflow.mockResolvedValue(OK_VALIDATE)
    // First create → critic error; do not auto-succeed.
    mutate.mockImplementationOnce((_v: unknown, opts: { onError: (e: unknown) => void }) =>
      opts.onError({ critique: { verdict: 'fail', summary: 'off-topic' } }),
    )
    render(<ImportYamlDialog open onClose={vi.fn()} />)

    typeYaml()
    fireEvent.click(screen.getByRole('button', { name: 'Validate' }))
    await waitFor(() => expect(screen.getByDisplayValue('Imported Agent')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: 'Create agent' }))
    await waitFor(() => expect(screen.getByText(/critic flagged/i)).toBeInTheDocument())

    // Second attempt via "Import anyway" → force=true
    fireEvent.click(screen.getByRole('button', { name: 'Import anyway' }))
    expect(mutate).toHaveBeenCalledTimes(2)
    expect((mutate.mock.calls[1][0] as { force: boolean }).force).toBe(true)
  })
})
