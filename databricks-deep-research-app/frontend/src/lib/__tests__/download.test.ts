import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { downloadTextFile, copyToClipboard, slugifyFilename } from '../download'

describe('slugifyFilename', () => {
  it('lowercases, replaces non-alphanumerics, and appends the extension', () => {
    expect(slugifyFilename('My Agent!', 'yaml')).toBe('my-agent.yaml')
    expect(slugifyFilename('  Trim  Me  ', 'yaml')).toBe('trim-me.yaml')
    expect(slugifyFilename('Café—Münch', 'yaml')).toBe('caf-m-nch.yaml')
  })

  it('falls back to "agent" when no usable characters remain', () => {
    expect(slugifyFilename('', 'yaml')).toBe('agent.yaml')
    expect(slugifyFilename('!!!', 'yaml')).toBe('agent.yaml')
  })
})

describe('downloadTextFile', () => {
  beforeEach(() => {
    // jsdom does not implement object URLs; provide stubs to spy on.
    if (!('createObjectURL' in URL)) {
      (URL as unknown as { createObjectURL: () => string }).createObjectURL = () => ''
    }
    if (!('revokeObjectURL' in URL)) {
      (URL as unknown as { revokeObjectURL: () => void }).revokeObjectURL = () => {}
    }
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('creates an anchor, clicks it, and revokes the object URL', () => {
    const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake')
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    // Hold the clicked anchor on an object: a bare `let` assigned only inside the
    // mock closure stays narrowed to `null` at the read site (TS can't see the
    // closure run), so `?.getAttribute` would resolve against `never`. A property
    // reference reverts to its declared type after the downloadTextFile() call.
    const clickedRef: { current: HTMLAnchorElement | null } = { current: null }
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(function (this: HTMLAnchorElement) {
        clickedRef.current = this
      })

    downloadTextFile('hello: world', 'demo.yaml', 'text/yaml')

    expect(createSpy).toHaveBeenCalledTimes(1)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(clickedRef.current?.getAttribute('download')).toBe('demo.yaml')
    expect(revokeSpy).toHaveBeenCalledWith('blob:fake')
    // anchor is cleaned up from the DOM
    expect(document.querySelector('a[download]')).toBeNull()
  })
})

describe('copyToClipboard', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('returns true when the Clipboard API succeeds', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    expect(await copyToClipboard('text')).toBe(true)
    expect(writeText).toHaveBeenCalledWith('text')
  })

  it('returns false when the Clipboard API is unavailable', async () => {
    vi.stubGlobal('navigator', {})
    expect(await copyToClipboard('text')).toBe(false)
  })

  it('returns false (never throws) when writeText rejects', async () => {
    vi.stubGlobal('navigator', {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
    })
    expect(await copyToClipboard('text')).toBe(false)
  })
})
