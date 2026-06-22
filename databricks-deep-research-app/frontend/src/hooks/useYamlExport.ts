import * as React from 'react'
import { copyToClipboard, downloadTextFile } from '@/lib/download'
import { showToast } from '@/lib/toast'

export interface YamlExportActions {
  /** Fetch the YAML and trigger a file download. */
  download: () => Promise<void>
  /** Fetch the YAML and copy it to the clipboard (falls back to a toast). */
  copy: () => Promise<void>
  /** True while a download/copy is in flight (lazy `getYaml` may hit the network). */
  busy: boolean
}

/**
 * Shared download/copy handlers for a YAML payload produced lazily by `getYaml`.
 *
 * `getYaml` is invoked on demand (so the export endpoint is only hit on click),
 * with success/error surfaced via {@link showToast}. Reused by the editor's
 * ExportYamlMenu and the agents-list per-agent menu so both behave identically.
 */
export function useYamlExport(
  getYaml: () => Promise<string>,
  filename: string,
): YamlExportActions {
  const [busy, setBusy] = React.useState(false)

  const run = React.useCallback(
    async (consume: (yaml: string) => void | Promise<void>) => {
      setBusy(true)
      try {
        const yaml = await getYaml()
        await consume(yaml)
      } catch (err) {
        showToast(err instanceof Error ? err.message : 'Export failed', 'error')
      } finally {
        setBusy(false)
      }
    },
    [getYaml],
  )

  const download = React.useCallback(
    () =>
      run((yaml) => {
        downloadTextFile(yaml, filename, 'text/yaml')
        showToast('YAML downloaded')
      }),
    [run, filename],
  )

  const copy = React.useCallback(
    () =>
      run(async (yaml) => {
        const ok = await copyToClipboard(yaml)
        showToast(
          ok ? 'YAML copied to clipboard' : 'Copy unavailable — use Download',
          ok ? 'success' : 'error',
        )
      }),
    [run],
  )

  return { download, copy, busy }
}
