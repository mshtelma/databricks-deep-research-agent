/**
 * Minimal transient toast.
 *
 * The app has no global toast provider; designer surfaces use inline alert
 * banners and `chat/MessageExportMenu` rolls its own DOM toast. This is the
 * shared, dependency-free version for fire-and-forget notifications (export
 * downloaded, copied, failed). Safe to call outside the browser (no-ops).
 */
export function showToast(message: string, type: 'success' | 'error' = 'success'): void {
  if (typeof document === 'undefined') return
  const toast = document.createElement('div')
  toast.setAttribute('role', 'status')
  toast.className = [
    'fixed bottom-4 right-4 z-[60] max-w-sm rounded-db-md px-4 py-2',
    'text-[13px] font-medium text-white shadow-db-md transition-opacity duration-200',
    type === 'success' ? 'bg-db-green-700' : 'bg-db-lava-600',
  ].join(' ')
  toast.textContent = message
  document.body.appendChild(toast)
  window.setTimeout(() => {
    toast.style.opacity = '0'
    window.setTimeout(() => toast.remove(), 200)
  }, 2600)
}
