/**
 * URL safety utilities for preventing XSS via protocol injection.
 */

/**
 * Check if a URL uses a safe protocol (http or https).
 * Prevents javascript: and other dangerous protocol URLs.
 */
export function isSafeUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return ['http:', 'https:'].includes(parsed.protocol);
  } catch {
    return false;
  }
}

/**
 * Open a URL in a new tab only if it uses a safe protocol.
 */
export function safeOpenUrl(url: string): void {
  if (isSafeUrl(url)) {
    window.open(url, '_blank', 'noopener,noreferrer');
  }
}
