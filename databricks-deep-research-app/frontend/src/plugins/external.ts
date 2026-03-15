/**
 * External Plugin Entry Point
 *
 * This file is the hook for child projects to inject their plugins.
 * Child projects override this via Vite alias to register custom plugins.
 *
 * @example
 * // In child project's vite.config.ts:
 * resolve: {
 *   alias: {
 *     '@plugins/external': './frontend-plugin/index.ts'
 *   }
 * }
 */

import { ComponentRegistry } from '@/core/plugins';

/**
 * Register external plugins. Called before React renders.
 * Default implementation does nothing - child projects override via Vite alias.
 */
export function registerExternalPlugins(): void {
  // Default: no external plugins
  // Child projects replace this entire file via Vite alias
  console.debug('[plugins] No external plugins registered (using default)');
}

/**
 * Get the ComponentRegistry for external plugins to use.
 * Re-exported here so child plugins can import from single location.
 */
export { ComponentRegistry };
