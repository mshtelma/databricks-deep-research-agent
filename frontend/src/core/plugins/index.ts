/**
 * Frontend Plugin System
 *
 * Public exports for child projects to extend the Deep Research UI.
 *
 * @example
 * ```typescript
 * // In a child project
 * import {
 *   ComponentRegistry,
 *   type FrontendPlugin,
 *   type OutputRenderer,
 * } from '@deep-research/core/plugins';
 *
 * const myPlugin: FrontendPlugin = {
 *   id: 'my-plugin',
 *   name: 'My Plugin',
 *   version: '1.0.0',
 *   outputRenderers: [...],
 *   panels: [...],
 * };
 *
 * ComponentRegistry.registerPlugin(myPlugin);
 * ```
 */

// Registry
export { ComponentRegistry, default } from './registry';

// Types
export type {
  FrontendPlugin,
  OutputRenderer,
  OutputRendererProps,
  CustomPanel,
  PanelProps,
  PanelSlot,
  EventLabelFormatter,
  SourceBadgeConfig,
  RenderContext,
} from './types';

// Default output renderers
export {
  defaultOutputRenderers,
  getRendererForType,
  GenericJsonRenderer,
  SynthesisReportRenderer,
} from './outputTypes';
