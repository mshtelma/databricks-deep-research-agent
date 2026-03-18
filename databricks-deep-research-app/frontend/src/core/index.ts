/**
 * Deep Research Core Exports
 *
 * This module exports the public API for child projects to extend
 * the Deep Research frontend application.
 *
 * @example
 * ```typescript
 * // Import from the core module
 * import { ComponentRegistry, type FrontendPlugin } from '@deep-research/core';
 *
 * // Or import specific sub-modules
 * import { ComponentRegistry } from '@deep-research/core/plugins';
 * ```
 */

// Plugin System
export * from './plugins';

// Re-export types that child projects commonly need
export type { Source, SourceType, Message, Chat, ResearchSession } from '@/types/index';
