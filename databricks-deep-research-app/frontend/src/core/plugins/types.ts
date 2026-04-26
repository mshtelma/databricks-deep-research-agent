/**
 * Frontend Plugin System Types
 *
 * Defines interfaces for extending the Deep Research UI from child projects.
 * Supports custom output renderers, panels, and event label formatters.
 */

import type { ComponentType } from 'react';
import type { SourceType, QueryMode, ResearchDepth } from '@/types/index';

/**
 * Render context passed to custom renderers
 */
export interface RenderContext {
  /** Current theme (light or dark) */
  theme: 'light' | 'dark';
  /** User ID for personalization */
  userId: string | null;
  /** Callback to navigate to different views */
  navigate?: (path: string) => void;
}

/**
 * Custom output renderer for specific output types
 */
export interface OutputRenderer {
  /** Unique identifier for this renderer */
  id: string;
  /** Output type this renderer handles (e.g., 'meeting_prep') */
  outputType: string;
  /** Human-readable name for the output type */
  displayName: string;
  /** Optional description */
  description?: string;
  /** The React component to render the output */
  component: ComponentType<OutputRendererProps>;
}

/**
 * Props passed to custom output renderer components
 */
export interface OutputRendererProps {
  /** The output data to render (matches the output schema) */
  data: Record<string, unknown>;
  /** Render context with theme and navigation */
  context: RenderContext;
  /** Optional CSS class name */
  className?: string;
}

/**
 * Panel slot positions in the UI
 */
export type PanelSlot =
  | 'sidebar-top'
  | 'sidebar-bottom'
  | 'main-header'
  | 'main-footer'
  | 'research-header'
  | 'research-footer';

/**
 * Custom panel for extending the UI
 */
export interface CustomPanel {
  /** Unique identifier for this panel */
  id: string;
  /** Slot where the panel should be rendered */
  slot: PanelSlot;
  /** Priority for ordering multiple panels in same slot (lower = first) */
  priority?: number;
  /** Human-readable name */
  displayName: string;
  /** Optional description */
  description?: string;
  /** The React component to render */
  component: ComponentType<PanelProps>;
}

/**
 * Props passed to custom panel components
 */
export interface PanelProps {
  /** Render context */
  context: RenderContext;
  /** Whether the panel is expanded (for collapsible panels) */
  isExpanded?: boolean;
  /** Callback to toggle expansion */
  onToggleExpand?: () => void;
  /** Optional CSS class name */
  className?: string;
}

/**
 * Event label formatter for activity feed customization
 */
export interface EventLabelFormatter {
  /** Event type this formatter handles */
  eventType: string;
  /** Format function returning display label */
  format: (eventData: Record<string, unknown>) => string;
  /** Get color class for the event */
  getColorClass?: (eventData: Record<string, unknown>) => string;
  /** Get icon for the event */
  getIcon?: (eventData: Record<string, unknown>) => string;
}

/**
 * Source badge customization for custom source types
 */
export interface SourceBadgeConfig {
  /** Source type this config applies to */
  sourceType: SourceType | string;
  /** Display label */
  label: string;
  /** Icon (emoji or component) */
  icon: string;
  /** Tailwind color classes */
  colorClass: string;
}

/**
 * Configuration for the message input component.
 * Plugins can override default input behavior.
 */
export interface InputConfig {
  /** Show query mode selector (simple/web_search/deep_research). Default: true */
  showModeSelector?: boolean;
  /** Show research depth selector. Default: true */
  showDepthSelector?: boolean;
  /** Show "verify sources" checkbox. Default: true */
  showVerifySources?: boolean;
  /** Custom placeholder text for input field */
  placeholder?: string;
  /** Default query mode when selector is hidden. Used instead of localStorage. */
  defaultQueryMode?: QueryMode;
  /** Default research depth when selector is hidden */
  defaultResearchDepth?: ResearchDepth;
  /** Default verify sources value when checkbox is hidden */
  defaultVerifySources?: boolean;
  /** Default output type for structured output. When set, jobs use this output type. */
  defaultOutputType?: string;
}

/**
 * Configuration for sidebar navigation links.
 * Plugins can hide framework-provided nav links that aren't relevant.
 */
export interface SidebarConfig {
  /** Show "Agents" navigation link in sidebar. Default: true */
  showAgentsLink?: boolean;
  /** Show "Templates" navigation link in sidebar. Default: true */
  showTemplatesLink?: boolean;
}

/**
 * Plugin registration interface
 */
export interface FrontendPlugin {
  /** Unique plugin identifier */
  id: string;
  /** Human-readable name */
  name: string;
  /** Version */
  version: string;
  /** Custom output renderers */
  outputRenderers?: OutputRenderer[];
  /** Custom panels */
  panels?: CustomPanel[];
  /** Event label formatters */
  eventLabelFormatters?: EventLabelFormatter[];
  /** Source badge configurations */
  sourceBadgeConfigs?: SourceBadgeConfig[];
  /** Configuration for message input component */
  inputConfig?: InputConfig;
  /** Configuration for sidebar navigation links */
  sidebarConfig?: SidebarConfig;
  /** Initialization callback */
  initialize?: (context: RenderContext) => void | Promise<void>;
  /** Cleanup callback */
  cleanup?: () => void | Promise<void>;
}
