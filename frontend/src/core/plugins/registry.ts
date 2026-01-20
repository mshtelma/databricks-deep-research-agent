/**
 * ComponentRegistry - Central registry for frontend extensions
 *
 * Manages registration and retrieval of custom output renderers,
 * panels, event formatters, and source badge configurations.
 */

import type {
  FrontendPlugin,
  OutputRenderer,
  CustomPanel,
  EventLabelFormatter,
  SourceBadgeConfig,
  RenderContext,
  PanelSlot,
} from './types';

/**
 * Central registry for frontend plugins and components.
 *
 * Child projects can register custom components to extend the UI
 * without modifying the parent project.
 *
 * @example
 * ```typescript
 * import { ComponentRegistry } from '@deep-research/core/plugins';
 *
 * ComponentRegistry.registerPlugin({
 *   id: 'my-plugin',
 *   name: 'My Plugin',
 *   version: '1.0.0',
 *   outputRenderers: [{
 *     id: 'meeting-prep-renderer',
 *     outputType: 'meeting_prep',
 *     displayName: 'Meeting Preparation',
 *     component: MeetingPrepRenderer,
 *   }],
 * });
 * ```
 */
export class ComponentRegistry {
  private static outputRenderers = new Map<string, OutputRenderer>();
  private static panels = new Map<string, CustomPanel>();
  private static eventFormatters = new Map<string, EventLabelFormatter>();
  private static sourceBadgeConfigs = new Map<string, SourceBadgeConfig>();
  private static plugins = new Map<string, FrontendPlugin>();
  private static initialized = false;

  /**
   * Register a complete plugin with all its components.
   */
  static registerPlugin(plugin: FrontendPlugin): void {
    if (this.plugins.has(plugin.id)) {
      console.warn(`Plugin '${plugin.id}' is already registered. Skipping.`);
      return;
    }

    this.plugins.set(plugin.id, plugin);

    // Register output renderers
    plugin.outputRenderers?.forEach((renderer) => {
      this.registerRenderer(renderer);
    });

    // Register panels
    plugin.panels?.forEach((panel) => {
      this.registerPanel(panel);
    });

    // Register event formatters
    plugin.eventLabelFormatters?.forEach((formatter) => {
      this.registerEventFormatter(formatter);
    });

    // Register source badge configs
    plugin.sourceBadgeConfigs?.forEach((config) => {
      this.registerSourceBadgeConfig(config);
    });

    console.log(`Plugin '${plugin.name}' (${plugin.version}) registered successfully.`);
  }

  /**
   * Unregister a plugin and all its components.
   */
  static unregisterPlugin(pluginId: string): void {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      return;
    }

    // Unregister output renderers
    plugin.outputRenderers?.forEach((renderer) => {
      this.outputRenderers.delete(renderer.outputType);
    });

    // Unregister panels
    plugin.panels?.forEach((panel) => {
      this.panels.delete(panel.id);
    });

    // Unregister event formatters
    plugin.eventLabelFormatters?.forEach((formatter) => {
      this.eventFormatters.delete(formatter.eventType);
    });

    // Unregister source badge configs
    plugin.sourceBadgeConfigs?.forEach((config) => {
      this.sourceBadgeConfigs.delete(config.sourceType);
    });

    this.plugins.delete(pluginId);
    console.log(`Plugin '${pluginId}' unregistered.`);
  }

  /**
   * Register a custom output renderer.
   */
  static registerRenderer(renderer: OutputRenderer, replace = false): void {
    if (this.outputRenderers.has(renderer.outputType) && !replace) {
      console.warn(
        `Output renderer for '${renderer.outputType}' already exists. Use replace=true to override.`
      );
      return;
    }
    this.outputRenderers.set(renderer.outputType, renderer);
  }

  /**
   * Get a renderer for a specific output type.
   */
  static getRenderer(outputType: string): OutputRenderer | undefined {
    return this.outputRenderers.get(outputType);
  }

  /**
   * List all registered output types.
   */
  static listOutputTypes(): string[] {
    return Array.from(this.outputRenderers.keys());
  }

  /**
   * Register a custom panel.
   */
  static registerPanel(panel: CustomPanel, replace = false): void {
    if (this.panels.has(panel.id) && !replace) {
      console.warn(`Panel '${panel.id}' already exists. Use replace=true to override.`);
      return;
    }
    this.panels.set(panel.id, panel);
  }

  /**
   * Get a panel by ID.
   */
  static getPanel(panelId: string): CustomPanel | undefined {
    return this.panels.get(panelId);
  }

  /**
   * Get all panels for a specific slot, ordered by priority.
   */
  static getPanelsForSlot(slot: PanelSlot): CustomPanel[] {
    return Array.from(this.panels.values())
      .filter((panel) => panel.slot === slot)
      .sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100));
  }

  /**
   * Register an event label formatter.
   */
  static registerEventFormatter(formatter: EventLabelFormatter, replace = false): void {
    if (this.eventFormatters.has(formatter.eventType) && !replace) {
      console.warn(
        `Event formatter for '${formatter.eventType}' already exists. Use replace=true to override.`
      );
      return;
    }
    this.eventFormatters.set(formatter.eventType, formatter);
  }

  /**
   * Get an event formatter for a specific event type.
   */
  static getEventFormatter(eventType: string): EventLabelFormatter | undefined {
    return this.eventFormatters.get(eventType);
  }

  /**
   * Format an event label using registered formatters.
   * Falls back to default formatting if no custom formatter exists.
   */
  static formatEventLabel(
    eventType: string,
    eventData: Record<string, unknown>,
    defaultLabel: string
  ): string {
    const formatter = this.eventFormatters.get(eventType);
    if (formatter) {
      return formatter.format(eventData);
    }
    return defaultLabel;
  }

  /**
   * Register a source badge configuration.
   */
  static registerSourceBadgeConfig(config: SourceBadgeConfig, replace = false): void {
    if (this.sourceBadgeConfigs.has(config.sourceType) && !replace) {
      console.warn(
        `Source badge config for '${config.sourceType}' already exists. Use replace=true to override.`
      );
      return;
    }
    this.sourceBadgeConfigs.set(config.sourceType, config);
  }

  /**
   * Get source badge configuration for a source type.
   */
  static getSourceBadgeConfig(sourceType: string): SourceBadgeConfig | undefined {
    return this.sourceBadgeConfigs.get(sourceType);
  }

  /**
   * Initialize all registered plugins.
   */
  static async initialize(context: RenderContext): Promise<void> {
    if (this.initialized) {
      return;
    }

    for (const plugin of this.plugins.values()) {
      if (plugin.initialize) {
        try {
          await plugin.initialize(context);
        } catch (error) {
          console.error(`Failed to initialize plugin '${plugin.id}':`, error);
        }
      }
    }

    this.initialized = true;
  }

  /**
   * Cleanup all registered plugins.
   */
  static async cleanup(): Promise<void> {
    for (const plugin of this.plugins.values()) {
      if (plugin.cleanup) {
        try {
          await plugin.cleanup();
        } catch (error) {
          console.error(`Failed to cleanup plugin '${plugin.id}':`, error);
        }
      }
    }

    this.initialized = false;
  }

  /**
   * Get all registered plugins.
   */
  static getPlugins(): FrontendPlugin[] {
    return Array.from(this.plugins.values());
  }

  /**
   * Check if a plugin is registered.
   */
  static hasPlugin(pluginId: string): boolean {
    return this.plugins.has(pluginId);
  }

  /**
   * Clear all registrations (useful for testing).
   */
  static clear(): void {
    this.outputRenderers.clear();
    this.panels.clear();
    this.eventFormatters.clear();
    this.sourceBadgeConfigs.clear();
    this.plugins.clear();
    this.initialized = false;
  }
}

export default ComponentRegistry;
