/**
 * Template types for prompt template management.
 *
 * This module defines types for:
 * - Template definitions (System, Step, Synthesis, Query)
 * - Template variables with type information
 * - Template rendering
 *
 * NOTE: All field names use camelCase to match the backend BaseSchema
 * serialization (alias_generator=to_camel, serialize_by_alias=True).
 */

/** Template types */
export type TemplateType = 'system' | 'step' | 'synthesis' | 'query';

/** Template visibility options */
export type TemplateVisibility = 'private' | 'workspace';

/** Template origin - who created the template */
export type TemplateOrigin = 'system' | 'plugin' | 'user';

/** Variable types supported in templates */
export type TemplateVariableType = 'string' | 'number' | 'boolean' | 'array' | 'object';

/** Variable definition in a template */
export interface TemplateVariable {
  /** Variable name (without curly braces) */
  name: string;
  /** Variable type for input rendering */
  type: TemplateVariableType;
  /** Whether the variable must be provided */
  required: boolean;
  /** Default value if not provided */
  default?: unknown;
  /** Human-readable description */
  description?: string;
}

/** Template definition */
export interface Template {
  id: string;
  ownerId: string;
  name: string;
  type: TemplateType;
  content: string;
  variables: TemplateVariable[];
  tags: string[];
  description: string | null;
  visibility: TemplateVisibility;
  isDefault: boolean;
  origin?: TemplateOrigin;
  createdAt: string;
  updatedAt: string;
}

/** Request to create a template */
export interface CreateTemplateRequest {
  name: string;
  type: TemplateType;
  content: string;
  variables?: TemplateVariable[];
  tags?: string[];
  description?: string;
  visibility?: TemplateVisibility;
  isDefault?: boolean;
}

/** Request to update a template */
export interface UpdateTemplateRequest {
  name?: string;
  content?: string;
  variables?: TemplateVariable[];
  tags?: string[];
  description?: string;
  visibility?: TemplateVisibility;
  isDefault?: boolean;
}

/** Request to render a template */
export interface RenderTemplateRequest {
  templateId: string;
  variables: Record<string, unknown>;
}

/** Response from template rendering */
export interface RenderTemplateResponse {
  renderedContent: string;
  missingVariables: string[];
  warnings: string[];
}

/** Response for template list */
export interface TemplateListResponse {
  templates: Template[];
  total: number;
  userTemplates: number;
  workspaceTemplates: number;
}

/** Filter parameters for template list */
export interface TemplateListParams {
  type?: TemplateType;
  visibility?: TemplateVisibility;
  search?: string;
  tags?: string[];
  includeSystem?: boolean;
}

// =============================================================================
// Display Utilities
// =============================================================================

/** Human-readable labels for template types */
export const TEMPLATE_TYPE_LABELS: Record<TemplateType, string> = {
  system: 'System Prompt',
  step: 'Research Step',
  synthesis: 'Synthesis',
  query: 'Query Transform',
};

/** Color mapping for template types */
export const TEMPLATE_TYPE_COLORS: Record<TemplateType, string> = {
  system: 'blue',
  step: 'green',
  synthesis: 'purple',
  query: 'orange',
};

/** Human-readable labels for variable types */
export const VARIABLE_TYPE_LABELS: Record<TemplateVariableType, string> = {
  string: 'Text',
  number: 'Number',
  boolean: 'Yes/No',
  array: 'List',
  object: 'JSON Object',
};

/** Human-readable labels for template origins */
export const TEMPLATE_ORIGIN_LABELS: Record<TemplateOrigin, string> = {
  system: 'Built-in',
  plugin: 'Plugin',
  user: 'Custom',
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Extract variable names from template content.
 * Matches {{variable_name}} patterns.
 */
export function extractVariables(content: string): string[] {
  const regex = /\{\{(\w+)\}\}/g;
  const matches = new Set<string>();
  let match;
  while ((match = regex.exec(content)) !== null) {
    const varName = match[1];
    if (varName) {
      matches.add(varName);
    }
  }
  return Array.from(matches);
}

/**
 * Get the label for a template type.
 */
export function getTemplateTypeLabel(type: TemplateType): string {
  return TEMPLATE_TYPE_LABELS[type] || type;
}

/**
 * Get the label for a variable type.
 */
export function getVariableTypeLabel(type: TemplateVariableType): string {
  return VARIABLE_TYPE_LABELS[type] || type;
}

/**
 * Get the label for a template origin.
 */
export function getTemplateOriginLabel(origin: TemplateOrigin): string {
  return TEMPLATE_ORIGIN_LABELS[origin] || origin;
}
