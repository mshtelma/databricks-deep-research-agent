/**
 * API client for the configuration catalog endpoint.
 *
 * Part of 009-custom-agent-config (T024).
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

/** Information about a single model endpoint. */
export interface EndpointInfo {
  name: string;
  endpointIdentifier: string;
  maxContextWindow: number;
  supportsStructuredOutput: boolean;
}

/** Information about a model tier/category. */
export interface ModelCategoryInfo {
  name: string;
  defaultEndpoints: string[];
  temperature: number;
  maxTokens: number;
}

/** Response from the endpoint catalog API. */
export interface EndpointCatalogResponse {
  categories: Record<string, ModelCategoryInfo>;
  endpoints: Record<string, EndpointInfo>;
}

/**
 * Fetch the model endpoint catalog.
 *
 * Returns all available model tiers (categories) and their endpoints,
 * used by the agent editor to populate model override dropdowns.
 */
export async function getModelCatalog(): Promise<EndpointCatalogResponse> {
  const response = await fetch(`${API_BASE_URL}/config/model-catalog`, {
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch model catalog: ${response.statusText}`);
  }

  return response.json();
}

/** Summary of a workspace serving endpoint. */
export interface ServingEndpointSummary {
  name: string;
  endpointType: string;
  state: string;
}

/** Response from the serving endpoints API. */
export interface ServingEndpointsResponse {
  endpoints: ServingEndpointSummary[];
  configEndpointNames: string[];
}

/** Fetch workspace serving endpoints for autocomplete. */
export async function getServingEndpoints(): Promise<ServingEndpointsResponse> {
  const response = await fetch(`${API_BASE_URL}/config/serving-endpoints`, {
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch serving endpoints: ${response.statusText}`);
  }
  return response.json();
}
