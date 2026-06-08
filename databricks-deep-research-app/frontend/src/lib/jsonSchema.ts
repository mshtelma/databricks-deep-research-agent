export function schemaProperties(
  schema: Record<string, unknown> | null | undefined,
): Record<string, Record<string, unknown>> {
  return schema?.['properties'] &&
    typeof schema['properties'] === 'object' &&
    !Array.isArray(schema['properties'])
    ? (schema['properties'] as Record<string, Record<string, unknown>>)
    : {};
}

export function schemaRequiredKeys(
  schema: Record<string, unknown> | null | undefined,
): string[] {
  return Array.isArray(schema?.['required']) ? (schema['required'] as string[]) : [];
}

export function defaultValueForSchema(schema: Record<string, unknown>): unknown {
  if ('default' in schema) return schema['default'];
  const type = schema['type'];
  if (type === 'string') return '';
  if (type === 'integer' || type === 'number') return 0;
  if (type === 'boolean') return false;
  if (type === 'array') return [];
  if (type === 'object') return {};
  return undefined;
}

export function defaultConfigForSchema(
  schema: Record<string, unknown> | null | undefined,
): Record<string, unknown> {
  // Seed an initial value ONLY for fields that declare an explicit `default`
  // or are `required`. Optional fields without a default are left absent, so
  // "unset" stays distinguishable from a zero-value: a blank `provider` means
  // "inherit the workspace default", and an absent `resolve_redirects` is filled
  // from app config at run time rather than pinned to `false`. (Previously every
  // property was seeded with a type zero-value — '', 0, false — which polluted
  // the config and defeated absent-key defaulting downstream.)
  const config: Record<string, unknown> = {};
  const required = new Set(schemaRequiredKeys(schema));
  for (const [key, propertySchema] of Object.entries(schemaProperties(schema))) {
    if (!('default' in propertySchema) && !required.has(key)) {
      continue;
    }
    const value = defaultValueForSchema(propertySchema);
    if (value !== undefined) {
      config[key] = value;
    }
  }
  return config;
}

export function requiredConfigErrors(
  schema: Record<string, unknown> | null | undefined,
  config: Record<string, unknown>,
): Record<string, string[]> {
  const errors: Record<string, string[]> = {};
  for (const key of schemaRequiredKeys(schema)) {
    const value = config[key];
    if (
      value === undefined ||
      value === null ||
      (typeof value === 'string' && value.trim().length === 0)
    ) {
      errors[key] = ['Required'];
    }
  }
  return errors;
}
