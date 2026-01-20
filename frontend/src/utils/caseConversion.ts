/**
 * Utility functions for case conversion between snake_case and camelCase.
 *
 * Used to bridge the gap between Python backend (snake_case) and
 * TypeScript frontend (camelCase) conventions.
 */

/**
 * Convert a single snake_case string to camelCase.
 *
 * @example
 * snakeToCamelString('message_id') // 'messageId'
 * snakeToCamelString('research_session_id') // 'researchSessionId'
 */
export function snakeToCamelString(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

/**
 * Recursively convert snake_case keys to camelCase in an object or array.
 * Handles nested objects and arrays.
 *
 * @example
 * snakeToCamel({ message_id: '123', research_session: { step_index: 0 } })
 * // { messageId: '123', researchSession: { stepIndex: 0 } }
 */
export function snakeToCamel<T = unknown>(obj: unknown): T {
  if (obj === null || obj === undefined) {
    return obj as T;
  }

  if (Array.isArray(obj)) {
    return obj.map(snakeToCamel) as T;
  }

  if (typeof obj === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
      const camelKey = snakeToCamelString(key);
      result[camelKey] = snakeToCamel(value);
    }
    return result as T;
  }

  return obj as T;
}

/**
 * Convert a single camelCase string to snake_case.
 *
 * @example
 * camelToSnakeString('messageId') // 'message_id'
 * camelToSnakeString('researchSessionId') // 'research_session_id'
 */
export function camelToSnakeString(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

/**
 * Recursively convert camelCase keys to snake_case in an object or array.
 * Handles nested objects and arrays.
 *
 * @example
 * camelToSnake({ messageId: '123', researchSession: { stepIndex: 0 } })
 * // { message_id: '123', research_session: { step_index: 0 } }
 */
export function camelToSnake<T = unknown>(obj: unknown): T {
  if (obj === null || obj === undefined) {
    return obj as T;
  }

  if (Array.isArray(obj)) {
    return obj.map(camelToSnake) as T;
  }

  if (typeof obj === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
      const snakeKey = camelToSnakeString(key);
      result[snakeKey] = camelToSnake(value);
    }
    return result as T;
  }

  return obj as T;
}

/**
 * Utility to safely get a value with both snake_case and camelCase key variants.
 * Returns the first defined value found.
 *
 * @example
 * getWithCaseFallback(event, 'messageId', 'message_id') // returns event.messageId ?? event.message_id
 */
export function getWithCaseFallback<T>(
  obj: Record<string, unknown>,
  camelKey: string,
  snakeKey: string
): T | undefined {
  return (obj[camelKey] ?? obj[snakeKey]) as T | undefined;
}
