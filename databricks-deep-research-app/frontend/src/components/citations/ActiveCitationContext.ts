import * as React from 'react';

/**
 * The citation key whose evidence card is currently open (`null` = none).
 *
 * Carries the "active marker" highlight to {@link CitationMarker} via context
 * instead of as a prop baked into the memoized `components` map of
 * `MarkdownRenderer`. This keeps the rendered marker DOM nodes stable across
 * hovers (only the active marker re-renders, none remount), which is what lets
 * the popover anchor stay valid and avoids the React #185 remount loop.
 */
export const ActiveCitationContext = React.createContext<string | null>(null);

ActiveCitationContext.displayName = 'ActiveCitationContext';
