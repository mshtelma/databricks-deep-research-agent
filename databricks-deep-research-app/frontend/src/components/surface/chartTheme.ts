/**
 * Chart theming bridge: SVG chart primitives use fill/stroke props, so the
 * Tailwind design tokens are mirrored here as constants.
 */

/** Series palette (cycled per y-key). */
export const CHART_SERIES_COLORS = [
  '#1B3139', // db navy
  '#FF3621', // db lava
  '#00A972', // db green
  '#FCA700', // db yellow
  '#4299E0', // db blue
  '#98102A', // db maroon
] as const;

export const CHART_GRID_COLOR = '#E8ECF0';
export const CHART_AXIS_COLOR = '#5A6F77';
export const CHART_TICK_FONT_SIZE = 11;

export function seriesColor(index: number): string {
  return CHART_SERIES_COLORS[index % CHART_SERIES_COLORS.length]!;
}
