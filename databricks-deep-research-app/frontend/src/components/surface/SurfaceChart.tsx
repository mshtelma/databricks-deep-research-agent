/* eslint-disable react-refresh/only-export-components */
/**
 * Dependency-free renderer for the surface `Chart` component (bar | line).
 *
 * Default export so the catalog can `React.lazy(() => import(...))`. Data is
 * table-shaped structured output: an array of row objects plus x/y column keys.
 * Rows with non-numeric y values are dropped per-series; `[Key]` citation
 * markers are stripped from x labels for display.
 */

import {
  CHART_AXIS_COLOR,
  CHART_GRID_COLOR,
  CHART_TICK_FONT_SIZE,
  seriesColor,
} from './chartTheme';

const CITATION_MARKER_RE = /\[[A-Za-z0-9][A-Za-z0-9-]*\]/g;

export interface SurfaceChartProps {
  rows: Record<string, unknown>[];
  kind: 'bar' | 'line';
  xKey: string;
  yKeys: string[];
  height?: number;
}

export function sanitizeRows(
  rows: Record<string, unknown>[],
  xKey: string,
  yKeys: string[],
): Record<string, unknown>[] {
  return rows.map((row) => {
    const out: Record<string, unknown> = {};
    const rawX = row[xKey];
    out[xKey] =
      typeof rawX === 'string'
        ? rawX.replace(CITATION_MARKER_RE, '').replace(/\s+/g, ' ').trim()
        : rawX;
    for (const key of yKeys) {
      const value = Number(row[key]);
      out[key] = Number.isFinite(value) ? value : null; // null → gap, no crash
    }
    return out;
  });
}

const SVG_WIDTH = 640;
const PADDING = { top: 16, right: 18, bottom: 38, left: 48 } as const;

function numericValues(
  data: Record<string, unknown>[],
  yKeys: string[],
): number[] {
  const values: number[] = [];
  for (const row of data) {
    for (const key of yKeys) {
      const value = row[key];
      if (typeof value === 'number' && Number.isFinite(value)) values.push(value);
    }
  }
  return values;
}

function truncateLabel(value: unknown): string {
  const label = value === undefined || value === null ? '' : String(value);
  return label.length > 14 ? `${label.slice(0, 13)}...` : label;
}

export function SurfaceChart({
  rows,
  kind,
  xKey,
  yKeys,
  height = 240,
}: SurfaceChartProps): React.ReactElement {
  const data = sanitizeRows(rows, xKey, yKeys);
  const values = numericValues(data, yKeys);
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(1, ...values);
  const yMin = minValue === maxValue ? minValue - 1 : minValue;
  const yMax = minValue === maxValue ? maxValue + 1 : maxValue;
  const chartWidth = SVG_WIDTH - PADDING.left - PADDING.right;
  const chartHeight = height - PADDING.top - PADDING.bottom;
  const yScale = (value: number): number =>
    PADDING.top + ((yMax - value) / (yMax - yMin)) * chartHeight;
  const xScale = (index: number): number =>
    data.length <= 1
      ? PADDING.left + chartWidth / 2
      : PADDING.left + (index / (data.length - 1)) * chartWidth;
  const zeroY = yScale(Math.max(0, yMin));
  const labelEvery = Math.max(1, Math.ceil(data.length / 6));

  return (
    <div
      className="w-full overflow-hidden rounded-db-md border border-db-gray-lines bg-white font-db-sans"
      data-testid="surface-chart"
    >
      <svg
        role="img"
        aria-label="Surface chart"
        viewBox={`0 0 ${SVG_WIDTH} ${height}`}
        className="h-auto w-full"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = PADDING.top + ratio * chartHeight;
          const value = yMax - ratio * (yMax - yMin);
          return (
            <g key={ratio}>
              <line
                x1={PADDING.left}
                x2={SVG_WIDTH - PADDING.right}
                y1={y}
                y2={y}
                stroke={CHART_GRID_COLOR}
                strokeDasharray="3 3"
              />
              <text
                x={PADDING.left - 8}
                y={y + 4}
                textAnchor="end"
                fontSize={CHART_TICK_FONT_SIZE}
                fill={CHART_AXIS_COLOR}
              >
                {Number.isInteger(value) ? value : value.toFixed(1)}
              </text>
            </g>
          );
        })}
        <line
          x1={PADDING.left}
          x2={SVG_WIDTH - PADDING.right}
          y1={zeroY}
          y2={zeroY}
          stroke={CHART_AXIS_COLOR}
        />
        {kind === 'bar'
          ? yKeys.flatMap((key, seriesIndex) => {
              const groupWidth = chartWidth / Math.max(data.length, 1);
              const barWidth = Math.max(3, groupWidth / (yKeys.length + 0.75));
              return data.map((row, rowIndex) => {
                const value = row[key];
                if (typeof value !== 'number' || !Number.isFinite(value)) return null;
                const y = yScale(value);
                const x =
                  PADDING.left +
                  rowIndex * groupWidth +
                  seriesIndex * barWidth +
                  groupWidth * 0.15;
                return (
                  <rect
                    key={`${key}-${rowIndex}`}
                    x={x}
                    y={Math.min(y, zeroY)}
                    width={barWidth}
                    height={Math.max(1, Math.abs(zeroY - y))}
                    fill={seriesColor(seriesIndex)}
                  />
                );
              });
            })
          : yKeys.map((key, seriesIndex) => {
              const points = data
                .map((row, rowIndex) => {
                  const value = row[key];
                  return typeof value === 'number' && Number.isFinite(value)
                    ? `${xScale(rowIndex)},${yScale(value)}`
                    : null;
                })
                .filter((point): point is string => point !== null)
                .join(' ');
              return (
                <polyline
                  key={key}
                  points={points}
                  fill="none"
                  stroke={seriesColor(seriesIndex)}
                  strokeWidth={2}
                />
              );
            })}
        {data.map((row, index) =>
          index % labelEvery === 0 ? (
            <text
              key={`x-${index}`}
              x={xScale(index)}
              y={height - 12}
              textAnchor="middle"
              fontSize={CHART_TICK_FONT_SIZE}
              fill={CHART_AXIS_COLOR}
            >
              {truncateLabel(row[xKey])}
            </text>
          ) : null,
        )}
        {yKeys.length > 1 &&
          yKeys.map((key, index) => (
            <g key={`legend-${key}`} transform={`translate(${PADDING.left + index * 110}, 12)`}>
              <rect width="8" height="8" fill={seriesColor(index)} />
              <text x="12" y="8" fontSize="11" fill={CHART_AXIS_COLOR}>
                {key}
              </text>
            </g>
          ))}
      </svg>
    </div>
  );
}

export default SurfaceChart;
