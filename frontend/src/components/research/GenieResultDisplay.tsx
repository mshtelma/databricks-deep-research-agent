/**
 * GenieResultDisplay - Component to display Genie query results.
 *
 * Features (T026):
 * - Tabular data display with truncation (max 100 rows)
 * - Collapsible section for generated SQL
 * - Narrative summary section
 * - Export to CSV button
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { GenieResult } from '@/types/dataSources';

interface GenieResultDisplayProps {
  result: GenieResult;
  sourceName?: string;
  className?: string;
}

export function GenieResultDisplay({
  result,
  sourceName,
  className,
}: GenieResultDisplayProps) {
  const [showSql, setShowSql] = React.useState(false);
  const [showAllRows, setShowAllRows] = React.useState(false);

  const displayRows = showAllRows ? result.rows : result.rows.slice(0, 10);
  const hasMoreRows = result.rows.length > 10 && !showAllRows;

  const handleExportCsv = () => {
    // Build CSV content
    const headers = result.columns.map((col) => col.name).join(',');
    const rows = result.rows.map((row) =>
      result.columns
        .map((col) => {
          const value = row[col.name];
          // Escape quotes and wrap in quotes if contains comma or quote
          if (value === null || value === undefined) return '';
          const str = String(value);
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        })
        .join(',')
    );
    const csvContent = [headers, ...rows].join('\n');

    // Download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `genie-result-${sourceName || 'export'}-${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={cn('rounded-lg border bg-card', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <DatabaseIcon className="h-4 w-4 text-purple-600" />
          <span className="font-medium text-sm">
            {sourceName ? `Genie: ${sourceName}` : 'Genie Query Result'}
          </span>
          {result.truncated && (
            <span className="px-2 py-0.5 rounded-full text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
              Truncated
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {result.totalRows} row{result.totalRows !== 1 ? 's' : ''}
            {result.executionTimeMs && ` in ${result.executionTimeMs}ms`}
          </span>
          <Button variant="ghost" size="sm" onClick={handleExportCsv}>
            <DownloadIcon className="h-4 w-4 mr-1" />
            CSV
          </Button>
        </div>
      </div>

      {/* Narrative Summary */}
      {result.narrativeSummary && (
        <div className="px-4 py-3 border-b bg-purple-50/50 dark:bg-purple-950/20">
          <p className="text-sm text-foreground">{result.narrativeSummary}</p>
        </div>
      )}

      {/* Data Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              {result.columns.map((column) => (
                <th
                  key={column.name}
                  className="px-3 py-2 text-left font-medium text-muted-foreground whitespace-nowrap"
                >
                  <div className="flex items-center gap-1">
                    <span>{column.name}</span>
                    <span className="text-xs text-muted-foreground/70">
                      ({formatColumnType(column.type)})
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className={cn(
                  'border-b last:border-b-0',
                  rowIndex % 2 === 0 ? 'bg-background' : 'bg-muted/20'
                )}
              >
                {result.columns.map((column) => (
                  <td key={column.name} className="px-3 py-2 whitespace-nowrap">
                    <CellValue value={row[column.name]} type={column.type} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Show More Rows Button */}
      {hasMoreRows && (
        <div className="px-4 py-2 border-t text-center">
          <Button variant="ghost" size="sm" onClick={() => setShowAllRows(true)}>
            Show all {result.rows.length} rows
          </Button>
        </div>
      )}

      {/* Generated SQL Section */}
      {result.generatedSql && (
        <div className="border-t">
          <button
            type="button"
            onClick={() => setShowSql(!showSql)}
            className="w-full flex items-center justify-between px-4 py-2 text-sm text-muted-foreground hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <CodeIcon className="h-4 w-4" />
              <span>Generated SQL</span>
            </div>
            <ChevronIcon
              className={cn(
                'h-4 w-4 transition-transform',
                showSql && 'rotate-180'
              )}
            />
          </button>
          {showSql && (
            <div className="px-4 py-3 bg-muted/30 border-t">
              <pre className="text-xs font-mono whitespace-pre-wrap overflow-x-auto text-foreground">
                {result.generatedSql}
              </pre>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigator.clipboard.writeText(result.generatedSql!)}
                className="mt-2"
              >
                <CopyIcon className="h-3 w-3 mr-1" />
                Copy SQL
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Helper to format column types
function formatColumnType(type: string): string {
  const typeMap: Record<string, string> = {
    STRING: 'str',
    INTEGER: 'int',
    LONG: 'long',
    FLOAT: 'float',
    DOUBLE: 'double',
    BOOLEAN: 'bool',
    DATE: 'date',
    TIMESTAMP: 'ts',
    DECIMAL: 'dec',
  };
  return typeMap[type.toUpperCase()] || type.toLowerCase();
}

// Cell value renderer
function CellValue({ value, type }: { value: unknown; type: string }) {
  if (value === null || value === undefined) {
    return <span className="text-muted-foreground italic">null</span>;
  }

  const typeUpper = type.toUpperCase();

  // Boolean
  if (typeUpper === 'BOOLEAN') {
    return (
      <span
        className={cn(
          'inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium',
          value ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        )}
      >
        {String(value)}
      </span>
    );
  }

  // Numbers
  if (['INTEGER', 'LONG', 'FLOAT', 'DOUBLE', 'DECIMAL'].includes(typeUpper)) {
    const numValue = typeof value === 'number' ? value : parseFloat(String(value));
    if (!isNaN(numValue)) {
      return (
        <span className="font-mono text-right">
          {typeUpper.includes('FLOAT') || typeUpper.includes('DOUBLE') || typeUpper === 'DECIMAL'
            ? numValue.toLocaleString(undefined, { maximumFractionDigits: 4 })
            : numValue.toLocaleString()}
        </span>
      );
    }
  }

  // Default: string representation
  const strValue = String(value);
  // Truncate long strings
  if (strValue.length > 100) {
    return (
      <span title={strValue}>
        {strValue.substring(0, 100)}
        <span className="text-muted-foreground">...</span>
      </span>
    );
  }
  return <span>{strValue}</span>;
}

// Icons
function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14a9 3 0 0 0 18 0V5" />
      <path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}

function DownloadIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}

function CodeIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function CopyIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
    </svg>
  );
}

export default GenieResultDisplay;
