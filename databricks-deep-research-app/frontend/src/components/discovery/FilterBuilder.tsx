/**
 * Filter Builder component for Vector Search queries.
 *
 * Provides an intuitive interface for building filter expressions:
 * - Add/remove filter rows
 * - Column dropdown (populated from index metadata)
 * - Operator dropdown (filtered by column data type)
 * - Value input (type-aware: text, number, date, list for IN)
 * - Real-time validation with error messages
 *
 * Part of US9b (T010w).
 */

import { useCallback } from 'react';
import type { FilterExpression, FilterColumnInfo, FilterOperator } from '@/types/discovery';

interface FilterBuilderProps {
  /** Available columns for filtering */
  columns: FilterColumnInfo[];
  /** Current filter expressions */
  filters: FilterExpression[];
  /** Called when filters change */
  onChange: (filters: FilterExpression[]) => void;
  /** Whether the builder is disabled */
  disabled?: boolean;
  /** Maximum number of filters allowed */
  maxFilters?: number;
}

// Operator labels for display
const OPERATOR_LABELS: Record<FilterOperator, string> = {
  '=': 'equals',
  '!=': 'not equals',
  '<': 'less than',
  '<=': 'less or equal',
  '>': 'greater than',
  '>=': 'greater or equal',
  LIKE: 'contains',
  'NOT LIKE': 'not contains',
  IN: 'in list',
};

// Operators available for each data type
const OPERATORS_BY_TYPE: Record<string, FilterOperator[]> = {
  string: ['=', '!=', 'LIKE', 'NOT LIKE', 'IN'],
  integer: ['=', '!=', '<', '<=', '>', '>=', 'IN'],
  float: ['=', '!=', '<', '<=', '>', '>='],
  timestamp: ['=', '!=', '<', '<=', '>', '>='],
  boolean: ['=', '!='],
};

interface FilterRowProps {
  filter: FilterExpression;
  columns: FilterColumnInfo[];
  index: number;
  onChange: (index: number, filter: FilterExpression) => void;
  onRemove: (index: number) => void;
  disabled?: boolean;
}

function FilterRow({ filter, columns, index, onChange, onRemove, disabled }: FilterRowProps) {
  const selectedColumn = columns.find((c) => c.name === filter.column);
  const availableOperators =
    selectedColumn ? OPERATORS_BY_TYPE[selectedColumn.data_type] || ['='] : ['='];

  const handleColumnChange = useCallback(
    (columnName: string) => {
      const column = columns.find((c) => c.name === columnName);
      const validOperators = column ? OPERATORS_BY_TYPE[column.data_type] || ['='] : ['='];
      // Reset operator if current one isn't valid for new column type
      const newOperator = validOperators.includes(filter.operator) ? filter.operator : '=';

      onChange(index, {
        ...filter,
        column: columnName,
        operator: newOperator,
        value: '', // Reset value when column changes
      });
    },
    [columns, filter, index, onChange]
  );

  const handleOperatorChange = useCallback(
    (operator: FilterOperator) => {
      // If switching to IN operator, convert value to array
      const newValue =
        operator === 'IN'
          ? Array.isArray(filter.value)
            ? filter.value
            : filter.value
              ? [filter.value]
              : []
          : Array.isArray(filter.value)
            ? filter.value[0] || ''
            : filter.value;

      onChange(index, {
        ...filter,
        operator,
        value: newValue,
      });
    },
    [filter, index, onChange]
  );

  const handleValueChange = useCallback(
    (rawValue: string) => {
      let value: string | number | (string | number)[];

      if (filter.operator === 'IN') {
        // Parse comma-separated values
        value = rawValue
          .split(',')
          .map((v) => v.trim())
          .filter(Boolean);
      } else if (selectedColumn?.data_type === 'integer') {
        value = parseInt(rawValue, 10) || 0;
      } else if (selectedColumn?.data_type === 'float') {
        value = parseFloat(rawValue) || 0;
      } else {
        value = rawValue;
      }

      onChange(index, { ...filter, value });
    },
    [filter, index, onChange, selectedColumn]
  );

  // Format value for display
  const displayValue = Array.isArray(filter.value) ? filter.value.join(', ') : String(filter.value);

  return (
    <div className="flex items-start gap-2 p-3 bg-white dark:bg-gray-700 rounded-md border border-gray-200 dark:border-gray-600">
      {/* Column Select */}
      <div className="flex-1 min-w-[120px]">
        <select
          value={filter.column}
          onChange={(e) => handleColumnChange(e.target.value)}
          disabled={disabled}
          className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
        >
          <option value="">Select column...</option>
          {columns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.data_type})
            </option>
          ))}
        </select>
      </div>

      {/* Operator Select */}
      <div className="w-32">
        <select
          value={filter.operator}
          onChange={(e) => handleOperatorChange(e.target.value as FilterOperator)}
          disabled={disabled || !filter.column}
          className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
        >
          {availableOperators.map((op) => (
            <option key={op} value={op}>
              {OPERATOR_LABELS[op as FilterOperator]}
            </option>
          ))}
        </select>
      </div>

      {/* Value Input */}
      <div className="flex-1 min-w-[150px]">
        <input
          type={selectedColumn?.data_type === 'boolean' ? 'checkbox' : 'text'}
          value={displayValue}
          onChange={(e) => handleValueChange(e.target.value)}
          disabled={disabled || !filter.column}
          placeholder={filter.operator === 'IN' ? 'value1, value2, ...' : 'Enter value...'}
          className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
        />
        {filter.operator === 'IN' && (
          <p className="mt-0.5 text-xs text-gray-500">Comma-separated values (max 1,024)</p>
        )}
      </div>

      {/* Remove Button */}
      <button
        type="button"
        onClick={() => onRemove(index)}
        disabled={disabled}
        className="p-1.5 text-gray-400 hover:text-red-500 disabled:opacity-50"
        title="Remove filter"
      >
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    </div>
  );
}

export function FilterBuilder({
  columns,
  filters,
  onChange,
  disabled = false,
  maxFilters = 10,
}: FilterBuilderProps) {
  const handleFilterChange = useCallback(
    (index: number, filter: FilterExpression) => {
      const newFilters = [...filters];
      newFilters[index] = filter;
      onChange(newFilters);
    },
    [filters, onChange]
  );

  const handleAddFilter = useCallback(() => {
    if (filters.length >= maxFilters) return;

    const newFilter: FilterExpression = {
      column: columns[0]?.name || '',
      operator: '=',
      value: '',
    };
    onChange([...filters, newFilter]);
  }, [columns, filters, maxFilters, onChange]);

  const handleRemoveFilter = useCallback(
    (index: number) => {
      const newFilters = filters.filter((_, i) => i !== index);
      onChange(newFilters);
    },
    [filters, onChange]
  );

  // Validate filters and collect errors
  const validationErrors: string[] = [];
  filters.forEach((filter, index) => {
    if (!filter.column) {
      validationErrors.push(`Filter ${index + 1}: Column is required`);
    }
    if (filter.operator === 'IN' && Array.isArray(filter.value) && filter.value.length > 1024) {
      validationErrors.push(`Filter ${index + 1}: IN clause exceeds 1,024 value limit`);
    }
  });

  return (
    <div className="space-y-3">
      {/* Filter List */}
      {filters.length === 0 ? (
        <p className="text-sm text-gray-500 dark:text-gray-400 py-2">
          No filters configured. Click "Add Filter" to filter results by column values.
        </p>
      ) : (
        <div className="space-y-2">
          {filters.map((filter, index) => (
            <FilterRow
              key={index}
              filter={filter}
              columns={columns}
              index={index}
              onChange={handleFilterChange}
              onRemove={handleRemoveFilter}
              disabled={disabled}
            />
          ))}
        </div>
      )}

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="text-sm text-red-600 dark:text-red-400">
          {validationErrors.map((error, i) => (
            <p key={i}>{error}</p>
          ))}
        </div>
      )}

      {/* Add Filter Button */}
      <button
        type="button"
        onClick={handleAddFilter}
        disabled={disabled || filters.length >= maxFilters || columns.length === 0}
        className="flex items-center gap-2 px-3 py-1.5 text-sm text-blue-600 dark:text-blue-400
                   hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        Add Filter
        {filters.length > 0 && (
          <span className="text-gray-400 dark:text-gray-500">
            ({filters.length}/{maxFilters})
          </span>
        )}
      </button>

      {columns.length === 0 && (
        <p className="text-sm text-amber-600 dark:text-amber-400">
          No filterable columns available for this index.
        </p>
      )}
    </div>
  );
}
