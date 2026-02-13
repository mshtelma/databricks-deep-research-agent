/**
 * Query Configuration Panel for Vector Search sources.
 *
 * Allows users to configure:
 * - Query type (ANN, HYBRID, FULL_TEXT)
 * - Number of results
 * - Score threshold
 * - Reranking settings
 * - Filters (via FilterBuilder)
 *
 * Part of US9b (T010v).
 */

import { useState, useEffect, useCallback } from 'react';
import type {
  VectorSearchQueryConfig,
  VectorSearchMetadata,
  QueryType,
  FilterExpression,
} from '@/types/discovery';
import {
  getQueryTypeLabel,
  getQueryTypeDescription,
  createDefaultQueryConfig,
} from '@/types/discovery';
import { FilterBuilder } from './FilterBuilder';

interface QueryConfigPanelProps {
  /** Source ID for API calls */
  sourceId: string;
  /** Vector Search metadata with capabilities */
  metadata: VectorSearchMetadata;
  /** Current configuration (undefined = use defaults) */
  config?: VectorSearchQueryConfig;
  /** Called when configuration changes */
  onChange: (config: VectorSearchQueryConfig) => void;
  /** Called when user wants to save */
  onSave?: (config: VectorSearchQueryConfig) => Promise<void>;
  /** Whether to show save button */
  showSaveButton?: boolean;
  /** Whether the panel is disabled */
  disabled?: boolean;
}

export function QueryConfigPanel({
  sourceId,
  metadata,
  config,
  onChange,
  onSave,
  showSaveButton = true,
  disabled = false,
}: QueryConfigPanelProps) {
  // Initialize with default config if none provided
  const [localConfig, setLocalConfig] = useState<VectorSearchQueryConfig>(() =>
    config || createDefaultQueryConfig(metadata)
  );
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Sync with external config changes
  useEffect(() => {
    if (config) {
      setLocalConfig(config);
      setIsDirty(false);
    }
  }, [config]);

  const handleChange = useCallback(
    (updates: Partial<VectorSearchQueryConfig>) => {
      const newConfig = { ...localConfig, ...updates };
      setLocalConfig(newConfig);
      setIsDirty(true);
      onChange(newConfig);
    },
    [localConfig, onChange]
  );

  const handleSave = useCallback(async () => {
    if (!onSave) return;

    setIsSaving(true);
    setSaveError(null);

    try {
      await onSave(localConfig);
      setIsDirty(false);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  }, [localConfig, onSave]);

  const handleFiltersChange = useCallback(
    (filters: FilterExpression[]) => {
      handleChange({ filters });
    },
    [handleChange]
  );

  // Determine which query types are supported
  const supportedQueryTypes = metadata.supported_query_types || ['ANN'];
  const allQueryTypes: QueryType[] = ['ANN', 'HYBRID', 'FULL_TEXT'];

  return (
    <div className="space-y-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
      {/* Query Type Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Query Type
        </label>
        <div className="space-y-2">
          {allQueryTypes.map((type) => {
            const isSupported = supportedQueryTypes.includes(type);
            const isSelected = localConfig.query_type === type;

            return (
              <label
                key={type}
                className={`flex items-start p-3 rounded-lg border cursor-pointer transition-colors ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
                } ${!isSupported || disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <input
                  type="radio"
                  name="query_type"
                  value={type}
                  checked={isSelected}
                  disabled={!isSupported || disabled}
                  onChange={() => handleChange({ query_type: type })}
                  className="mt-0.5 mr-3"
                />
                <div>
                  <div className="font-medium text-gray-900 dark:text-gray-100">
                    {getQueryTypeLabel(type)}
                    {!isSupported && (
                      <span className="ml-2 text-xs text-gray-500">(Not supported)</span>
                    )}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {getQueryTypeDescription(type)}
                  </div>
                </div>
              </label>
            );
          })}
        </div>
      </div>

      {/* Result Settings */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label
            htmlFor={`num_results_${sourceId}`}
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Number of Results
          </label>
          <input
            id={`num_results_${sourceId}`}
            type="number"
            min={1}
            max={localConfig.query_type === 'ANN' ? 100 : 200}
            value={localConfig.num_results}
            onChange={(e) => handleChange({ num_results: parseInt(e.target.value, 10) || 10 })}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                       bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
                       focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
          />
          <p className="mt-1 text-xs text-gray-500">
            {localConfig.query_type === 'ANN' ? '1-100' : '1-200 (hybrid/full-text limit)'}
          </p>
        </div>

        <div>
          <label
            htmlFor={`score_threshold_${sourceId}`}
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Score Threshold (optional)
          </label>
          <input
            id={`score_threshold_${sourceId}`}
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={localConfig.score_threshold ?? ''}
            onChange={(e) =>
              handleChange({
                score_threshold: e.target.value ? parseFloat(e.target.value) : undefined,
              })
            }
            disabled={disabled}
            placeholder="0.0 - 1.0"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                       bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
                       focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
          />
          <p className="mt-1 text-xs text-gray-500">Filter results below this similarity score</p>
        </div>
      </div>

      {/* Reranking Settings */}
      {metadata.supports_reranking && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Reranking
            </label>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={localConfig.enable_reranking}
                onChange={(e) => handleChange({ enable_reranking: e.target.checked })}
                disabled={disabled}
                className="sr-only peer"
              />
              <div
                className="w-11 h-6 bg-gray-200 peer-focus:ring-2 peer-focus:ring-blue-300
                            dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-600
                            peer-checked:after:translate-x-full peer-checked:after:border-white
                            after:content-[''] after:absolute after:top-[2px] after:left-[2px]
                            after:bg-white after:border-gray-300 after:border after:rounded-full
                            after:h-5 after:w-5 after:transition-all dark:border-gray-600
                            peer-checked:bg-blue-600"
              ></div>
            </label>
          </div>

          {localConfig.enable_reranking && (
            <div className="mt-2">
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
                Columns to rerank (text columns)
              </label>
              <div className="flex flex-wrap gap-2">
                {metadata.filter_columns
                  .filter((col) => col.data_type === 'string')
                  .map((col) => {
                    const isSelected = localConfig.columns_to_rerank?.includes(col.name);
                    return (
                      <button
                        key={col.name}
                        type="button"
                        onClick={() => {
                          const current = localConfig.columns_to_rerank || [];
                          const newCols = isSelected
                            ? current.filter((c) => c !== col.name)
                            : [...current, col.name];
                          handleChange({ columns_to_rerank: newCols });
                        }}
                        disabled={disabled}
                        className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                          isSelected
                            ? 'border-blue-500 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                            : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                        } disabled:opacity-50`}
                      >
                        {col.name}
                      </button>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Filter Builder */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Filters
        </label>
        <FilterBuilder
          columns={metadata.filter_columns}
          filters={localConfig.filters}
          onChange={handleFiltersChange}
          disabled={disabled}
        />
      </div>

      {/* Save Button */}
      {showSaveButton && onSave && (
        <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-600">
          {saveError && <p className="text-sm text-red-600">{saveError}</p>}
          <div className="flex items-center gap-3 ml-auto">
            {isDirty && (
              <span className="text-sm text-amber-600 dark:text-amber-400">Unsaved changes</span>
            )}
            <button
              type="button"
              onClick={handleSave}
              disabled={disabled || isSaving || !isDirty}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isSaving ? 'Saving...' : 'Save Configuration'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
