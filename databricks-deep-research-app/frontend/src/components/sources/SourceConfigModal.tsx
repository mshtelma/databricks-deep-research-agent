/**
 * SourceConfigModal - Modal for configuring data sources.
 *
 * Supports:
 * - Vector Search source configuration
 * - Genie space configuration (T025)
 * - Knowledge Assistant configuration (T048)
 *
 * Features:
 * - Form validation
 * - Connection testing
 * - Dynamic example questions for Genie
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import type {
  DataSourceType,
  DataSource,
  CreateVectorSearchSourceRequest,
  CreateGenieSourceRequest,
  CreateKnowledgeAssistantSourceRequest,
  DataSourceValidationResult,
  VectorSearchConfig,
  GenieConfig,
  KnowledgeAssistantConfig,
} from '@/types/dataSources';

type SourceFormData =
  | CreateVectorSearchSourceRequest
  | CreateGenieSourceRequest
  | CreateKnowledgeAssistantSourceRequest;

interface SourceConfigModalProps {
  isOpen: boolean;
  sourceType: DataSourceType | null;
  /** Existing source for edit mode - if provided, form will pre-populate */
  existingSource?: DataSource;
  onClose: () => void;
  onSave: (data: SourceFormData, sourceType: DataSourceType) => Promise<void>;
  onTest?: (data: SourceFormData, sourceType: DataSourceType) => Promise<DataSourceValidationResult>;
  isSaving?: boolean;
}

export function SourceConfigModal({
  isOpen,
  sourceType,
  existingSource,
  onClose,
  onSave,
  onTest,
  isSaving = false,
}: SourceConfigModalProps) {
  const dialogRef = React.useRef<HTMLDivElement>(null);
  const [testResult, setTestResult] = React.useState<DataSourceValidationResult | null>(null);
  const [isTesting, setIsTesting] = React.useState(false);

  // Vector Search form state
  const [vsName, setVsName] = React.useState('');
  const [vsDescription, setVsDescription] = React.useState('');
  const [vsEndpointName, setVsEndpointName] = React.useState('');
  const [vsIndexName, setVsIndexName] = React.useState('');
  const [vsEnableReranking, setVsEnableReranking] = React.useState(false);
  const [vsQueryType, setVsQueryType] = React.useState<'ann' | 'hybrid'>('ann');

  // Genie form state
  const [genieName, setGenieName] = React.useState('');
  const [genieDescription, setGenieDescription] = React.useState('');
  const [genieSpaceId, setGenieSpaceId] = React.useState('');
  const [genieExampleQuestions, setGenieExampleQuestions] = React.useState<string[]>(['']);
  const [genieMaxRows, setGenieMaxRows] = React.useState(100);

  // Knowledge Assistant form state
  const [kaName, setKaName] = React.useState('');
  const [kaDescription, setKaDescription] = React.useState('');
  const [kaEndpointName, setKaEndpointName] = React.useState('');
  const [kaPassContext, setKaPassContext] = React.useState(true);

  // Reset form when modal opens/closes or source type changes
  // Pre-populate when editing an existing source
  React.useEffect(() => {
    if (isOpen) {
      setTestResult(null);

      if (existingSource) {
        // EDIT MODE: Pre-populate from existing source
        if (existingSource.type === 'vector_search') {
          const config = existingSource.config as VectorSearchConfig;
          setVsName(existingSource.name);
          setVsDescription(existingSource.description || '');
          setVsEndpointName(config.endpoint_name || '');
          setVsIndexName(config.index_name || '');
          setVsEnableReranking(config.enable_reranking || false);
          setVsQueryType(config.query_type || 'ann');
        } else if (existingSource.type === 'genie') {
          const config = existingSource.config as GenieConfig;
          setGenieName(existingSource.name);
          setGenieDescription(existingSource.description || '');
          setGenieSpaceId(config.space_id || '');
          setGenieExampleQuestions(config.example_questions?.length ? config.example_questions : ['']);
          setGenieMaxRows(config.max_rows || 100);
        } else if (existingSource.type === 'knowledge_assistant') {
          const config = existingSource.config as KnowledgeAssistantConfig;
          setKaName(existingSource.name);
          setKaDescription(existingSource.description || '');
          setKaEndpointName(config.endpoint_name || '');
          setKaPassContext(config.pass_context !== false);
        }
      } else {
        // CREATE MODE: Reset all forms
        setVsName('');
        setVsDescription('');
        setVsEndpointName('');
        setVsIndexName('');
        setVsEnableReranking(false);
        setVsQueryType('ann');
        setGenieName('');
        setGenieDescription('');
        setGenieSpaceId('');
        setGenieExampleQuestions(['']);
        setGenieMaxRows(100);
        setKaName('');
        setKaDescription('');
        setKaEndpointName('');
        setKaPassContext(true);
      }
    }
  }, [isOpen, sourceType, existingSource]);

  // Close on escape key
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Close on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dialogRef.current && !dialogRef.current.contains(e.target as Node) && isOpen) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  const getFormData = (): SourceFormData | null => {
    if (sourceType === 'vector_search') {
      if (!vsName.trim() || !vsEndpointName.trim() || !vsIndexName.trim()) return null;
      return {
        name: vsName.trim(),
        description: vsDescription.trim() || undefined,
        endpoint_name: vsEndpointName.trim(),
        index_name: vsIndexName.trim(),
        enable_reranking: vsEnableReranking,
        query_type: vsQueryType,
      };
    }

    if (sourceType === 'genie') {
      if (!genieName.trim() || !genieSpaceId.trim()) return null;
      const filteredQuestions = genieExampleQuestions.filter((q) => q.trim());
      return {
        name: genieName.trim(),
        description: genieDescription.trim() || undefined,
        space_id: genieSpaceId.trim(),
        example_questions: filteredQuestions.length > 0 ? filteredQuestions : undefined,
        max_rows: genieMaxRows,
      };
    }

    if (sourceType === 'knowledge_assistant') {
      if (!kaName.trim() || !kaEndpointName.trim()) return null;
      return {
        name: kaName.trim(),
        description: kaDescription.trim() || undefined,
        endpoint_name: kaEndpointName.trim(),
        pass_context: kaPassContext,
      };
    }

    return null;
  };

  const handleSave = async () => {
    const data = getFormData();
    if (!data || !sourceType) return;
    await onSave(data, sourceType);
  };

  const handleTest = async () => {
    const data = getFormData();
    if (!data || !sourceType || !onTest) return;

    setIsTesting(true);
    setTestResult(null);
    try {
      const result = await onTest(data, sourceType);
      setTestResult(result);
    } catch (error) {
      setTestResult({
        isValid: false,
        message: error instanceof Error ? error.message : 'Connection test failed',
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleAddExampleQuestion = () => {
    setGenieExampleQuestions([...genieExampleQuestions, '']);
  };

  const handleRemoveExampleQuestion = (index: number) => {
    setGenieExampleQuestions(genieExampleQuestions.filter((_, i) => i !== index));
  };

  const handleExampleQuestionChange = (index: number, value: string) => {
    const updated = [...genieExampleQuestions];
    updated[index] = value;
    setGenieExampleQuestions(updated);
  };

  const isFormValid = getFormData() !== null;

  if (!isOpen || !sourceType) return null;

  const getTitle = () => {
    const prefix = existingSource ? 'Edit' : 'Add';
    switch (sourceType) {
      case 'vector_search':
        return `${prefix} Vector Search Source`;
      case 'genie':
        return `${prefix} Genie Space`;
      case 'knowledge_assistant':
        return `${prefix} Knowledge Assistant`;
      default:
        return `${prefix} Data Source`;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />

      {/* Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-config-title"
        className={cn(
          'relative z-50 w-full max-w-lg rounded-lg bg-background p-6 shadow-lg max-h-[90vh] overflow-y-auto',
          'animate-in fade-in-0 zoom-in-95'
        )}
      >
        <h3 id="source-config-title" className="text-lg font-semibold mb-4">
          {getTitle()}
        </h3>

        {/* Vector Search Form */}
        {sourceType === 'vector_search' && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">Name *</label>
              <Input
                value={vsName}
                onChange={(e) => setVsName(e.target.value)}
                placeholder="My Vector Search Index"
                disabled={isSaving}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Description</label>
              <textarea
                value={vsDescription}
                onChange={(e) => setVsDescription(e.target.value)}
                placeholder="Optional description of the data in this index..."
                rows={2}
                disabled={isSaving}
                className={cn(
                  'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
                  'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                  'disabled:cursor-not-allowed disabled:opacity-50'
                )}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Endpoint Name *</label>
              <Input
                value={vsEndpointName}
                onChange={(e) => setVsEndpointName(e.target.value)}
                placeholder="my_vector_search_endpoint"
                disabled={isSaving}
              />
              <p className="text-xs text-muted-foreground mt-1">
                The Databricks Vector Search endpoint name
              </p>
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Index Name *</label>
              <Input
                value={vsIndexName}
                onChange={(e) => setVsIndexName(e.target.value)}
                placeholder="catalog.schema.index_name"
                disabled={isSaving}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Full three-level name: catalog.schema.index
              </p>
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={vsEnableReranking}
                  onChange={(e) => setVsEnableReranking(e.target.checked)}
                  disabled={isSaving}
                  className="rounded border-input"
                />
                Enable Reranking
              </label>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Query Type:</span>
                <select
                  value={vsQueryType}
                  onChange={(e) => setVsQueryType(e.target.value as 'ann' | 'hybrid')}
                  disabled={isSaving}
                  className="rounded-md border border-input bg-background px-2 py-1 text-sm"
                >
                  <option value="ann">ANN</option>
                  <option value="hybrid">Hybrid</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Genie Form (T025) */}
        {sourceType === 'genie' && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">Name *</label>
              <Input
                value={genieName}
                onChange={(e) => setGenieName(e.target.value)}
                placeholder="Sales Analytics Genie"
                disabled={isSaving}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Space ID *</label>
              <Input
                value={genieSpaceId}
                onChange={(e) => setGenieSpaceId(e.target.value)}
                placeholder="01EX4M8Y5H..."
                disabled={isSaving}
              />
              <p className="text-xs text-muted-foreground mt-1">
                The Genie space ID from your Databricks workspace
              </p>
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Description</label>
              <textarea
                value={genieDescription}
                onChange={(e) => setGenieDescription(e.target.value)}
                placeholder="Describe what data this Genie space contains and what questions it can answer..."
                rows={3}
                disabled={isSaving}
                className={cn(
                  'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
                  'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                  'disabled:cursor-not-allowed disabled:opacity-50'
                )}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Example Questions</label>
              <p className="text-xs text-muted-foreground mb-2">
                Add example questions to help users understand what this Genie can answer
              </p>
              <div className="space-y-2">
                {genieExampleQuestions.map((question, index) => (
                  <div key={index} className="flex gap-2">
                    <Input
                      value={question}
                      onChange={(e) => handleExampleQuestionChange(index, e.target.value)}
                      placeholder={`Example question ${index + 1}...`}
                      disabled={isSaving}
                      className="flex-1"
                    />
                    {genieExampleQuestions.length > 1 && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => handleRemoveExampleQuestion(index)}
                        disabled={isSaving}
                        className="text-muted-foreground hover:text-destructive"
                      >
                        <XIcon className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleAddExampleQuestion}
                disabled={isSaving}
                className="mt-2"
              >
                <PlusIcon className="h-4 w-4 mr-1" />
                Add Question
              </Button>
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Max Rows</label>
              <Input
                type="number"
                value={genieMaxRows}
                onChange={(e) => setGenieMaxRows(parseInt(e.target.value, 10) || 100)}
                min={1}
                max={1000}
                disabled={isSaving}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Maximum rows to return in results (default: 100)
              </p>
            </div>
          </div>
        )}

        {/* Knowledge Assistant Form (T048) */}
        {sourceType === 'knowledge_assistant' && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">Name *</label>
              <Input
                value={kaName}
                onChange={(e) => setKaName(e.target.value)}
                placeholder="Legal Document Assistant"
                disabled={isSaving}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Endpoint Name *</label>
              <Input
                value={kaEndpointName}
                onChange={(e) => setKaEndpointName(e.target.value)}
                placeholder="my_assistant_endpoint"
                disabled={isSaving}
              />
              <p className="text-xs text-muted-foreground mt-1">
                The Databricks serving endpoint name for the Knowledge Assistant
              </p>
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Description</label>
              <textarea
                value={kaDescription}
                onChange={(e) => setKaDescription(e.target.value)}
                placeholder="Describe what domain expertise this assistant has..."
                rows={3}
                disabled={isSaving}
                className={cn(
                  'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
                  'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                  'disabled:cursor-not-allowed disabled:opacity-50'
                )}
              />
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={kaPassContext}
                  onChange={(e) => setKaPassContext(e.target.checked)}
                  disabled={isSaving}
                  className="rounded border-input"
                />
                Include Research Context
              </label>
              <p className="text-xs text-muted-foreground mt-1 ml-5">
                Pass current research findings to the assistant for more relevant answers
              </p>
            </div>
          </div>
        )}

        {/* Test Result */}
        {testResult && (
          <div
            className={cn(
              'mt-4 p-3 rounded-md text-sm',
              testResult.isValid
                ? 'bg-green-50 text-green-800 dark:bg-green-950 dark:text-green-200'
                : 'bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-200'
            )}
          >
            <div className="flex items-center gap-2">
              {testResult.isValid ? (
                <CheckIcon className="h-4 w-4 text-green-600" />
              ) : (
                <XCircleIcon className="h-4 w-4 text-red-600" />
              )}
              <span className="font-medium">
                {testResult.isValid ? 'Connection successful' : 'Connection failed'}
              </span>
            </div>
            <p className="mt-1 text-xs">{testResult.message}</p>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-between mt-6">
          <div>
            {onTest && (
              <Button
                variant="outline"
                onClick={handleTest}
                disabled={!isFormValid || isTesting || isSaving}
                loading={isTesting}
              >
                {isTesting ? 'Testing...' : 'Test Connection'}
              </Button>
            )}
          </div>
          <div className="flex gap-3">
            <Button variant="outline" onClick={onClose} disabled={isSaving}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!isFormValid || isSaving} loading={isSaving}>
              {isSaving ? 'Saving...' : existingSource ? 'Update Source' : 'Save Source'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Icons
function PlusIcon({ className }: { className?: string }) {
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
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function XIcon({ className }: { className?: string }) {
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
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
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
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function XCircleIcon({ className }: { className?: string }) {
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
      <circle cx="12" cy="12" r="10" />
      <path d="m15 9-6 6M9 9l6 6" />
    </svg>
  );
}

export default SourceConfigModal;
