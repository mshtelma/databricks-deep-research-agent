/**
 * QuickAddSourceButton - Dropdown to add a new data source from chat.
 *
 * Features:
 * - Dropdown menu with source type options
 * - Opens SourceConfigModal for the selected type
 * - Refreshes discovery after successful creation
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { SourceConfigModal } from '@/components/sources/SourceConfigModal';
import {
  useCreateVectorSearchSource,
  useCreateGenieSource,
  useCreateKnowledgeAssistantSource,
  useValidateConnection,
} from '@/hooks/useDataSources';
import { useRefreshDiscovery } from '@/hooks/useDiscoveredSources';
import { cn } from '@/lib/utils';
import type {
  DataSourceType,
  CreateVectorSearchSourceRequest,
  CreateGenieSourceRequest,
  CreateKnowledgeAssistantSourceRequest,
  DataSourceValidationResult,
} from '@/types/dataSources';

type SourceFormData =
  | CreateVectorSearchSourceRequest
  | CreateGenieSourceRequest
  | CreateKnowledgeAssistantSourceRequest;

interface QuickAddSourceButtonProps {
  disabled?: boolean;
  className?: string;
}

export function QuickAddSourceButton({ disabled, className }: QuickAddSourceButtonProps) {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  const [showModal, setShowModal] = React.useState(false);
  const [sourceType, setSourceType] = React.useState<DataSourceType | null>(null);
  const menuRef = React.useRef<HTMLDivElement>(null);

  const createVS = useCreateVectorSearchSource();
  const createGenie = useCreateGenieSource();
  const createKA = useCreateKnowledgeAssistantSource();
  const validateConnection = useValidateConnection();
  const refreshDiscovery = useRefreshDiscovery();

  // Close menu on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    if (isMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isMenuOpen]);

  const handleAddSource = (type: DataSourceType) => {
    setSourceType(type);
    setShowModal(true);
    setIsMenuOpen(false);
  };

  const handleSave = async (data: SourceFormData, type: DataSourceType) => {
    switch (type) {
      case 'vector_search':
        await createVS.mutateAsync(data as CreateVectorSearchSourceRequest);
        break;
      case 'genie':
        await createGenie.mutateAsync(data as CreateGenieSourceRequest);
        break;
      case 'knowledge_assistant':
        await createKA.mutateAsync(data as CreateKnowledgeAssistantSourceRequest);
        break;
    }
    setShowModal(false);
    setSourceType(null);
    // Refresh discovery to show new source
    refreshDiscovery.mutate({});
  };

  const handleTest = async (
    data: SourceFormData,
    type: DataSourceType
  ): Promise<DataSourceValidationResult> => {
    // Build validation request based on type
    let request: { type: DataSourceType; endpoint_name?: string; index_name?: string; space_id?: string };

    switch (type) {
      case 'vector_search': {
        const vsData = data as CreateVectorSearchSourceRequest;
        request = {
          type,
          endpoint_name: vsData.endpoint_name,
          index_name: vsData.index_name,
        };
        break;
      }
      case 'genie': {
        const genieData = data as CreateGenieSourceRequest;
        request = {
          type,
          space_id: genieData.space_id,
        };
        break;
      }
      case 'knowledge_assistant': {
        const kaData = data as CreateKnowledgeAssistantSourceRequest;
        request = {
          type,
          endpoint_name: kaData.endpoint_name,
        };
        break;
      }
      default:
        return { isValid: false, message: 'Unknown source type' };
    }

    const result = await validateConnection.mutateAsync(request);
    return {
      isValid: result.has_access,
      message: result.has_access
        ? 'Connection successful! You have access to this resource.'
        : result.error_message || 'Connection failed. Please check your configuration.',
    };
  };

  const isSaving = createVS.isPending || createGenie.isPending || createKA.isPending;

  return (
    <div className={cn('relative', className)} ref={menuRef}>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        disabled={disabled}
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        className="h-7 w-7"
        title="Add data source"
      >
        <PlusIcon className="h-4 w-4" />
      </Button>

      {/* Dropdown Menu */}
      {isMenuOpen && (
        <div className="absolute bottom-full mb-1 right-0 z-50 w-56 rounded-md border bg-popover p-1 shadow-md animate-in fade-in-0 zoom-in-95">
          <div className="px-2 py-1.5 text-sm font-medium text-muted-foreground">
            Add Data Source
          </div>
          <div className="h-px bg-border my-1" />
          <button
            type="button"
            onClick={() => handleAddSource('vector_search')}
            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground"
          >
            <DatabaseIcon className="h-4 w-4 text-blue-600" />
            Vector Search Index
          </button>
          <button
            type="button"
            onClick={() => handleAddSource('genie')}
            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground"
          >
            <ChartIcon className="h-4 w-4 text-purple-600" />
            Genie Space
          </button>
          <button
            type="button"
            onClick={() => handleAddSource('knowledge_assistant')}
            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground"
          >
            <BotIcon className="h-4 w-4 text-emerald-600" />
            Serving Endpoint
          </button>
        </div>
      )}

      <SourceConfigModal
        isOpen={showModal}
        sourceType={sourceType}
        onClose={() => {
          setShowModal(false);
          setSourceType(null);
        }}
        onSave={handleSave}
        onTest={handleTest}
        isSaving={isSaving}
      />
    </div>
  );
}

// Icons
function PlusIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function ChartIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M3 3v18h18" />
      <path d="m19 9-5 5-4-4-3 3" />
    </svg>
  );
}

function BotIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M12 8V4H8" />
      <rect width="16" height="12" x="4" y="8" rx="2" />
    </svg>
  );
}

export default QuickAddSourceButton;
