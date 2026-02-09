/**
 * Header component with AIditor branding, model selector, and settings.
 */

import { Settings, RefreshCw, ArrowUpFromLine } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import type { ModelInfo } from './types';

interface HeaderProps {
  models: ModelInfo[];
  modelsLoading: boolean;
  selectedModel: string | null;
  defaultModel: string | null;
  onSelectModel: (model: string) => void;
  onRefreshModels: () => void;
  onOpenSettings: () => void;
  /** Callback to export current markdown back to chat */
  onExportToChat?: () => void;
}

export function Header({
  models,
  modelsLoading,
  selectedModel,
  defaultModel,
  onSelectModel,
  onRefreshModels,
  onOpenSettings,
  onExportToChat,
}: HeaderProps) {
  const currentModel = models.find((m) => m.name === selectedModel);

  return (
    <TooltipProvider>
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center px-4">
          {/* Logo */}
          <div className="flex items-center gap-2 mr-6">
            <h1 className="text-xl font-bold">
              <span className="text-primary">AI</span>
              <span>ditor</span>
            </h1>
            <span className="text-xs text-muted-foreground hidden sm:inline">
              AI-Assisted Markdown Editor
            </span>
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Model Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground hidden md:inline">Model:</span>
            <Select
              value={selectedModel || defaultModel || ''}
              onValueChange={onSelectModel}
              disabled={modelsLoading}
            >
              <SelectTrigger className="w-[220px] h-9">
                <SelectValue placeholder="Select model...">
                  {currentModel ? (
                    <div className="flex items-center gap-2">
                      <span className="truncate">{currentModel.display_name}</span>
                      <Badge
                        variant={currentModel.status === 'READY' ? 'default' : 'secondary'}
                        className="text-[10px] px-1 py-0"
                      >
                        {currentModel.status === 'READY' ? '✓' : '○'}
                      </Badge>
                    </div>
                  ) : (
                    'Select model...'
                  )}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.name} value={model.name}>
                    <div className="flex items-center gap-2 w-full">
                      <span className="flex-1">{model.display_name}</span>
                      <Badge
                        variant={model.status === 'READY' ? 'default' : 'secondary'}
                        className="text-[10px] px-1.5"
                      >
                        {model.status}
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onRefreshModels}
                  disabled={modelsLoading}
                  className="h-9 w-9"
                >
                  <RefreshCw className={`h-4 w-4 ${modelsLoading ? 'animate-spin' : ''}`} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Refresh models</TooltipContent>
            </Tooltip>
          </div>

          {/* Export to Chat */}
          {onExportToChat && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onExportToChat}
                  className="h-9 ml-2 gap-1.5 text-xs"
                >
                  <ArrowUpFromLine className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Export to Chat</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Copy markdown and switch to Chat tab</TooltipContent>
            </Tooltip>
          )}

          {/* Settings */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={onOpenSettings}
                className="h-9 w-9"
              >
                <Settings className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Settings</TooltipContent>
          </Tooltip>
        </div>
      </header>
    </TooltipProvider>
  );
}
