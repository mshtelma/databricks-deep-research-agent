/**
 * Settings Panel with LLM, MCP, and Shortcuts configuration.
 */

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { RefreshCw, Plus, Trash2 } from 'lucide-react';
import type {
  ModelInfo,
  MCPShortcut,
  GenieSpaceInfo,
  VectorIndexInfo,
  ExternalConnectionInfo,
  MCPEndpointType,
} from './types';
import { HIGHLIGHT_COLORS, RESERVED_KEYS } from './types';

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
  // Models
  models: ModelInfo[];
  modelsLoading: boolean;
  selectedModel: string | null;
  defaultModel: string | null;
  onSelectModel: (model: string) => void;
  onRefreshModels: () => void;
  // MCP Endpoints
  genieSpaces: GenieSpaceInfo[];
  vectorIndexes: VectorIndexInfo[];
  externalConnections: ExternalConnectionInfo[];
  mcpLoading: boolean;
  onRefreshMcp: () => void;
  // Shortcuts
  mcpShortcuts: MCPShortcut[];
  onUpdateShortcuts: (shortcuts: MCPShortcut[]) => void;
}

export function SettingsPanel({
  open,
  onClose,
  models,
  modelsLoading,
  selectedModel,
  defaultModel,
  onSelectModel,
  onRefreshModels,
  genieSpaces,
  vectorIndexes,
  externalConnections,
  mcpLoading,
  onRefreshMcp,
  mcpShortcuts,
  onUpdateShortcuts,
}: SettingsPanelProps) {
  const [localShortcuts, setLocalShortcuts] = useState<MCPShortcut[]>(mcpShortcuts);

  useEffect(() => {
    setLocalShortcuts(mcpShortcuts);
  }, [mcpShortcuts]);

  const handleSaveShortcuts = () => {
    onUpdateShortcuts(localShortcuts);
  };

  const handleAddShortcut = () => {
    // Find an available key
    const usedKeys = new Set([
      ...RESERVED_KEYS,
      ...localShortcuts.map((s) => s.key.toLowerCase()),
    ]);
    const availableKeys = 'ghijklmnopqrtuvwxyz'.split('').filter((k) => !usedKeys.has(k));
    
    if (availableKeys.length === 0) {
      alert('No available keys for new shortcuts');
      return;
    }

    const newShortcut: MCPShortcut = {
      key: availableKeys[0]!.toUpperCase(),
      type: 'MCP',
      endpointType: 'genie',
      endpointId: genieSpaces[0]?.id || '',
      endpointName: genieSpaces[0]?.name || 'Select endpoint',
      instruction: '',
      color: HIGHLIGHT_COLORS.genie,
      resultBehavior: 'insert_below',
    };

    setLocalShortcuts([...localShortcuts, newShortcut]);
  };

  const handleRemoveShortcut = (index: number) => {
    setLocalShortcuts(localShortcuts.filter((_, i) => i !== index));
  };

  const handleUpdateShortcut = (index: number, updates: Partial<MCPShortcut>) => {
    const updated = [...localShortcuts];
    updated[index] = { ...updated[index]!, ...updates };
    setLocalShortcuts(updated);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Configure LLM models, MCP endpoints, and keyboard shortcuts.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="llm" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="llm">LLM Model</TabsTrigger>
            <TabsTrigger value="mcp">MCP Endpoints</TabsTrigger>
            <TabsTrigger value="shortcuts">Shortcuts</TabsTrigger>
          </TabsList>

          {/* LLM Tab */}
          <TabsContent value="llm" className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Select Model</Label>
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefreshModels}
                disabled={modelsLoading}
              >
                <RefreshCw className={`h-4 w-4 ${modelsLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>

            {modelsLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : (
              <Select
                value={selectedModel || defaultModel || ''}
                onValueChange={onSelectModel}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      <div className="flex items-center gap-2">
                        <span>{model.display_name}</span>
                        <Badge
                          variant={model.status === 'READY' ? 'default' : 'secondary'}
                          className="text-xs"
                        >
                          {model.status}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </TabsContent>

          {/* MCP Tab */}
          <TabsContent value="mcp" className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Available Endpoints</Label>
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefreshMcp}
                disabled={mcpLoading}
              >
                <RefreshCw className={`h-4 w-4 ${mcpLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>

            <ScrollArea className="h-[300px] pr-4">
              {mcpLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Genie Spaces */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">Genie Spaces</h4>
                    {genieSpaces.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No Genie Spaces available</p>
                    ) : (
                      <div className="space-y-2">
                        {genieSpaces.map((space) => (
                          <div
                            key={space.id}
                            className="p-2 border rounded-md text-sm"
                          >
                            <div className="font-medium">{space.name}</div>
                            <div className="text-xs text-muted-foreground">
                              ID: {space.id}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Tables: {space.tables.join(', ')}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Vector Indexes */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">Vector Search Indexes</h4>
                    {vectorIndexes.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No indexes available</p>
                    ) : (
                      <div className="space-y-2">
                        {vectorIndexes.map((index) => (
                          <div
                            key={index.name}
                            className="p-2 border rounded-md text-sm"
                          >
                            <div className="font-medium">{index.name}</div>
                            <div className="text-xs text-muted-foreground">
                              {index.num_docs.toLocaleString()} documents
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* External Connections */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">External Connections</h4>
                    {externalConnections.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No connections available</p>
                    ) : (
                      <div className="space-y-2">
                        {externalConnections.map((conn) => (
                          <div
                            key={conn.name}
                            className="p-2 border rounded-md text-sm"
                          >
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{conn.name}</span>
                              <Badge variant="outline">{conn.type}</Badge>
                              <Badge
                                variant={conn.status === 'active' ? 'default' : 'secondary'}
                              >
                                {conn.status}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Shortcuts Tab */}
          <TabsContent value="shortcuts" className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>MCP Shortcuts</Label>
              <Button variant="outline" size="sm" onClick={handleAddShortcut}>
                <Plus className="h-4 w-4 mr-1" />
                Add
              </Button>
            </div>

            <ScrollArea className="h-[250px] pr-4">
              {localShortcuts.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No custom shortcuts configured
                </p>
              ) : (
                <div className="space-y-3">
                  {localShortcuts.map((shortcut, index) => (
                    <div
                      key={index}
                      className="p-3 border rounded-md space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Badge
                            style={{ backgroundColor: shortcut.color }}
                            className="text-white"
                          >
                            {shortcut.key}
                          </Badge>
                          <span className="text-sm">{shortcut.endpointName}</span>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveShortcut(index)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <Select
                          value={shortcut.endpointType}
                          onValueChange={(value: MCPEndpointType) => {
                            const colorMap: Record<MCPEndpointType, string> = {
                              genie: HIGHLIGHT_COLORS.genie,
                              vector: HIGHLIGHT_COLORS.vector,
                              external: HIGHLIGHT_COLORS.external,
                              knowledge_assistant: HIGHLIGHT_COLORS.knowledge_assistant,
                            };
                            handleUpdateShortcut(index, {
                              endpointType: value,
                              color: colorMap[value],
                            });
                          }}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="genie">Genie</SelectItem>
                            <SelectItem value="vector">Vector Search</SelectItem>
                            <SelectItem value="external">External</SelectItem>
                          </SelectContent>
                        </Select>

                        <Select
                          value={shortcut.endpointId}
                          onValueChange={(value) => {
                            let name = value;
                            if (shortcut.endpointType === 'genie') {
                              name = genieSpaces.find((s) => s.id === value)?.name || value;
                            } else if (shortcut.endpointType === 'vector') {
                              name = value;
                            } else {
                              name = externalConnections.find((c) => c.name === value)?.name || value;
                            }
                            handleUpdateShortcut(index, {
                              endpointId: value,
                              endpointName: name,
                            });
                          }}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue placeholder="Select endpoint" />
                          </SelectTrigger>
                          <SelectContent>
                            {shortcut.endpointType === 'genie' &&
                              genieSpaces.map((s) => (
                                <SelectItem key={s.id} value={s.id}>
                                  {s.name}
                                </SelectItem>
                              ))}
                            {shortcut.endpointType === 'vector' &&
                              vectorIndexes.map((i) => (
                                <SelectItem key={i.name} value={i.name}>
                                  {i.name}
                                </SelectItem>
                              ))}
                            {shortcut.endpointType === 'external' &&
                              externalConnections.map((c) => (
                                <SelectItem key={c.name} value={c.name}>
                                  {c.name}
                                </SelectItem>
                              ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>

            <Button onClick={handleSaveShortcuts} className="w-full">
              Save Shortcuts
            </Button>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
