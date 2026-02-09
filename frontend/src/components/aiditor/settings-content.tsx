/**
 * Settings Content - displayed as a tab instead of a dialog.
 */

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { RefreshCw, Plus, Trash2, RotateCcw } from 'lucide-react';
import type {
  BuiltinShortcut,
  ModelInfo,
  MCPShortcut,
  GenieSpaceInfo,
  VectorIndexInfo,
  ExternalConnectionInfo,
  KnowledgeAssistantInfo,
  CustomShortcutEndpointType,
} from './types';
import { DEFAULT_BUILTIN_SHORTCUTS, CUSTOM_SHORTCUT_COLORS, RESERVED_KEYS } from './types';

interface SettingsContentProps {
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
  knowledgeAssistants: KnowledgeAssistantInfo[];
  mcpLoading: boolean;
  onRefreshMcp: () => void;
  // Shortcuts
  mcpShortcuts: MCPShortcut[];
  onUpdateShortcuts: (shortcuts: MCPShortcut[]) => void;
  builtinShortcuts: BuiltinShortcut[];
  onUpdateBuiltinShortcuts: (shortcuts: BuiltinShortcut[]) => void;
}

export function SettingsContent({
  models,
  modelsLoading,
  selectedModel,
  onSelectModel,
  onRefreshModels,
  genieSpaces,
  vectorIndexes,
  externalConnections,
  knowledgeAssistants,
  mcpLoading,
  onRefreshMcp,
  mcpShortcuts,
  onUpdateShortcuts,
  builtinShortcuts,
  onUpdateBuiltinShortcuts,
}: SettingsContentProps) {
  const [localShortcuts, setLocalShortcuts] = useState<MCPShortcut[]>(mcpShortcuts);
  const [localBuiltins, setLocalBuiltins] = useState<BuiltinShortcut[]>(builtinShortcuts);
  const [listeningId, setListeningId] = useState<string | null>(null);
  const [keyCaptureError, setKeyCaptureError] = useState<string | null>(null);

  useEffect(() => {
    setLocalShortcuts(mcpShortcuts);
  }, [mcpShortcuts]);

  useEffect(() => {
    setLocalBuiltins(builtinShortcuts);
  }, [builtinShortcuts]);

  const handleSaveShortcuts = () => {
    onUpdateShortcuts(localShortcuts);
  };

  const handleSaveBuiltinShortcuts = () => {
    onUpdateBuiltinShortcuts(localBuiltins);
  };

  const normalizeShortcut = (shortcut: string) =>
    shortcut
      .toLowerCase()
      .replace('⌘', 'cmd+')
      .replace('ctrl', 'ctrl+')
      .replace('control', 'ctrl+')
      .replace('option', 'alt+')
      .replace('alt', 'alt+')
      .replace(/\s+/g, '');

  const formatShortcut = (event: KeyboardEvent) => {
    const parts: string[] = [];
    if (event.metaKey) parts.push('⌘');
    if (event.ctrlKey && !event.metaKey) parts.push('Ctrl');
    if (event.altKey) parts.push('Alt');
    if (event.shiftKey) parts.push('Shift');

    let key = event.key;
    if (key === 'Escape') key = 'Esc';
    if (key.length === 1) key = key.toUpperCase();
    return `${parts.join(parts.length && key ? '+' : '')}${key}`;
  };

  // Unified key capture effect for both builtin and custom shortcuts
  useEffect(() => {
    if (!listeningId) return;

    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();

      const key = event.key;
      if (['Shift', 'Control', 'Meta', 'Alt'].includes(key)) {
        return;
      }

      const nextShortcut = formatShortcut(event);
      const normalizedNext = normalizeShortcut(nextShortcut);

      // Check for conflicts with builtin shortcuts (excluding the one being edited)
      const conflictsWithBuiltins = localBuiltins.some(
        (shortcut) =>
          shortcut.id !== listeningId &&
          normalizeShortcut(shortcut.key) === normalizedNext
      );
      // Check for conflicts with custom shortcuts (excluding the one being edited)
      const conflictsWithCustom = localShortcuts.some(
        (shortcut, idx) =>
          `custom-${idx}` !== listeningId &&
          normalizeShortcut(shortcut.key) === normalizedNext
      );

      if (conflictsWithBuiltins || conflictsWithCustom) {
        setKeyCaptureError('Shortcut already in use');
        return;
      }

      // Check if this is a builtin or custom shortcut being edited
      const isBuiltin = localBuiltins.some((s) => s.id === listeningId);
      if (isBuiltin) {
        setLocalBuiltins((current) =>
          current.map((shortcut) =>
            shortcut.id === listeningId ? { ...shortcut, key: nextShortcut } : shortcut
          )
        );
      } else if (listeningId.startsWith('custom-')) {
        const index = parseInt(listeningId.replace('custom-', ''), 10);
        if (!isNaN(index)) {
          setLocalShortcuts((current) =>
            current.map((shortcut, i) =>
              i === index ? { ...shortcut, key: nextShortcut } : shortcut
            )
          );
        }
      }

      setKeyCaptureError(null);
      setListeningId(null);
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [listeningId, localBuiltins, localShortcuts]);

  const handleAddShortcut = () => {
    const usedKeys = new Set([
      ...RESERVED_KEYS,
      ...localShortcuts.map((s) => s.key.toLowerCase()),
      ...localBuiltins.map((s) => normalizeShortcut(s.key)),
    ]);
    const availableKeys = 'ghijklmnopqrtuvwxyz1234567890'.split('').filter((k) => !usedKeys.has(k));

    if (availableKeys.length === 0) {
      alert('No available keys for new shortcuts');
      return;
    }

    const newShortcut: MCPShortcut = {
      key: availableKeys[0]!.toUpperCase(),
      type: 'LLM',
      endpointType: 'llm',
      endpointId: '',
      endpointName: 'LLM Custom',
      instruction: '',
      color: CUSTOM_SHORTCUT_COLORS.llm,
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
    <div className="space-y-6">
      {/* Keyboard Shortcuts Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Keyboard Shortcuts</CardTitle>
              <CardDescription>
                Click on a shortcut to reassign it. Press the new key combination when prompted.
              </CardDescription>
              {keyCaptureError && (
                <div className="mt-2 text-xs text-destructive">{keyCaptureError}</div>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              className="gap-1"
              onClick={() => setLocalBuiltins(DEFAULT_BUILTIN_SHORTCUTS)}
            >
              <RotateCcw className="h-4 w-4" />
              Reset to Defaults
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {localBuiltins.map((shortcut) => (
              <div
                key={shortcut.id}
                className="flex items-center justify-between py-3 px-4 rounded-lg bg-muted/50 hover:bg-muted transition-colors cursor-pointer"
                onClick={() => {
                  setListeningId(shortcut.id);
                  setKeyCaptureError(null);
                }}
              >
                <div>
                  <div className="font-medium text-sm">{shortcut.action}</div>
                  <div className="text-xs text-muted-foreground">{shortcut.description}</div>
                </div>
                <kbd className="px-2.5 py-1.5 bg-background border rounded-md text-sm font-mono shadow-sm">
                  {listeningId === shortcut.id ? 'Press keys…' : shortcut.key}
                </kbd>
              </div>
            ))}
          </div>
          <Button onClick={handleSaveBuiltinShortcuts} className="w-full mt-4">
            Save Changes
          </Button>
        </CardContent>
      </Card>

      {/* Custom Shortcuts Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Custom Shortcuts</CardTitle>
              <CardDescription>
                Create custom shortcuts that call the LLM with a preset instruction, or query an MCP endpoint.
                Click a key badge to remap it.
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={handleAddShortcut} className="gap-1">
              <Plus className="h-4 w-4" />
              Add Custom Shortcut
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {localShortcuts.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No custom shortcuts configured. Click &quot;Add Custom Shortcut&quot; to create one.
            </p>
          ) : (
            <div className="space-y-3">
              {localShortcuts.map((shortcut, index) => (
                <div
                  key={index}
                  className="p-4 rounded-lg border bg-background space-y-3"
                >
                  <div className="flex items-center gap-4">
                    {/* Clickable Key Badge for remapping (Issue 6) */}
                    <kbd
                      className="min-w-[40px] px-3 py-2 rounded-md text-center font-mono font-bold text-white cursor-pointer hover:opacity-80 transition-opacity"
                      style={{ backgroundColor: shortcut.color }}
                      onClick={() => {
                        setListeningId(`custom-${index}`);
                        setKeyCaptureError(null);
                      }}
                      title="Click to remap key"
                    >
                      {listeningId === `custom-${index}` ? '...' : shortcut.key}
                    </kbd>

                    {/* Shortcut Type Select (LLM or MCP endpoints) */}
                    <Select
                      value={shortcut.endpointType}
                      onValueChange={(value: CustomShortcutEndpointType) => {
                        const updates: Partial<MCPShortcut> = {
                          endpointType: value,
                          color: CUSTOM_SHORTCUT_COLORS[value],
                        };
                        if (value === 'llm') {
                          updates.type = 'LLM';
                          updates.endpointId = '';
                          updates.endpointName = 'LLM Custom';
                        } else {
                          updates.type = 'MCP';
                          updates.endpointId = '';
                          updates.endpointName = 'Select endpoint';
                        }
                        handleUpdateShortcut(index, updates);
                      }}
                    >
                      <SelectTrigger className="w-[130px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="llm">LLM (Custom)</SelectItem>
                        <SelectItem value="genie">Genie Space</SelectItem>
                        <SelectItem value="vector">Vector Search</SelectItem>
                        <SelectItem value="external">External</SelectItem>
                        <SelectItem value="knowledge_assistant">Knowledge Assistant</SelectItem>
                      </SelectContent>
                    </Select>

                    {/* Endpoint Select (only for MCP types) */}
                    {shortcut.endpointType !== 'llm' && (
                      <Select
                        value={shortcut.endpointId}
                        onValueChange={(value) => {
                          let name = value;
                          if (shortcut.endpointType === 'genie') {
                            name = genieSpaces.find((s) => s.id === value)?.name || value;
                          } else if (shortcut.endpointType === 'vector') {
                            name = value;
                          } else if (shortcut.endpointType === 'knowledge_assistant') {
                            name = knowledgeAssistants.find((ka) => ka.tile_id === value)?.name || value;
                          } else {
                            name = externalConnections.find((c) => c.name === value)?.name || value;
                          }
                          handleUpdateShortcut(index, {
                            endpointId: value,
                            endpointName: name,
                          });
                        }}
                      >
                        <SelectTrigger className="w-[180px]">
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
                          {shortcut.endpointType === 'knowledge_assistant' &&
                            knowledgeAssistants.map((ka) => (
                              <SelectItem key={ka.tile_id} value={ka.tile_id}>
                                {ka.name}
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    )}

                    {/* Spacer */}
                    <div className="flex-1" />

                    {/* Remove Button */}
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleRemoveShortcut(index)}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* Instruction Input (full width row) */}
                  <Input
                    placeholder={shortcut.endpointType === 'llm'
                      ? 'Custom instruction for the LLM (e.g., "Make more formal", "Add examples")...'
                      : 'Query template for this endpoint (use {{text}} for selected text)...'
                    }
                    value={shortcut.instruction || ''}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      handleUpdateShortcut(index, { instruction: e.target.value })
                    }
                    className="w-full"
                  />
                </div>
              ))}

              <Button onClick={handleSaveShortcuts} className="w-full mt-4">
                Save Changes
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* MCP Endpoints Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">MCP Endpoints</CardTitle>
              <CardDescription>
                Available Genie Spaces, Vector Search indexes, and external connections.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={onRefreshMcp}
              disabled={mcpLoading}
              className="gap-1"
            >
              <RefreshCw className={`h-4 w-4 ${mcpLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {/* Genie Spaces */}
            <div>
              <h4 className="text-sm font-medium mb-2">Genie Spaces</h4>
              {genieSpaces.length === 0 ? (
                <p className="text-xs text-muted-foreground">None available</p>
              ) : (
                <div className="space-y-1">
                  {genieSpaces.map((space) => (
                    <div key={space.id} className="text-xs p-2 rounded bg-muted/50">
                      {space.name}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Vector Indexes */}
            <div>
              <h4 className="text-sm font-medium mb-2">Vector Indexes</h4>
              {vectorIndexes.length === 0 ? (
                <p className="text-xs text-muted-foreground">None available</p>
              ) : (
                <div className="space-y-1">
                  {vectorIndexes.map((idx) => (
                    <div key={idx.name} className="text-xs p-2 rounded bg-muted/50">
                      {idx.name}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* External Connections */}
            <div>
              <h4 className="text-sm font-medium mb-2">External</h4>
              {externalConnections.length === 0 ? (
                <p className="text-xs text-muted-foreground">None available</p>
              ) : (
                <div className="space-y-1">
                  {externalConnections.map((conn) => (
                    <div key={conn.name} className="text-xs p-2 rounded bg-muted/50 flex items-center gap-1">
                      {conn.name}
                      <Badge variant="outline" className="text-[10px] px-1">
                        {conn.type}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Selection Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">LLM Model</CardTitle>
              <CardDescription>
                Select the model to use for processing edits.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={onRefreshModels}
              disabled={modelsLoading}
              className="gap-1"
            >
              <RefreshCw className={`h-4 w-4 ${modelsLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[200px]">
            <div className="space-y-2">
              {models.map((model) => (
                <div
                  key={model.name}
                  onClick={() => onSelectModel(model.name)}
                  className={`
                    flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors
                    ${selectedModel === model.name 
                      ? 'bg-primary/10 border border-primary' 
                      : 'bg-muted/50 hover:bg-muted border border-transparent'}
                  `}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        selectedModel === model.name ? 'bg-primary' : 'bg-muted-foreground/30'
                      }`}
                    />
                    <div>
                      <div className="font-medium text-sm">{model.display_name}</div>
                      <div className="text-xs text-muted-foreground">{model.task}</div>
                    </div>
                  </div>
                  <Badge variant={model.status === 'READY' ? 'default' : 'secondary'}>
                    {model.status}
                  </Badge>
                </div>
              ))}
            </div>
          </ScrollArea>

          {selectedModel && (
            <div className="mt-4 p-3 rounded-lg bg-muted/50">
              <div className="text-xs text-muted-foreground">Selected:</div>
              <div className="font-medium">{models.find(m => m.name === selectedModel)?.display_name}</div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
