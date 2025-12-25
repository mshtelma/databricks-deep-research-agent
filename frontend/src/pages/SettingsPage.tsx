import * as React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { preferencesApi } from '@/api/client';
import { SystemInstructionsEditor } from '@/components/settings/SystemInstructionsEditor';
import { cn } from '@/lib/utils';
import type { ResearchDepth } from '@/types';

const DEPTH_OPTIONS: { value: ResearchDepth; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'System chooses based on query complexity' },
  { value: 'light', label: 'Light', description: 'Quick search (1-3 steps)' },
  { value: 'medium', label: 'Medium', description: 'Balanced research (3-6 steps)' },
  { value: 'extended', label: 'Extended', description: 'Deep research (5-10 steps)' },
];

export function SettingsPage() {
  const queryClient = useQueryClient();

  // Fetch current preferences
  const { data: preferences, isLoading } = useQuery({
    queryKey: ['preferences'],
    queryFn: () => preferencesApi.get(),
  });

  // Local state for edits
  const [systemInstructions, setSystemInstructions] = React.useState('');
  const [defaultDepth, setDefaultDepth] = React.useState<ResearchDepth>('auto');
  const [hasInstructionChanges, setHasInstructionChanges] = React.useState(false);

  // Sync local state with fetched preferences
  React.useEffect(() => {
    if (preferences) {
      setSystemInstructions(preferences.system_instructions || '');
      setDefaultDepth(preferences.default_depth);
    }
  }, [preferences]);

  // Update preferences mutation
  const updateMutation = useMutation({
    mutationFn: (data: { system_instructions?: string; default_depth?: ResearchDepth }) =>
      preferencesApi.update(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['preferences'] });
      setHasInstructionChanges(false);
    },
  });

  const handleInstructionsChange = (value: string) => {
    setSystemInstructions(value);
    setHasInstructionChanges(value !== (preferences?.system_instructions || ''));
  };

  const handleSaveInstructions = () => {
    updateMutation.mutate({ system_instructions: systemInstructions });
  };

  const handleDepthChange = (depth: ResearchDepth) => {
    setDefaultDepth(depth);
    updateMutation.mutate({ default_depth: depth });
  };

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-muted-foreground">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-2xl mx-auto p-6 space-y-8">
        <div>
          <h1 className="text-2xl font-bold mb-2">Settings</h1>
          <p className="text-muted-foreground">
            Customize your research assistant experience.
          </p>
        </div>

        {/* System Instructions Section */}
        <section className="space-y-4">
          <div className="border-b pb-2">
            <h2 className="text-lg font-semibold">Research Customization</h2>
          </div>
          <SystemInstructionsEditor
            value={systemInstructions}
            onChange={handleInstructionsChange}
            onSave={handleSaveInstructions}
            isSaving={updateMutation.isPending}
            hasChanges={hasInstructionChanges}
          />
        </section>

        {/* Default Research Depth Section */}
        <section className="space-y-4">
          <div className="border-b pb-2">
            <h2 className="text-lg font-semibold">Default Research Depth</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            Set the default depth for new research queries. You can override this
            on a per-query basis.
          </p>
          <div className="space-y-2">
            {DEPTH_OPTIONS.map((option) => (
              <label
                key={option.value}
                className={cn(
                  'flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors',
                  defaultDepth === option.value
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-primary/50'
                )}
              >
                <input
                  type="radio"
                  name="default-depth"
                  value={option.value}
                  checked={defaultDepth === option.value}
                  onChange={() => handleDepthChange(option.value)}
                  disabled={updateMutation.isPending}
                  className="mt-0.5 accent-primary"
                />
                <div>
                  <span className="font-medium text-sm">{option.label}</span>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {option.description}
                  </p>
                </div>
              </label>
            ))}
          </div>
        </section>

        {/* Keyboard Shortcuts Section */}
        <section className="space-y-4">
          <div className="border-b pb-2">
            <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
          </div>
          <div className="rounded-lg border divide-y">
            <ShortcutRow
              shortcut={navigator.platform.includes('Mac') ? 'Cmd + N' : 'Ctrl + N'}
              description="Create new chat"
            />
            <ShortcutRow
              shortcut={navigator.platform.includes('Mac') ? 'Cmd + Enter' : 'Ctrl + Enter'}
              description="Send message"
            />
            <ShortcutRow
              shortcut={navigator.platform.includes('Mac') ? 'Cmd + K' : 'Ctrl + K'}
              description="Focus search"
            />
            <ShortcutRow
              shortcut="Escape"
              description="Close dialogs"
            />
          </div>
        </section>

        {/* About Section */}
        <section className="space-y-4">
          <div className="border-b pb-2">
            <h2 className="text-lg font-semibold">About</h2>
          </div>
          <div className="text-sm text-muted-foreground space-y-2">
            <p>
              <strong>Deep Research Agent</strong> - An AI-powered research
              assistant that performs multi-step research with real-time web
              search capabilities.
            </p>
            <p>
              Powered by Databricks and Claude. Research queries are executed
              through a 5-agent architecture: Coordinator, Planner, Researcher,
              Reflector, and Synthesizer.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

function ShortcutRow({ shortcut, description }: { shortcut: string; description: string }) {
  return (
    <div className="flex items-center justify-between px-4 py-3">
      <span className="text-sm text-muted-foreground">{description}</span>
      <kbd className="px-2 py-1 text-xs font-mono bg-muted rounded border">
        {shortcut}
      </kbd>
    </div>
  );
}
