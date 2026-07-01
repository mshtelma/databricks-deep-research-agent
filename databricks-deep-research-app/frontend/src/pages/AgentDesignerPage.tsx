/**
 * AgentDesignerPage — main designer route.
 *
 * Route: /designer/:id
 *   id === 'new'  → new-agent creation flow
 *   id !== 'new'  → edit-agent flow (loads via getAgentV2WithEtag)
 *
 * Layout (Databricks Agentic Designer):
 *   AppShell sidebar | TopBar + Canvas (centered max-720px) | Inspector | DesignerChat
 *
 * The dedicated left ToolsPanel from the original layout has been removed —
 * its workflow-tools role is now absorbed by the Inspector's no-selection
 * "Workspace tools" view (per the design's final iteration).
 */

import * as React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Check,
  Zap,
  Save,
  History as HistoryIcon,
  Play,
  ChevronLeft,
  Hash as HashIcon,
} from 'lucide-react';

import {
  getAgentV2WithEtag,
  getAgentV2,
  createAgentV2,
  updateAgentV2,
  listRevisions,
  getRevision,
  EtagConflictError,
  parseAgentCriticError,
  type WorkflowValidationResult,
  type AgentV2Response,
} from '@/api/agentsV2';
import { getRegistry, validateWorkflow, exportYamlFromDefinition } from '@/api/agentDesigner';
import { useAgentEditorStore } from '@/stores/agentEditorStore';
import { buildDesignerSavePayload } from '@/lib/agentDesignerSave';
import { slugifyFilename } from '@/lib/download';
import { AppShell } from '@/components/layout/AppShell';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { BlockEditor } from '@/components/agentDesigner/BlockEditor';
import { ConfigPanel } from '@/components/agentDesigner/ConfigPanel';
import { EtagConflictModal } from '@/components/agentDesigner/EtagConflictModal';
import {
  DeployDropdown,
  DeploymentsSection,
  StatusPanel,
} from '@/components/agentDesigner/deploy';
import { RevisionList } from '@/components/agentDesigner/RevisionList';
import { RevisionPreview } from '@/components/agentDesigner/RevisionPreview';
import { ExportYamlMenu } from '@/components/agentDesigner/ExportYamlMenu';
import type { DeploymentResponse } from '@/types/deployment';
import * as clientMetrics from '@/lib/clientMetrics';
import { createDraftWorkflow, isWorkflowEmpty } from '@/lib/workflowAst';
import type { RegistryResponse } from '@/types/agentDesigner';
import {
  deploymentIdentityMatches,
  summarizeRootChildren,
} from '@/components/agentDesigner/deploy/revisionProvenance';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * localStorage key consumed by `MessageInput` (chat composer) to pre-select
 * an agent for the next research session. Kept in sync with
 * `databricks-deep-research-app/frontend/src/components/chat/MessageInput.tsx`.
 */
const SELECTED_AGENT_KEY = 'deep-research-selected-agent';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatRelativeShort(input?: string | null): string {
  if (!input) return '—';
  const date = new Date(input);
  if (Number.isNaN(date.getTime())) return '—';
  const diff = Date.now() - date.getTime();
  const min = Math.round(diff / 60_000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  if (day === 1) return 'Yesterday';
  if (day < 7) return `${day}d ago`;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// ---------------------------------------------------------------------------
// DesignerInner
// ---------------------------------------------------------------------------

interface DesignerInnerProps {
  id: string;
  registry: RegistryResponse;
}

function DesignerInner({ id, registry }: DesignerInnerProps): React.ReactElement {
  const navigate = useNavigate();
  const isNew = id === 'new';

  // Store selectors
  const ast = useAgentEditorStore((s) => s.ast);
  const etag = useAgentEditorStore((s) => s.etag);
  const isDirty = useAgentEditorStore((s) => s.isDirty);
  const validationErrors = useAgentEditorStore((s) => s.validationErrors);
  const pendingValidationAgentId = useAgentEditorStore((s) => s.pendingValidationAgentId);
  const { load, markClean, markValidationErrors, setPendingValidationAgentId } =
    useAgentEditorStore.getState();

  // Local state for new-agent name/description
  const [localName, setLocalName] = React.useState('');
  const [localDescription, setLocalDescription] = React.useState('');

  // ETag conflict modal state
  const [conflictModalOpen, setConflictModalOpen] = React.useState(false);
  const [conflictEtag, setConflictEtag] = React.useState<string | null>(null);
  // W15: populated on conflict so the merge UI in EtagConflictModal has the
  // server-side AST to diff against. Cleared on close.
  const [conflictServerAst, setConflictServerAst] = React.useState<
    import('@/types/ast').AST | null
  >(null);

  // Tab state: 'edit' | 'revisions' | 'settings' | 'deployments'
  const [activeTab, setActiveTab] = React.useState<
    'edit' | 'revisions' | 'settings' | 'deployments'
  >('edit');
  const [selectedRevId, setSelectedRevId] = React.useState<string | null>(null);

  // Save-flash state for the top bar status pill
  const [savedFlash, setSavedFlash] = React.useState(false);
  const [deploySyncError, setDeploySyncError] = React.useState<string | null>(null);
  // Non-blocking save notice: shown after a successful save with advisory
  // critic verdict, or after a save failure (non-conflict errors).
  const [saveNotice, setSaveNotice] = React.useState<{
    kind: 'warning' | 'error';
    message: string;
    /** Full validation result — present for advisory/warning saves. */
    validation?: WorkflowValidationResult | null;
  } | null>(null);
  const queryClient = useQueryClient();

  // Surface a (possibly background) save-time validation verdict in the save
  // banner. Reused for the inline (cache-hit) verdict and the polled result.
  const applyValidationNotice = React.useCallback(
    (validation: WorkflowValidationResult | null): void => {
      if (
        validation &&
        validation.verdict !== 'pass' &&
        validation.verdict !== 'skipped'
      ) {
        const summary =
          typeof validation.summary === 'string' ? validation.summary : '';
        setSaveNotice({
          kind: 'warning',
          message: `Saved with advisory notice (${validation.verdict})${summary ? `: ${summary}` : ''}`,
          validation,
        });
      } else {
        setSaveNotice(null);
        setSavedFlash(true);
        window.setTimeout(() => setSavedFlash(false), 1000);
      }
    },
    [],
  );

  // The advisory validator runs in the BACKGROUND on a cache miss so the save
  // never blocks on the slow critic (the cause of "Request timed out after
  // 30000ms"). Poll GET /agents-v2/{id} until the verdict lands, then surface
  // it. Bounded + never silent: on exhaustion we say validation is still
  // running. Fire-and-forget (guarded by a ref) so it is not cancelled by the
  // store-flag clear that re-runs the effect below.
  const validationPollRef = React.useRef<string | null>(null);
  const startValidationPoll = React.useCallback(
    (agentId: string): void => {
      if (validationPollRef.current === agentId) return;
      validationPollRef.current = agentId;
      setSaveNotice({ kind: 'warning', message: 'Saved ✓ · validating workflow…' });
      void (async () => {
        try {
          for (let i = 0; i < 40; i++) {
            await new Promise((resolve) => window.setTimeout(resolve, 3000));
            let agent: AgentV2Response | null = null;
            try {
              agent = await getAgentV2(agentId);
            } catch {
              continue; // transient error — keep polling
            }
            if (!agent.validation_pending) {
              applyValidationNotice(agent.validation ?? null);
              return;
            }
          }
          setSaveNotice({
            kind: 'warning',
            message:
              'Saved ✓ — validation is still running. Ask the designer chat to re-check it.',
          });
        } finally {
          if (validationPollRef.current === agentId) validationPollRef.current = null;
        }
      })();
    },
    [applyValidationNotice],
  );

  // New-agent saves navigate to /designer/{id}; the pending-validation signal
  // is carried across that remount via the editor store and consumed here.
  React.useEffect(() => {
    if (pendingValidationAgentId && pendingValidationAgentId === id) {
      setPendingValidationAgentId(null);
      startValidationPoll(pendingValidationAgentId);
    }
  }, [pendingValidationAgentId, id, setPendingValidationAgentId, startValidationPoll]);

  React.useEffect(() => {
    if (activeTab === 'revisions') {
      clientMetrics.emit('revisions_tab_opened', undefined, { agent_id: id });
    }
  }, [activeTab, id]);

  // -------------------------------------------------------------------------
  // Initialize new-agent flow
  // -------------------------------------------------------------------------

  const initializedRef = React.useRef(false);
  React.useEffect(() => {
    if (isNew && !initializedRef.current) {
      initializedRef.current = true;
      useAgentEditorStore.setState({
        ast: createDraftWorkflow(),
        agentId: null,
        etag: null,
        isDirty: false,
        validationErrors: [],
        selectedPath: null,
      });
    }
  }, [isNew]);

  // -------------------------------------------------------------------------
  // Edit flow: fetch agent
  // -------------------------------------------------------------------------

  const agentQuery = useQuery({
    queryKey: ['agents-v2', id, 'with-etag'],
    queryFn: () => getAgentV2WithEtag(id),
    enabled: !isNew,
    staleTime: Infinity,
  });

  React.useEffect(() => {
    if (agentQuery.data) {
      load({ agent: agentQuery.data.agent, etag: agentQuery.data.etag });
    }
  }, [agentQuery.data, load]);

  // -------------------------------------------------------------------------
  // Derive name/description for header display
  // -------------------------------------------------------------------------

  const agentName = isNew ? localName : (agentQuery.data?.agent.name ?? '');
  const agentDescription = isNew ? localDescription : (agentQuery.data?.agent.description ?? '');
  const agentIdFull = agentQuery.data?.agent.id ?? '';
  const agentUpdatedAt = agentQuery.data?.agent.updated_at ?? null;
  const agentIdShort = !isNew && agentQuery.data?.agent.id
    ? agentQuery.data.agent.id.slice(0, 16)
    : 'new-draft';

  // Export the LIVE canvas (including unsaved edits and brand-new agents) as
  // YAML. Uses the SAME buildDesignerSavePayload transform as Save, so the
  // exported document matches what would be persisted for an equal canvas.
  const exportFilename = slugifyFilename(agentName || 'agent', 'yaml');
  const handleExportYaml = React.useCallback((): Promise<string> => {
    if (!ast) return Promise.reject(new Error('No workflow canvas to export'));
    const { definition } = buildDesignerSavePayload(ast, {
      isNew,
      localName,
      agentName,
      localDescription,
      agentDescription,
    });
    return exportYamlFromDefinition(definition);
  }, [ast, isNew, localName, agentName, localDescription, agentDescription]);

  // -------------------------------------------------------------------------
  // Save mutation
  // -------------------------------------------------------------------------

  // One-shot "Save as draft anyway" intent: the coverage gate is force-overridable
  // (the heuristic can false-positive), structural errors are not. Threaded via a ref
  // so the existing no-arg mutate()/mutateAsync() callers stay unchanged.
  const forceSaveRef = React.useRef(false);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const force = forceSaveRef.current;
      forceSaveRef.current = false; // consume
      if (!ast) throw new Error('No AST to save');
      if (isWorkflowEmpty(ast)) {
        markValidationErrors([
          {
            message: 'Add at least one block before saving this workflow.',
            path: 'root.children',
            line: null,
            kind: 'validation',
          },
        ]);
        return null;
      }

      const payload = buildDesignerSavePayload(ast, {
        isNew,
        agentName,
        localName,
        agentDescription,
        localDescription,
      });
      const { definition } = payload;

      const result = await validateWorkflow(definition as unknown as Record<string, unknown>);
      if (!result.valid) {
        const errs = result.errors as import('@/types/ast').ValidationError[];
        // Coverage-only failures are force-overridable ("Save as draft anyway");
        // any structural error is a hard block regardless of force.
        const onlyCoverage =
          errs.length > 0 && errs.every((e) => e.kind === 'coverage');
        if (!(onlyCoverage && force)) {
          markValidationErrors(errs);
          return null;
        }
      }

      markValidationErrors([]);

      if (isNew) {
        const { agent, etag: newEtag } = await createAgentV2(
          {
            name: payload.name,
            description: payload.description,
            definition: definition as unknown as Record<string, unknown>,
          },
          { force },
        );
        markClean(newEtag);
        return { agent, etag: newEtag, isNew: true };
      } else {
        const currentEtag = etag ?? '';
        const { agent, etag: newEtag } = await updateAgentV2(
          id,
          {
            name: payload.name,
            description: payload.description,
            definition: definition as unknown as Record<string, unknown>,
          },
          currentEtag,
          { force },
        );
        markClean(newEtag);
        return { agent, etag: newEtag, isNew: false };
      }
    },
    onSuccess: (data) => {
      if (!data) return;
      void queryClient.invalidateQueries({ queryKey: ['agents-v2'] });
      if (data.isNew) {
        // The save navigates to /designer/{id}; carry any pending-validation
        // signal across the remount so the new page polls for the verdict.
        if (data.agent.validation_pending) {
          setPendingValidationAgentId(data.agent.id);
        }
        void navigate(`/designer/${data.agent.id}`);
        return;
      }
      if (data.agent.validation_pending) {
        // Advisory validation is running in the background — poll for the
        // verdict (the save itself returned instantly, never timing out).
        startValidationPoll(data.agent.id);
      } else {
        // Verdict already known (cache hit) or not applicable.
        applyValidationNotice(data.agent.validation ?? null);
      }
    },
    onError: (error) => {
      if (error instanceof EtagConflictError) {
        setConflictEtag(error.current_etag);
        setConflictModalOpen(true);
        return;
      }
      const criticError = parseAgentCriticError(error);
      if (criticError) {
        const summary =
          typeof criticError.critique?.summary === 'string'
            ? criticError.critique.summary
            : criticError.message;
        setSaveNotice({ kind: 'error', message: `Save blocked by critic: ${summary}` });
        return;
      }
      const message =
        error instanceof Error ? error.message : 'Save failed. Please try again.';
      setSaveNotice({ kind: 'error', message });
    },
  });

  function handleSave(): void {
    saveMutation.mutate();
  }

  // -------------------------------------------------------------------------
  // Test run — bind agent for the next chat session and navigate to /chat.
  // The MessageInput composer reads SELECTED_AGENT_KEY and forwards it as
  // `agentId` on the QuerySubmission, which the framework orchestrator
  // resolves via _resolve_agent_v2_workflow.
  // -------------------------------------------------------------------------

  const runMutation = useMutation({
    mutationFn: async (): Promise<string | null> => {
      let targetAgentId: string | null = isNew ? null : id;

      // Persist any pending edits before running so the run reflects what the
      // user sees on screen. Empty/invalid drafts short-circuit to the save
      // mutation's existing validation banner.
      if (isNew || isDirty) {
        const result = await saveMutation.mutateAsync();
        if (!result) return null;
        targetAgentId = result.agent.id;
      }

      if (!targetAgentId) return null;
      try {
        window.localStorage.setItem(SELECTED_AGENT_KEY, targetAgentId);
      } catch {
        // localStorage may be unavailable (private mode); the chat will fall
        // back to the default agent selection. Not fatal.
      }
      clientMetrics.emit('agent_run_clicked', undefined, { agent_id: targetAgentId });
      return targetAgentId;
    },
    onSuccess: (targetAgentId) => {
      if (targetAgentId) {
        void navigate('/chat');
      }
    },
  });

  function handleRun(): void {
    runMutation.mutate();
  }

  // -------------------------------------------------------------------------
  // Phase 2-B: DeployDropdown — surfaces all 4 deployment modes.
  // Chat-picker visibility is handled via the D2-shim in the API layer
  // (deployments.py flips agent.visibility on ACTIVE transition for IN_APP).
  // -------------------------------------------------------------------------

  const [latestDeployment, setLatestDeployment] =
    React.useState<DeploymentResponse | null>(null);

  const revisionsQuery = useQuery({
    queryKey: ['agents-v2', id, 'revisions', 'latest'],
    queryFn: async () => listRevisions(id, 1),
    enabled: !isNew,
  });
  const latestRevisionId = revisionsQuery.data?.items[0]?.rev_id ?? null;

  /**
   * W5: Resolve the revision id to deploy. If the canvas is dirty (or
   * brand-new), auto-save first so the wizard ships the AST the user
   * actually sees on screen — mirrors Test Run's pre-flight at line ~272
   * (extracted here so Deploy gets the same safety). Returns:
   *   - the freshly-saved revision id when a save occurred
   *   - the cached ``latestRevisionId`` when the canvas was already clean
   *   - ``null`` when validation failed or there's nothing to save (the
   *     deploy dropdown will abort opening any wizard)
   */
  const ensureSavedRevisionId = React.useCallback(async (): Promise<string | null> => {
    setDeploySyncError(null);
    if (!ast) {
      setDeploySyncError('No workflow canvas is loaded to deploy.');
      return null;
    }

    const visiblePayload = buildDesignerSavePayload(ast, {
      isNew,
      agentName,
      localName,
      agentDescription,
      localDescription,
    });
    const dirtyAtClick = isDirty;
    let resolvedAgentId = id;
    let resolvedRevisionId = latestRevisionId;

    if (isNew || isDirty) {
      const result = await saveMutation.mutateAsync();
      if (!result) {
        setDeploySyncError('Save or validation failed. Fix the workflow before deploying.');
        return null;
      }
      resolvedAgentId = result.agent.id;
      await queryClient.invalidateQueries({ queryKey: ['agents-v2'] });
      const fresh = await listRevisions(result.agent.id, 1);
      resolvedRevisionId = fresh.items[0]?.rev_id ?? null;
    }

    if (!resolvedRevisionId) {
      setDeploySyncError('No saved revision is available to deploy.');
      return null;
    }

    const revision = await getRevision(resolvedAgentId, resolvedRevisionId);
    console.debug('[DEPLOY_HANDOFF]', {
      visible_agent_id: resolvedAgentId,
      visible_agent_name: agentName,
      visible_agent_description: agentDescription,
      visible_workflow_name: visiblePayload.definition.name,
      visible_workflow_description: visiblePayload.definition.description,
      visible_root_child_summary: summarizeRootChildren(visiblePayload.definition),
      resolved_revision_id: resolvedRevisionId,
      revision_workflow_name: revision.definition.name,
      revision_workflow_description: revision.definition.description,
      revision_root_child_summary: summarizeRootChildren(revision.definition),
      dirty_at_click: dirtyAtClick,
    });

    if (!deploymentIdentityMatches(revision.definition, visiblePayload.definition)) {
      setDeploySyncError(
        'The saved revision does not match the visible canvas. Save again before deploying.',
      );
      return null;
    }

    // The save just created a new revision row server-side; invalidate the
    // cached "latest" so subsequent reads see it and re-fetch the head.
    await queryClient.invalidateQueries({
      queryKey: ['agents-v2', resolvedAgentId, 'revisions', 'latest'],
    });
    return resolvedRevisionId;
  }, [
    agentDescription,
    agentName,
    ast,
    id,
    isDirty,
    isNew,
    latestRevisionId,
    localDescription,
    localName,
    queryClient,
    saveMutation,
  ]);

  // -------------------------------------------------------------------------
  // ETag conflict modal handlers
  // -------------------------------------------------------------------------

  function handleConflictReload(): void {
    void agentQuery.refetch().then((result) => {
      if (result.data) {
        load({ agent: result.data.agent, etag: result.data.etag });
      }
    });
  }

  // W15: when the conflict modal opens, fetch the latest server AST so the
  // merge UI inside EtagConflictModal has something to diff against.
  // Cleared on close to avoid stale data leaking into a subsequent
  // conflict on a different agent.
  React.useEffect(() => {
    if (!conflictModalOpen) {
      setConflictServerAst(null);
      return;
    }
    if (isNew) return;
    let cancelled = false;
    void getAgentV2WithEtag(id).then(({ agent }) => {
      if (cancelled) return;
      setConflictServerAst(agent.definition as unknown as import('@/types/ast').AST);
    });
    return () => {
      cancelled = true;
    };
  }, [conflictModalOpen, id, isNew]);

  async function handleConflictMergeSave(
    mergedAst: import('@/types/ast').AST,
    serverEtag: string,
  ): Promise<void> {
    // W15: user resolved the three-way merge and clicked "Save merge". We
    // PUT the merged AST with the server's current etag so the write
    // lands cleanly on top of the latest revision.
    const payload = buildDesignerSavePayload(mergedAst, {
      isNew,
      agentName,
      localName,
      agentDescription,
      localDescription,
    });
    const { definition } = payload;
    try {
      const { agent, etag: newEtag } = await updateAgentV2(
        id,
        {
          name: payload.name,
          description: payload.description,
          definition: definition as unknown as Record<string, unknown>,
        },
        serverEtag,
      );
      markClean(newEtag);
      load({ agent, etag: newEtag });
      setConflictModalOpen(false);
    } catch (error) {
      if (error instanceof EtagConflictError) {
        // Race: someone else wrote between the fetch and our PUT. Re-arm
        // the modal with the new etag — the user can re-merge.
        setConflictEtag(error.current_etag);
      }
    }
  }

  async function handleForceOverwrite(): Promise<void> {
    if (!ast) return;
    const payload = buildDesignerSavePayload(ast, {
      isNew,
      agentName,
      localName,
      agentDescription,
      localDescription,
    });
    const { definition } = payload;
    try {
      const { etag: latestEtag } = await getAgentV2WithEtag(id);
      const currentEtag = latestEtag ?? '';
      const { agent, etag: newEtag } = await updateAgentV2(
        id,
        {
          name: payload.name,
          description: payload.description,
          definition: definition as unknown as Record<string, unknown>,
        },
        currentEtag,
      );
      markClean(newEtag);
      load({ agent, etag: newEtag });
    } catch (error) {
      if (error instanceof EtagConflictError) {
        setConflictEtag(error.current_etag);
        setConflictModalOpen(true);
      }
    }
  }

  // Save button disabled state
  const saveDisabled = !isNew && !isDirty;

  // -------------------------------------------------------------------------
  // Loading / error states (edit flow only)
  // -------------------------------------------------------------------------

  if (!isNew && agentQuery.isLoading) {
    return (
      <AppShell>
        <div className="flex flex-1 items-center justify-center bg-db-oat-light text-[13px] text-db-gray-text">
          Loading agent…
        </div>
      </AppShell>
    );
  }

  if (!isNew && agentQuery.isError) {
    return (
      <AppShell>
        <div className="flex flex-1 items-center justify-center bg-db-oat-light text-[13px] text-db-lava-700">
          {agentQuery.error instanceof Error
            ? agentQuery.error.message
            : 'Failed to load agent'}
        </div>
      </AppShell>
    );
  }

  // -------------------------------------------------------------------------
  // Status badge
  // -------------------------------------------------------------------------

  let statusEl: React.ReactNode;
  if (validationErrors.length > 0) {
    statusEl = (
      <span
        className="inline-flex items-center gap-1 rounded-db-pill bg-db-lava-100 px-2 py-0.5 font-db-mono text-[11px] font-semibold uppercase tracking-[0.04em] text-db-lava-700"
        aria-label={`${validationErrors.length} validation error${validationErrors.length === 1 ? '' : 's'}`}
      >
        {validationErrors.length} error{validationErrors.length === 1 ? '' : 's'}
      </span>
    );
  } else if (isDirty) {
    statusEl = (
      <span className="inline-flex items-center gap-1 text-[12px] text-db-yellow-700">
        <Zap size={13} /> Unsaved changes
      </span>
    );
  } else {
    statusEl = (
      <span
        className={`inline-flex items-center gap-1 text-[12px] ${
          savedFlash ? 'text-db-green-700 db-anim-saveFlash' : 'text-db-gray-text'
        }`}
      >
        <Check size={13} /> Saved
      </span>
    );
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <AppShell>
      <div className="db-root flex min-h-0 flex-1 flex-col bg-db-oat-light font-db-sans text-db-navy-800">
        {/* TopBar */}
        <header className="flex h-14 shrink-0 items-center gap-3.5 border-b border-db-gray-lines bg-white px-5">
          <button
            type="button"
            onClick={() => navigate('/agents')}
            aria-label="Back to agents"
            className="rounded p-1 text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800"
          >
            <ChevronLeft size={14} />
          </button>
          <div className="flex items-center gap-2 text-[13px] text-db-gray-text">
            <span>workspace</span>
            <span className="text-db-navy-300">/</span>
            <button
              type="button"
              onClick={() => navigate('/agents')}
              className="hover:text-db-navy-800"
            >
              Agents
            </button>
            <span className="text-db-navy-300">/</span>
            <span className="font-medium text-db-navy-800">{agentName || 'Untitled Agent'}</span>
          </div>
          <span className="inline-flex items-center gap-1.5 rounded-db-pill bg-db-oat-medium px-2 py-0.5 font-db-mono text-[11px] text-db-gray-text">
            <span className="h-[5px] w-[5px] rounded-full bg-db-green-700" />
            {agentIdShort}
          </span>

          <div className="ml-auto flex items-center gap-3">
            {statusEl}
            <button
              type="button"
              onClick={handleSave}
              disabled={saveDisabled || saveMutation.isPending}
              aria-label="Save agent"
              className="inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium disabled:cursor-not-allowed disabled:opacity-55"
            >
              <Save size={13} /> {saveMutation.isPending ? 'Saving…' : 'Save'}
            </button>
            {validationErrors.length > 0 &&
              validationErrors.every((e) => e.kind === 'coverage') && (
                <button
                  type="button"
                  onClick={() => {
                    forceSaveRef.current = true;
                    saveMutation.mutate();
                  }}
                  disabled={saveMutation.isPending}
                  aria-label="Save as draft anyway"
                  title="The workflow doesn't yet cover every requested topic. Save it as a draft and keep refining."
                  className="inline-flex items-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-3 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium disabled:cursor-not-allowed disabled:opacity-55"
                >
                  Save as draft anyway
                </button>
              )}
            <ExportYamlMenu
              getYaml={handleExportYaml}
              filename={exportFilename}
              disabled={!ast}
            />
            {!isNew && (
              <button
                type="button"
                onClick={() =>
                  setActiveTab((t) => (t === 'revisions' ? 'edit' : 'revisions'))
                }
                aria-pressed={activeTab === 'revisions'}
                className={`inline-flex items-center gap-1.5 rounded-db-md border px-3 py-1.5 text-[13px] font-medium transition-colors ${
                  activeTab === 'revisions'
                    ? 'border-db-navy-300 bg-db-oat-medium text-db-navy-800'
                    : 'border-db-gray-lines bg-white text-db-navy-800 hover:border-db-navy-300 hover:bg-db-oat-medium'
                }`}
              >
                <HistoryIcon size={13} /> Revisions
              </button>
            )}
            <span className="h-5 w-px bg-db-gray-lines" />
            <button
              type="button"
              onClick={handleRun}
              disabled={
                runMutation.isPending ||
                saveMutation.isPending ||
                (isNew && !localName.trim())
              }
              aria-label="Test run agent"
              title={
                isNew && !localName.trim()
                  ? 'Name the agent before running'
                  : 'Save and open a new chat session bound to this agent'
              }
              className="inline-flex items-center gap-1.5 rounded-db-md bg-db-navy-800 px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-db-navy-900 disabled:cursor-not-allowed disabled:opacity-55"
            >
              <Play size={11} /> {runMutation.isPending ? 'Starting…' : 'Test run'}
            </button>
            {/* DeployDropdown surfaces all 4 deployment modes (D1: legacy
                visibility-flip button removed; chat-picker visibility is
                handled by the D2-shim in deployments.py). */}
            {!isNew && latestRevisionId && (
              <DeployDropdown
                agentId={id}
                agentName={agentName || 'Untitled Agent'}
                revisionId={latestRevisionId}
                onBeforeDeploy={ensureSavedRevisionId}
                onDeployed={(deployment) => {
                  setLatestDeployment(deployment);
                }}
              />
            )}
            {latestDeployment && (
              <div className="ml-2 max-w-[320px]">
                <StatusPanel
                  deploymentId={latestDeployment.id}
                  deployment={latestDeployment}
                />
              </div>
            )}
          </div>
        </header>

        {deploySyncError && (
          <div
            role="alert"
            className="border-b border-db-lava-300 bg-db-lava-100 px-5 py-2 text-[12px] text-db-lava-700"
          >
            {deploySyncError}
          </div>
        )}

        {saveNotice && (
          <div
            role="alert"
            className={`border-b px-5 py-2 text-[12px] ${
              saveNotice.kind === 'error'
                ? 'border-db-lava-300 bg-db-lava-100 text-db-lava-700'
                : 'border-db-yellow-300 bg-db-yellow-100 text-db-yellow-700'
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <span>{saveNotice.message}</span>
                {saveNotice.validation && Array.isArray(saveNotice.validation.directives) && saveNotice.validation.directives.length > 0 && (
                  <ul className="mt-1.5 space-y-0.5">
                    {saveNotice.validation.directives.map((d, idx) => (
                      <li
                        key={`${d.node_path}-${idx}`}
                        className={`flex items-baseline gap-1 leading-[1.4] ${
                          d.severity === 'blocking'
                            ? 'font-medium'
                            : 'opacity-85'
                        }`}
                      >
                        <span aria-hidden="true">•</span>
                        <span>
                          <span className="font-db-mono">{d.node_path}</span>
                          {': '}
                          {d.issue}
                          {' → '}
                          {d.suggested_action}
                          {d.severity === 'blocking' && (
                            <span className="ml-1 rounded-sm bg-current/20 px-1 py-px text-[10px] font-semibold uppercase tracking-[0.04em]">
                              blocking
                            </span>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {saveNotice.validation && saveNotice.validation.verdict !== 'pass' && (
                  <button
                    type="button"
                    onClick={() => {
                      const v = saveNotice.validation!;
                      const lines = [
                        `Please fix these validation issues from the last save (verdict=${v.verdict}): ${v.summary}`,
                        ...(v.directives ?? []).map(
                          (d) => `- [${d.node_path}] ${d.issue} — ${d.suggested_action}`,
                        ),
                      ];
                      useAgentEditorStore.getState().setPendingChatSeed(lines.join('\n'));
                      setSaveNotice(null);
                    }}
                    className={`rounded-db-md border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                      saveNotice.kind === 'error'
                        ? 'border-db-lava-400 bg-white text-db-lava-700 hover:bg-db-lava-100'
                        : 'border-db-yellow-500 bg-white text-db-yellow-800 hover:bg-db-yellow-100'
                    }`}
                  >
                    Ask designer to fix
                  </button>
                )}
                <button
                  type="button"
                  aria-label="Dismiss save notice"
                  onClick={() => setSaveNotice(null)}
                  className="rounded p-0.5 opacity-70 hover:opacity-100"
                >
                  ✕
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Body: canvas + inspector + chat */}
        <div className="flex min-h-0 flex-1">
          <main className="flex-1 overflow-auto bg-db-oat-light">
            <div className="mx-auto max-w-[720px] px-7 pb-20 pt-6">
              {/* Name + description + meta */}
              <div className="mb-[18px]">
                <input
                  value={isNew ? localName : agentName}
                  onChange={(e) => {
                    if (isNew) setLocalName(e.target.value);
                  }}
                  readOnly={!isNew}
                  placeholder="Agent name"
                  aria-label="Agent name"
                  className="w-full border-0 bg-transparent p-0 font-db-sans text-[22px] font-medium leading-[1.2] tracking-[-0.015em] text-db-navy-800 outline-none placeholder:text-db-navy-300 read-only:cursor-default"
                />
                <input
                  value={isNew ? localDescription : agentDescription}
                  onChange={(e) => {
                    if (isNew) setLocalDescription(e.target.value);
                  }}
                  readOnly={!isNew}
                  placeholder="Description (optional)"
                  aria-label="Agent description"
                  className="mt-1 w-full border-0 bg-transparent p-0 font-db-sans text-[13px] leading-[1.5] text-db-gray-text outline-none placeholder:text-db-navy-300 read-only:cursor-default"
                />
                <div className="mt-3 flex items-center gap-3 text-[11px] text-db-gray-text">
                  {agentIdFull && (
                    <span className="inline-flex items-center gap-1.5" title={agentIdFull}>
                      <HashIcon size={11} />
                      <span className="truncate font-db-mono">{agentIdFull}</span>
                    </span>
                  )}
                  {agentUpdatedAt && (
                    <span className="inline-flex items-center gap-1.5">
                      <HistoryIcon size={11} /> Modified {formatRelativeShort(agentUpdatedAt)}
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1 rounded-db-pill bg-db-blue-100 px-2 py-0.5 font-db-mono text-[10px] font-medium tracking-[0.02em] text-db-blue-700">
                    Agent V2
                  </span>
                </div>
              </div>

              {/* Tabs */}
              <div className="mb-5 flex border-b border-db-gray-lines">
                <TabButton
                  active={activeTab === 'edit'}
                  onClick={() => setActiveTab('edit')}
                  label="Edit"
                />
                {!isNew && (
                  <TabButton
                    active={activeTab === 'revisions'}
                    onClick={() => setActiveTab('revisions')}
                    label="Revisions"
                  />
                )}
                <TabButton
                  active={activeTab === 'settings'}
                  onClick={() => setActiveTab('settings')}
                  label="Settings"
                />
                {!isNew && (
                  <TabButton
                    active={activeTab === 'deployments'}
                    onClick={() => setActiveTab('deployments')}
                    label="Deployments"
                  />
                )}
              </div>

              {/* Tab content */}
              {activeTab === 'edit' && (
                <ErrorBoundary name="Designer">
                  <BlockEditor registry={registry} />
                </ErrorBoundary>
              )}
              {activeTab === 'revisions' && !isNew && (
                <div className="flex min-h-[400px] gap-4 rounded-db-md border border-db-gray-lines bg-white p-4">
                  <div className="w-72 shrink-0 overflow-auto">
                    <RevisionList
                      agentId={id}
                      selectedRevId={selectedRevId}
                      onSelectRevision={setSelectedRevId}
                    />
                  </div>
                  <div className="flex-1 overflow-auto border-l border-db-gray-lines pl-4">
                    {selectedRevId ? (
                      <RevisionPreview agentId={id} revId={selectedRevId} />
                    ) : (
                      <div className="flex h-full items-center justify-center text-[13px] text-db-gray-text">
                        Select a revision to preview
                      </div>
                    )}
                  </div>
                </div>
              )}
              {activeTab === 'settings' && (
                <div className="rounded-db-md border border-db-gray-lines bg-white p-6 text-[13px] leading-[1.55] text-db-gray-text">
                  <p className="mb-1 font-db-mono text-[10px] font-medium uppercase tracking-[0.06em] text-db-gray-text">
                    Research depth
                  </p>
                  <p className="mb-3 max-w-prose text-[12px] leading-[1.5] text-db-gray-text">
                    Scales how deeply this agent's researchers dig — their tool-call
                    budgets and any loop / plan-and-execute iteration counts.{' '}
                    <strong>Standard</strong> keeps the saved budgets;{' '}
                    <strong>Deep</strong> raises them for more thorough research;{' '}
                    <strong>Light</strong> reduces them for faster, shallower runs.
                    A per-chat selection overrides this saved default.
                  </p>
                  <select
                    aria-label="Research depth"
                    value={ast?.research_effort ?? 'standard'}
                    onChange={(e) => {
                      if (!ast) return;
                      const value = e.target.value as 'light' | 'standard' | 'deep';
                      useAgentEditorStore
                        .getState()
                        .setAst({ ...ast, research_effort: value });
                    }}
                    disabled={!ast}
                    className="w-64 rounded-db-md border border-db-gray-lines bg-white px-2 py-1.5 text-[13px] text-db-gray-text disabled:opacity-50"
                  >
                    <option value="light">Light — faster, shallower</option>
                    <option value="standard">Standard (default)</option>
                    <option value="deep">Deep — slower, more thorough</option>
                  </select>
                  <p className="mt-4 text-[12px] text-db-gray-text">
                    More agent-level settings (visibility, run-as principal, output
                    schema) coming soon.
                  </p>
                </div>
              )}
              {activeTab === 'deployments' && !isNew && (
                <DeploymentsSection agentId={id} />
              )}
            </div>
          </main>

          {/* Inspector — Direction 1 (Tabbed Inspector): Configure · Tools ·
              Approvals · Co-pilot. The co-pilot is the 4th tab (the separate
              Designer Chat column was removed). */}
          <ConfigPanel registry={registry} chatSessionId={isNew ? undefined : id} />
        </div>

        {/* ETag conflict modal */}
        <EtagConflictModal
          open={conflictModalOpen}
          onOpenChange={setConflictModalOpen}
          currentEtag={conflictEtag}
          onReload={handleConflictReload}
          onForceOverwrite={() => {
            void handleForceOverwrite();
          }}
          /* W15: enable the V1.5 three-way merge path. Both ASTs and the
             onSaveMerge handler are now wired — previously the modal had
             these props declared but the call site omitted them, so the
             merge UI was dead code. */
          localAst={ast ?? undefined}
          serverAst={conflictServerAst ?? undefined}
          onSaveMerge={(mergedAst, etag) => {
            void handleConflictMergeSave(mergedAst, etag);
          }}
        />

      </div>
    </AppShell>
  );
}

function TabButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}): React.ReactElement {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`-mb-px border-b-2 px-4 py-2.5 font-db-sans text-[13px] font-medium transition-colors ${
        active
          ? 'border-db-lava-600 text-db-navy-800'
          : 'border-transparent text-db-gray-text hover:text-db-navy-800'
      }`}
    >
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// AgentDesignerPage — loads registry then renders DesignerInner
// ---------------------------------------------------------------------------

export function AgentDesignerPage(): React.ReactElement {
  const { id = 'new' } = useParams<{ id: string }>();

  const registryQuery = useQuery({
    queryKey: ['agent-designer', 'registry'],
    queryFn: getRegistry,
    staleTime: Infinity,
  });

  if (registryQuery.isLoading) {
    return (
      <AppShell>
        <div className="flex flex-1 items-center justify-center bg-db-oat-light text-[13px] text-db-gray-text">
          Loading designer…
        </div>
      </AppShell>
    );
  }

  if (registryQuery.isError || !registryQuery.data) {
    return (
      <AppShell>
        <div className="flex flex-1 items-center justify-center bg-db-oat-light text-[13px] text-db-lava-700">
          {registryQuery.error instanceof Error
            ? registryQuery.error.message
            : 'Failed to load registry'}
        </div>
      </AppShell>
    );
  }

  return <DesignerInner id={id} registry={registryQuery.data} />;
}

export default AgentDesignerPage;
