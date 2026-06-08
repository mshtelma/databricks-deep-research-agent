import type { RevisionProvenance } from './revisionProvenance'

import { InfoCard } from './dialog-primitives'

export function RevisionProvenanceCard({
  provenance,
}: {
  provenance: RevisionProvenance | null
}) {
  if (!provenance) return null
  return (
    <div
      data-testid="revision-provenance-card"
      className="rounded-md border border-db-gray-lines bg-white px-3 py-2 text-[11px] text-db-gray-text"
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-medium text-db-navy-800">Deploying revision</span>
        <code>{provenance.shortAgentId}</code>
        <span>/</span>
        <code>{provenance.shortRevisionId}</code>
        <span className="text-db-navy-300">·</span>
        <span className="font-medium text-db-navy-800">{provenance.workflowName}</span>
      </div>
      {provenance.descriptionPreview ? (
        <div className="mt-1 line-clamp-2">{provenance.descriptionPreview}</div>
      ) : null}
      {provenance.rootChildSummary.length > 0 ? (
        <div className="mt-1 truncate font-db-mono text-[10px]">
          {provenance.rootChildSummary.join(' -> ')}
        </div>
      ) : null}
      {provenance.isDefaultScaffold ? (
        <InfoCard color="yellow">
          This revision looks like the empty default scaffold. The backend will
          block deployment until a designed workflow revision is saved or
          selected.
        </InfoCard>
      ) : null}
    </div>
  )
}
