/**
 * Deterministic sample report for the Designer Preview "simulate" flow.
 *
 * Pure function: builds report-shaped markdown (echoing the compiled query
 * and surface inputs) plus a tiny fake citationData map so the preview's
 * ReportRegion shows the REAL rendering path — MarkdownRenderer with
 * interactive, verdict-colored citation chips — without executing anything.
 *
 * Content is intentionally generic (no domain vocabulary): everything
 * variable comes from the caller's own agent name / query / inputs.
 */

import type { CitationContext } from '@/components/common';
import type { Claim } from '@/types/citation';
import type { CompiledSubmission } from '@/lib/surfaceCompile';
import type { Surface } from '@/types/surface';
import { isPathRef } from '@/types/surface';

export interface SampleReport {
  markdown: string;
  citationData: Map<string, CitationContext>;
}

const MAX_QUERY_ECHO = 220;

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function makeSampleClaim(key: string, text: string): Claim {
  return {
    id: `sample-claim-${key}`,
    claimText: text,
    claimType: 'general',
    confidenceLevel: 'high',
    positionStart: 0,
    positionEnd: text.length,
    verificationVerdict: 'supported',
    verificationReasoning: 'Sample claim — no verification was performed.',
    abstained: false,
    citations: [],
    corrections: [],
    numericDetail: null,
    citationKey: key,
    citationKeys: [key],
  };
}

/**
 * Build the deterministic sample report for one compiled action.
 */
export function buildSampleReport(
  agentName: string,
  compiled: CompiledSubmission,
): SampleReport {
  const name = agentName.trim() || 'This agent';
  const query = compiled.query.trim();

  const lines: string[] = [];
  lines.push(`## Sample findings`);
  lines.push('');
  if (query) {
    lines.push(`**Request.** ${truncate(query, MAX_QUERY_ECHO)}`);
    lines.push('');
  }

  const inputEntries = Object.entries(compiled.surfaceInputs);
  if (inputEntries.length > 0) {
    lines.push(`**Inputs**`);
    lines.push('');
    for (const [key, value] of inputEntries) {
      lines.push(`- **${key}:** ${String(value)}`);
    }
    lines.push('');
  }

  lines.push(
    `${name} will research your request and write its findings here. ` +
      `Each substantive claim carries a citation marker linking it to the ` +
      `source it was drawn from [1].`,
  );
  lines.push('');
  lines.push(
    `Where the run gathers quantitative evidence, figures are reported with ` +
      `their units and sources, and verified claims are highlighted in the ` +
      `final report [2].`,
  );
  lines.push('');
  lines.push(`| Section | What appears here |`);
  lines.push(`| --- | --- |`);
  lines.push(`| Findings | The synthesized answer to your request |`);
  lines.push(`| Citations | Links from claims to their sources |`);
  lines.push(`| Verification | Per-claim support verdicts (when enabled) |`);
  lines.push('');
  lines.push(
    `*Run this action in a chat to replace this placeholder with a real report.*`,
  );

  const citationData = new Map<string, CitationContext>([
    [
      '1',
      {
        claim: makeSampleClaim(
          '1',
          'Each substantive claim carries a citation marker.',
        ),
        verdict: 'supported',
        url: 'https://example.com/sample-source-1',
      },
    ],
    [
      '2',
      {
        claim: makeSampleClaim(
          '2',
          'Figures are reported with their units and sources.',
        ),
        verdict: 'supported',
        url: 'https://example.com/sample-source-2',
      },
    ],
  ]);

  return { markdown: lines.join('\n'), citationData };
}

// ---------------------------------------------------------------------------
// Structured-output sample payloads (Preview simulate flow)
// ---------------------------------------------------------------------------

/** Output components and the prop naming their slot pointer. */
const SAMPLE_SLOT_PROPS: Record<string, string> = {
  Table: 'source',
  MetricGrid: 'source',
  KeyFindings: 'source',
  Chart: 'source',
  List: 'items',
};

interface SampleColumn {
  key: string;
  label: string;
  type: string;
}

function sampleTableRows(columns: SampleColumn[]): Record<string, unknown>[] {
  const firstStringKey = columns.find((c) => c.type === 'string')?.key;
  return [1, 2, 3].map((n) => {
    const row: Record<string, unknown> = {};
    for (const col of columns) {
      if (col.type === 'number') {
        row[col.key] = n * 10 + columns.indexOf(col);
      } else if (col.type === 'date') {
        row[col.key] = `2026-0${n}-01`;
      } else {
        const marker =
          col.key === firstStringKey && n <= 2 ? ` [${n}]` : '';
        row[col.key] = `${col.label} sample ${n}${marker}`;
      }
    }
    return row;
  });
}

/**
 * Deterministic sample data for every structured-output slot of *action*'s
 * binding — so the Preview's Tables/Metrics/Findings/Charts render populated.
 * Returns null when the binding has no slots.
 */
export function buildSamplePayload(
  surface: Surface,
  action: string,
): Record<string, unknown> | null {
  const binding = surface.bindings.find((b) => b.action === action);
  if (!binding) return null;
  const prefix = `${binding.output.target}/data/`;
  const payload: Record<string, unknown> = {};

  for (const comp of surface.components) {
    const pointerProp = SAMPLE_SLOT_PROPS[comp.component];
    if (!pointerProp) continue;
    const ref = comp.props[pointerProp];
    if (!isPathRef(ref) || !ref.path.startsWith(prefix)) continue;
    const slot = ref.path.slice(prefix.length);
    if (!slot || slot.includes('/')) continue;

    if (comp.component === 'Table') {
      const raw = comp.props['columns'];
      const columns: SampleColumn[] = Array.isArray(raw)
        ? raw.filter(
            (c): c is SampleColumn =>
              typeof c === 'object' &&
              c !== null &&
              typeof (c as SampleColumn).key === 'string' &&
              typeof (c as SampleColumn).label === 'string' &&
              typeof (c as SampleColumn).type === 'string',
          )
        : [];
      if (columns.length > 0) payload[slot] = sampleTableRows(columns);
    } else if (comp.component === 'Chart') {
      if (payload[slot] !== undefined) continue; // shared Table slot wins
      const xKey =
        typeof comp.props['x_key'] === 'string' ? comp.props['x_key'] : 'x';
      const rawY = comp.props['y_keys'];
      const yKeys = Array.isArray(rawY)
        ? rawY.filter((k): k is string => typeof k === 'string')
        : [];
      payload[slot] = [1, 2, 3].map((n) => {
        const row: Record<string, unknown> = { [xKey]: `Sample ${n}` };
        yKeys.forEach((key, index) => {
          row[key] = n * 10 + index;
        });
        return row;
      });
    } else if (comp.component === 'MetricGrid') {
      payload[slot] = [
        { label: 'Sample metric', value: '42', unit: '%', delta: '+3 vs prior' },
        { label: 'Second metric', value: '7' },
      ];
    } else {
      payload[slot] = [
        'First sample finding [1].',
        'Second sample finding [2].',
        'Third sample item.',
      ];
    }
  }
  return Object.keys(payload).length > 0 ? payload : null;
}
