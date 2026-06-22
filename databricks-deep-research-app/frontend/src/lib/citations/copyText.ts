/**
 * buildCitationCopyText - assemble a plain-text snippet for the "Copy" action on
 * an evidence card. Pure and domain-agnostic: works for any source/quote and
 * degrades gracefully when fields are missing.
 *
 * Format:
 *   "<supporting quote>"
 *
 *   Source: <title> — <url>
 *
 * Falls back to the claim text when no quote is available, and omits the Source
 * line entirely when neither title nor url is present.
 */
import type { Citation } from '@/types/citation';

export function buildCitationCopyText(
  citation: Citation | null | undefined,
  claimText?: string | null
): string {
  const span = citation?.evidenceSpan;
  const source = span?.source;

  const quote = span?.quoteText?.trim();
  const title = source?.title?.trim();
  // `url` is the denormalized source URL; tolerate a legacy `sourceUrl` shape.
  const url =
    source?.url ||
    (span as { sourceUrl?: string } | undefined)?.sourceUrl ||
    undefined;

  const lines: string[] = [];

  if (quote) {
    lines.push(`"${quote}"`);
  } else if (claimText && claimText.trim()) {
    lines.push(claimText.trim());
  }

  const attribution = [title, url].filter(Boolean).join(' — ');
  if (attribution) {
    lines.push(`Source: ${attribution}`);
  }

  return lines.join('\n\n');
}

export default buildCitationCopyText;
