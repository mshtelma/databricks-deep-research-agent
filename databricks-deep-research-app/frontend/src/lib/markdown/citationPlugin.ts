/**
 * citationPlugin - Remark plugin and utilities for parsing citations.
 *
 * Supports THREE citation formats:
 * 1. Human-readable keys: [Key] (e.g., [Arxiv], [Zhipu], [Github-2])
 * 2. Numeric markers: [N] (e.g., [1], [2], [12]) - legacy support
 * 3. Markdown links: [Title](url) - treated as inline citations
 *
 * Human-readable keys are generated from source domains/authors:
 * - arxiv.org → [Arxiv]
 * - docs.databricks.com → [Docs]
 * - Collisions handled: [Arxiv], [Arxiv-2], [Arxiv-3]
 */

import type { Plugin } from 'unified';
import type { Root, Text, Parent } from 'mdast';
import { visit } from 'unist-util-visit';

// Regular expression to match human-readable citation keys
// Matches: [Arxiv], [Zhipu], [Github], [Arxiv-2], [Docs-3], [My-site], [Salary-after-tax]
// Must start with a letter, can contain letters, numbers, hyphens (including mid-key hyphens)
// Synced with backend regex: r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]"
const CITATION_KEY_REGEX = /\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]/g;

// Legacy: Regular expression to match numeric citation markers [N]
// Matches [1], [2], [12], etc. but not [0] or negative numbers
const CITATION_REGEX = /\[(\d+)\]/g;

// Regular expression to match markdown links [Title](url)
const MARKDOWN_LINK_REGEX = /\[([^\]]+)\]\(([^)]+)\)/g;

/**
 * Citation info extracted from markdown link
 */
export interface LinkCitationInfo {
  index: number;
  title: string;
  url: string;
}

/**
 * Custom MDAST node type for citations
 */
export interface CitationNode {
  type: 'citation';
  /** Human-readable citation key (e.g., "Arxiv", "Zhipu", "Github-2") */
  citationKey: string;
  /** Legacy numeric index for backwards compatibility */
  index?: number;
  title?: string;
  url?: string;
  data?: {
    hName?: string;
    hProperties?: Record<string, unknown>;
  };
}

/**
 * Check if a node is a parent node
 */
function isParent(node: unknown): node is Parent {
  return typeof node === 'object' && node !== null && Array.isArray((node as Parent).children);
}

/**
 * Build a citation index from markdown content.
 * Extracts all [Title](url) links and assigns sequential indices.
 *
 * @param content - Markdown content to scan
 * @returns Map of index to citation info
 */
export function buildLinkCitationIndex(content: string): Map<number, LinkCitationInfo> {
  const urlToIndex = new Map<string, number>();
  const indexToInfo = new Map<number, LinkCitationInfo>();
  let currentIndex = 1;

  let match;
  const regex = new RegExp(MARKDOWN_LINK_REGEX.source, 'g');

  while ((match = regex.exec(content)) !== null) {
    const [, title, url] = match;
    if (!title || !url) continue;

    // Skip anchor links and javascript: links
    if (url.startsWith('#') || url.startsWith('javascript:')) continue;

    if (!urlToIndex.has(url)) {
      urlToIndex.set(url, currentIndex);
      indexToInfo.set(currentIndex, { index: currentIndex, title, url });
      currentIndex++;
    }
  }

  return indexToInfo;
}

/**
 * Get the citation index for a URL.
 * Returns undefined if URL is not in the citation index.
 */
export function getCitationIndexForUrl(
  url: string,
  citationIndex: Map<number, LinkCitationInfo>
): number | undefined {
  for (const [idx, info] of citationIndex) {
    if (info.url === url) return idx;
  }
  return undefined;
}

/**
 * Remark plugin to parse citation markers into custom nodes.
 *
 * Supports TWO formats:
 * 1. Human-readable keys: [Arxiv], [Zhipu], [Github-2] (preferred)
 * 2. Numeric markers: [1], [2], [12] (legacy)
 *
 * The plugin first tries to match human-readable keys, then falls back to numeric.
 */
export const remarkCitations: Plugin<[], Root> = () => {
  return (tree: Root) => {
    visit(tree, 'text', (node: Text, index: number | undefined, parent: Parent | undefined) => {
      if (index === undefined || !parent) return;
      if (!isParent(parent)) return;

      const value = node.value;
      const matches: Array<{ citationKey: string; index?: number; start: number; end: number }> = [];

      // First try human-readable key pattern: [Arxiv], [Zhipu], [Github-2]
      let match: RegExpExecArray | null;
      const keyRegex = new RegExp(CITATION_KEY_REGEX.source, 'g');
      while ((match = keyRegex.exec(value)) !== null) {
        const matchGroup = match[1];
        if (!matchGroup) continue;
        matches.push({
          citationKey: matchGroup,
          start: match.index,
          end: match.index + match[0].length,
        });
      }

      // If no key matches, try numeric pattern: [1], [2], [12] (legacy)
      if (matches.length === 0) {
        const numericRegex = new RegExp(CITATION_REGEX.source, 'g');
        while ((match = numericRegex.exec(value)) !== null) {
          const matchGroup = match[1];
          if (!matchGroup) continue;
          const citationIndex = parseInt(matchGroup, 10);
          // Only process valid citation indices (>= 1)
          if (citationIndex >= 1) {
            matches.push({
              citationKey: matchGroup, // Use number as key for legacy
              index: citationIndex,
              start: match.index,
              end: match.index + match[0].length,
            });
          }
        }
      }

      if (matches.length === 0) return;

      // Sort matches by start position (in case regex finds them out of order)
      matches.sort((a, b) => a.start - b.start);

      // Build replacement nodes
      const newNodes: Array<Text | CitationNode> = [];
      let lastEnd = 0;

      for (const { citationKey, index: citIndex, start, end } of matches) {
        // Add text before this citation
        if (start > lastEnd) {
          const textBefore = value.slice(lastEnd, start);
          newNodes.push({
            type: 'text',
            value: textBefore,
          });
        }

        // Add citation node with citationKey
        newNodes.push({
          type: 'citation',
          citationKey: citationKey,
          index: citIndex,
          data: {
            hName: 'citation-marker',
            hProperties: {
              'data-citation-key': citationKey,
              // Keep index for legacy compatibility
              ...(citIndex !== undefined && { 'data-citation-index': citIndex }),
            },
          },
        });

        lastEnd = end;
      }

      // Add remaining text after last citation
      if (lastEnd < value.length) {
        newNodes.push({
          type: 'text',
          value: value.slice(lastEnd),
        });
      }

      // Replace the original text node with new nodes
      parent.children.splice(index, 1, ...newNodes as Array<typeof parent.children[number]>);
    });
  };
};

/**
 * Extract all citation keys from markdown text.
 * Supports both human-readable keys [Arxiv] and numeric [1] format.
 *
 * @param markdown - Markdown content to scan
 * @returns Array of unique citation keys
 */
export function extractCitationKeys(markdown: string): string[] {
  const keys: string[] = [];

  // First try human-readable keys
  let match: RegExpExecArray | null;
  const keyRegex = new RegExp(CITATION_KEY_REGEX.source, 'g');
  while ((match = keyRegex.exec(markdown)) !== null) {
    const key = match[1];
    if (key && !keys.includes(key)) {
      keys.push(key);
    }
  }

  // If no keys found, try numeric indices (legacy)
  if (keys.length === 0) {
    const numericRegex = new RegExp(CITATION_REGEX.source, 'g');
    while ((match = numericRegex.exec(markdown)) !== null) {
      const key = match[1];
      if (key && !keys.includes(key)) {
        keys.push(key);
      }
    }
  }

  return keys;
}

/**
 * Extract all citation indices from markdown text (numeric [N] format)
 * Useful for pre-processing to know which citations need data
 * @deprecated Use extractCitationKeys() instead
 */
export function extractCitationIndices(markdown: string): number[] {
  const indices: number[] = [];
  let match: RegExpExecArray | null;
  const regex = new RegExp(CITATION_REGEX.source, 'g');

  while ((match = regex.exec(markdown)) !== null) {
    const matchGroup = match[1];
    if (!matchGroup) continue;
    const index = parseInt(matchGroup, 10);
    if (index >= 1 && !indices.includes(index)) {
      indices.push(index);
    }
  }

  return indices.sort((a, b) => a - b);
}

/**
 * Replace citation markers with a custom format
 * Useful for server-side rendering or exporting
 */
export function replaceCitationMarkers(
  markdown: string,
  replacer: (index: number) => string
): string {
  return markdown.replace(CITATION_REGEX, (match, indexStr) => {
    const index = parseInt(indexStr, 10);
    if (index >= 1) {
      return replacer(index);
    }
    return match;
  });
}

/**
 * Count citations in markdown text
 */
export function countCitations(markdown: string): number {
  return extractCitationIndices(markdown).length;
}

/**
 * Count markdown links in content (potential link-based citations)
 */
export function countLinkCitations(markdown: string): number {
  return buildLinkCitationIndex(markdown).size;
}

/**
 * Check if content uses link-style citations [Title](url) vs numeric [N]
 */
export function detectCitationStyle(markdown: string): 'numeric' | 'link' | 'mixed' | 'none' {
  const numericCount = countCitations(markdown);
  const linkCount = countLinkCitations(markdown);

  if (numericCount > 0 && linkCount > 0) return 'mixed';
  if (numericCount > 0) return 'numeric';
  if (linkCount > 0) return 'link';
  return 'none';
}

export default remarkCitations;
