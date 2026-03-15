/**
 * Markdown processing utilities for citation support
 */

export {
  remarkCitations,
  extractCitationKeys,
  extractCitationIndices,
  replaceCitationMarkers,
  countCitations,
} from './citationPlugin';
export type { CitationNode } from './citationPlugin';
