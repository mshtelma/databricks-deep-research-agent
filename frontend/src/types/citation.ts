/**
 * Citation type definitions for claim-level attribution.
 *
 * This module defines TypeScript types for the 6-stage citation verification pipeline:
 * - Claims and their verification status
 * - Evidence spans and source metadata
 * - Citation links between claims and evidence
 * - Numeric claim details with QA verification
 * - Verification summaries and correction metrics
 */

/** Claim type classification */
export type ClaimType = 'general' | 'numeric';

/** Four-tier verification verdict */
export type VerificationVerdict =
  | 'supported'
  | 'partial'
  | 'unsupported'
  | 'contradicted';

/** HaluGate-style confidence level for routing */
export type ConfidenceLevel = 'high' | 'medium' | 'low';

/** Citation correction type */
export type CorrectionType = 'keep' | 'replace' | 'remove' | 'add_alternate';

/** Numeric value derivation type */
export type DerivationType = 'direct' | 'computed';

/** Source metadata for evidence cards */
export interface SourceMetadata {
  id: string;
  title: string | null;
  url: string | null;
  author: string | null;
  publishedDate: string | null;
  contentType: string | null;
  /** Total pages in document (for PDFs) */
  totalPages?: number;
}

/** Evidence span from a source document */
export interface EvidenceSpan {
  id: string;
  sourceId: string;
  quoteText: string;
  startOffset: number | null;
  endOffset: number | null;
  sectionHeading: string | null;
  relevanceScore: number | null;
  hasNumericContent: boolean;
  /** Denormalized source metadata for convenience */
  source: SourceMetadata;
}

/** Citation link between claim and evidence */
export interface Citation {
  evidenceSpan: EvidenceSpan;
  confidenceScore: number | null;
  isPrimary: boolean;
}

/** Citation correction record */
export interface CitationCorrection {
  id: string;
  correctionType: CorrectionType;
  originalEvidence: EvidenceSpan | null;
  correctedEvidence: EvidenceSpan | null;
  reasoning: string | null;
}

/** QA verification result for numeric claims */
export interface QAVerificationResult {
  question: string;
  claimAnswer: string;
  evidenceAnswer: string;
  match: boolean;
  normalizedComparison: {
    claimValue: number;
    evidenceValue: number;
  } | null;
}

/** Computation step for derived numeric values */
export interface ComputationStep {
  operation: string;
  inputs: Array<{
    value: number;
    sourceId: string;
    quote: string;
  }>;
  result: number;
}

/** Extended details for numeric claims */
export interface NumericClaimDetail {
  rawValue: string;
  normalizedValue: number | null;
  unit: string | null;
  entityReference: string | null;
  derivationType: DerivationType;
  computationDetails: {
    steps: ComputationStep[];
    formula: string | null;
  } | null;
  assumptions: {
    currencyYear: number | null;
    exchangeRate: number | null;
    roundingMethod: string | null;
    [key: string]: unknown;
  } | null;
  qaVerification: QAVerificationResult[] | null;
}

/** Full claim with all citation and verification data */
export interface Claim {
  id: string;
  claimText: string;
  claimType: ClaimType;
  confidenceLevel: ConfidenceLevel | null;
  positionStart: number;
  positionEnd: number;
  verificationVerdict: VerificationVerdict | null;
  verificationReasoning: string | null;
  abstained: boolean;
  citations: Citation[];
  corrections: CitationCorrection[];
  numericDetail: NumericClaimDetail | null;
  /** Primary citation key (e.g., "Arxiv", "Zhipu", "Github-2") */
  citationKey: string | null;
  /** All citation keys in this claim for multi-marker sentences */
  citationKeys: string[] | null;
}

/** Verification summary for a message */
export interface VerificationSummary {
  totalClaims: number;
  supportedCount: number;
  partialCount: number;
  unsupportedCount: number;
  contradictedCount: number;
  abstainedCount: number;
  unsupportedRate: number;
  contradictedRate: number;
  warning: boolean;
}

/** Correction metrics for a message */
export interface CorrectionMetrics {
  totalCorrections: number;
  keepCount: number;
  replaceCount: number;
  removeCount: number;
  addAlternateCount: number;
  correctionRate: number;
}

/** Response for GET /messages/{id}/claims endpoint */
export interface MessageClaimsResponse {
  messageId: string;
  claims: Claim[];
  verificationSummary: VerificationSummary;
  correctionMetrics: CorrectionMetrics | null;
}

/** Response for GET /claims/{id}/evidence endpoint */
export interface ClaimEvidenceResponse {
  claimId: string;
  claimText: string;
  verificationVerdict: VerificationVerdict | null;
  citations: Citation[];
}

/** Provenance export format */
export interface ProvenanceExport {
  exportedAt: string;
  messageId: string;
  claims: Array<{
    claimText: string;
    claimType: ClaimType;
    verdict: VerificationVerdict | null;
    confidenceLevel: ConfidenceLevel | null;
    citations: Array<{
      sourceUrl: string | null;
      sourceTitle: string | null;
      quote: string;
      isPrimary: boolean;
    }>;
    numericDetail: NumericClaimDetail | null;
    corrections: CitationCorrection[];
  }>;
  summary: VerificationSummary;
}

// SSE Event Types for real-time updates

/** Claim generated during interleaved synthesis */
export interface ClaimGeneratedEvent {
  eventType: 'claim_generated';
  timestamp: string;
  claimText: string;
  positionStart: number;
  positionEnd: number;
  evidencePreview: string;
  confidenceLevel: ConfidenceLevel;
}

/** Claim verification completed */
export interface ClaimVerifiedEvent {
  eventType: 'claim_verified';
  timestamp: string;
  claimId: string;
  claimText: string;
  positionStart: number;
  positionEnd: number;
  verdict: VerificationVerdict;
  confidenceLevel: ConfidenceLevel;
  evidencePreview: string;
  reasoning: string | null;
  /** Primary citation key for citationData mapping (e.g., "Arxiv", "Zhipu") */
  citationKey: string | null;
  /** All citation keys for multi-source claims */
  citationKeys: string[] | null;
}

/** Citation corrected during post-processing */
export interface CitationCorrectedEvent {
  eventType: 'citation_corrected';
  timestamp: string;
  claimId: string;
  correctionType: CorrectionType;
  reasoning: string | null;
}

/** Numeric claim detected */
export interface NumericClaimDetectedEvent {
  eventType: 'numeric_claim_detected';
  timestamp: string;
  claimId: string;
  rawValue: string;
  normalizedValue: string | null;
  unit: string | null;
  derivationType: DerivationType;
  qaVerified: boolean;
}

/** Verification summary for a message */
export interface VerificationSummaryEvent {
  eventType: 'verification_summary';
  timestamp: string;
  messageId: string;
  totalClaims: number;
  supported: number;
  partial: number;
  unsupported: number;
  contradicted: number;
  abstainedCount: number;
  citationCorrections: number;
  warning: boolean;
}

/** Union type for all citation-related SSE events */
export type CitationStreamEvent =
  | ClaimGeneratedEvent
  | ClaimVerifiedEvent
  | CitationCorrectedEvent
  | NumericClaimDetectedEvent
  | VerificationSummaryEvent;

/** Verdict color mapping for UI */
export const VERDICT_COLORS: Record<VerificationVerdict, string> = {
  supported: 'green',
  partial: 'amber',
  unsupported: 'red',
  contradicted: 'purple',
};

/** Verdict labels for display */
export const VERDICT_LABELS: Record<VerificationVerdict, string> = {
  supported: 'Supported',
  partial: 'Partially Supported',
  unsupported: 'Unsupported',
  contradicted: 'Contradicted',
};

/** Confidence level colors for UI */
export const CONFIDENCE_COLORS: Record<ConfidenceLevel, string> = {
  high: 'green',
  medium: 'amber',
  low: 'red',
};
