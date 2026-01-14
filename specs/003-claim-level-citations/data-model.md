# Data Model: Claim-Level Citation Granularity

**Feature**: 003-claim-level-citations
**Date**: 2025-12-25
**Status**: Updated with 2024-2025 SOTA Research
**Last Updated**: 2025-12-25

## Overview

This document defines the data model extensions required for claim-level citation granularity. The design extends the existing `Message` and `Source` models to support:

- Atomic claim extraction with confidence classification
- Evidence span linking with ranking
- Four-tier verification verdicts (SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED)
- Citation correction tracking
- QA-based numeric claim verification

## Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌───────────────────┐
│     Message     │───1:N─│      Claim      │───N:M─│   EvidenceSpan    │
│                 │       │                 │       │                   │
│ id              │       │ id              │       │ id                │
│ content         │       │ message_id      │       │ source_id         │
│ metadata        │       │ claim_text      │       │ quote_text        │
└─────────────────┘       │ claim_type      │       │ start_offset      │
                          │ confidence_level│ NEW   │ end_offset        │
                          │ position_*      │       │ section           │
                          │ verdict         │       │ relevance_score   │ NEW
                          │ reasoning       │       │ has_numeric       │ NEW
                          └────────┬────────┘       └─────────┬─────────┘
                                   │                          │
                                   └──────────┬───────────────┘
                                              │
                                       ┌──────┴──────┐
                                       │   Citation  │
                                       │             │
                                       │ id          │
                                       │ claim_id    │
                                       │ evidence_id │
                                       │ confidence  │
                                       │ is_primary  │ NEW
                                       └─────────────┘

┌─────────────────┐       ┌─────────────────────┐
│      Claim      │───1:1─│    NumericClaim     │
│ (type=numeric)  │       │                     │
│                 │       │ claim_id (PK/FK)    │
└─────────────────┘       │ raw_value           │
                          │ normalized_value    │
                          │ unit                │
                          │ entity_reference    │
                          │ derivation_type     │
                          │ computation_details │
                          │ assumptions         │
                          │ qa_verification     │ NEW
                          └─────────────────────┘

┌─────────────────┐       ┌─────────────────────────┐
│      Claim      │───1:N─│   CitationCorrection    │  NEW
│                 │       │                         │
└─────────────────┘       │ id                      │
                          │ claim_id                │
                          │ original_evidence_id    │
                          │ corrected_evidence_id   │
                          │ correction_type         │
                          │ reasoning               │
                          └─────────────────────────┘
```

## Entities

### 1. Claim

An atomic factual assertion extracted from an agent response message.

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| id | UUID | PK | Unique claim identifier |
| message_id | UUID | FK → messages.id, NOT NULL, CASCADE | Parent agent message |
| claim_text | TEXT | NOT NULL | The extracted claim text |
| claim_type | VARCHAR(20) | NOT NULL, CHECK IN ('general', 'numeric') | Type of claim |
| **confidence_level** | VARCHAR(20) | CHECK IN ('high', 'medium', 'low') | **NEW**: HaluGate-style confidence for routing |
| position_start | INT | NOT NULL | Start character offset in message content |
| position_end | INT | NOT NULL | End character offset in message content |
| verification_verdict | VARCHAR(20) | CHECK IN ('supported', 'partial', 'unsupported', **'contradicted'**) | **UPDATED**: Four-tier verdict |
| verification_reasoning | TEXT | NULLABLE | Explanation for the verdict |
| **abstained** | BOOLEAN | DEFAULT FALSE | **NEW**: Whether verification was abstained due to insufficient evidence |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Creation timestamp |

**Indexes**:
- `idx_claims_message_id` on `message_id` (for retrieving claims by message)
- `idx_claims_verdict` on `verification_verdict` (for filtering by support level)
- `idx_claims_confidence` on `confidence_level` (for routing analysis)

**Relationships**:
- Belongs to one `Message`
- Has many `Citation` links to `EvidenceSpan`
- Has many `CitationCorrection` records
- May have one `NumericClaim` extension (if claim_type = 'numeric')

### 2. EvidenceSpan

A minimal text passage from a source document that supports one or more claims.

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| id | UUID | PK | Unique evidence span identifier |
| source_id | UUID | FK → sources.id, NOT NULL, CASCADE | Parent source document |
| quote_text | TEXT | NOT NULL | The exact supporting quote |
| start_offset | INT | NULLABLE | Start position in source content |
| end_offset | INT | NULLABLE | End position in source content |
| section_heading | VARCHAR(500) | NULLABLE | Section/page context |
| **relevance_score** | FLOAT | NULLABLE | **NEW**: Relevance ranking score (0.0-1.0) |
| **has_numeric_content** | BOOLEAN | DEFAULT FALSE | **NEW**: Whether span contains numeric data |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Creation timestamp |

**Indexes**:
- `idx_evidence_spans_source_id` on `source_id` (for retrieving spans by source)
- `idx_evidence_spans_relevance` on `relevance_score` (for ranked retrieval)

**Relationships**:
- Belongs to one `Source`
- Has many `Citation` links to `Claim`

### 3. Citation

A link between a claim and supporting evidence (many-to-many join table with additional attributes).

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| id | UUID | PK | Unique citation identifier |
| claim_id | UUID | FK → claims.id, NOT NULL, CASCADE | The claim being cited |
| evidence_span_id | UUID | FK → evidence_spans.id, NOT NULL, CASCADE | The supporting evidence |
| confidence_score | FLOAT | NULLABLE, CHECK 0.0-1.0 | Confidence in the attribution |
| **is_primary** | BOOLEAN | DEFAULT TRUE | **NEW**: Whether this is the primary citation (vs alternate) |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Creation timestamp |

**Constraints**:
- `UNIQUE(claim_id, evidence_span_id)` - prevent duplicate links

**Indexes**:
- `idx_citations_claim_id` on `claim_id`
- `idx_citations_evidence_span_id` on `evidence_span_id`

**Relationships**:
- Belongs to one `Claim`
- Belongs to one `EvidenceSpan`

### 4. NumericClaim

Extended metadata for numeric claims requiring special provenance tracking.

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| claim_id | UUID | PK, FK → claims.id, CASCADE | Parent claim (1:1 extension) |
| raw_value | TEXT | NOT NULL | The value as stated in source |
| normalized_value | NUMERIC | NULLABLE | Standardized numeric value |
| unit | VARCHAR(50) | NULLABLE | Unit of measurement |
| entity_reference | TEXT | NULLABLE | The entity the value describes |
| derivation_type | VARCHAR(20) | NOT NULL, CHECK IN ('direct', 'computed') | How the value was obtained |
| computation_details | JSONB | NULLABLE | Calculation steps if derived |
| assumptions | JSONB | NULLABLE | Applied assumptions (currency year, etc.) |
| **qa_verification** | JSONB | NULLABLE | **NEW**: QAFactEval verification results |

**QA Verification JSONB Structure** (NEW):
```json
{
  "questions": [
    {
      "question": "What was the Q4 2024 revenue?",
      "claim_answer": "$3.2 billion",
      "evidence_answer": "$3.2B",
      "match": true,
      "normalized_comparison": {
        "claim_value": 3200000000,
        "evidence_value": 3200000000
      }
    }
  ],
  "overall_match": true,
  "verification_timestamp": "2025-12-25T10:30:00Z"
}
```

**Relationships**:
- Extension of one `Claim` (claim_type must be 'numeric')

### 5. CitationCorrection (NEW)

Tracks corrections made to citations during the CiteFix post-processing stage.

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| id | UUID | PK | Unique correction identifier |
| claim_id | UUID | FK → claims.id, NOT NULL, CASCADE | The claim whose citation was corrected |
| original_evidence_span_id | UUID | FK → evidence_spans.id, NULLABLE | Original (incorrect) citation |
| corrected_evidence_span_id | UUID | FK → evidence_spans.id, NULLABLE | Corrected citation (NULL if removed) |
| correction_type | VARCHAR(20) | NOT NULL, CHECK IN ('keep', 'replace', 'remove', 'add_alternate') | Type of correction |
| reasoning | TEXT | NULLABLE | Explanation for the correction |
| created_at | TIMESTAMPTZ | NOT NULL, DEFAULT NOW() | Creation timestamp |

**Correction Types**:
- `keep`: Original citation is correct, no change needed (~60% of cases)
- `replace`: Found better citation from evidence pool (~25% of cases)
- `remove`: No valid citation exists, citation removed (~10% of cases)
- `add_alternate`: Multiple valid citations, added alternate (~5% of cases)

**Indexes**:
- `idx_citation_corrections_claim_id` on `claim_id`
- `idx_citation_corrections_type` on `correction_type` (for metrics)

**Relationships**:
- Belongs to one `Claim`
- References original and corrected `EvidenceSpan` (optional)

---

## Enums and Constants

### VerificationVerdict (UPDATED - Four Tiers)

```python
class VerificationVerdict(str, Enum):
    SUPPORTED = "supported"       # Claim FULLY entailed by evidence
    PARTIAL = "partial"           # Some aspects supported, others unstated
    UNSUPPORTED = "unsupported"   # No evidence basis (not contradicted)
    CONTRADICTED = "contradicted" # Evidence DIRECTLY opposes claim (NEW)
```

### ConfidenceLevel (NEW)

```python
class ConfidenceLevel(str, Enum):
    HIGH = "high"     # Direct quotes, exact matches (>0.85)
    MEDIUM = "medium" # Paraphrased facts (0.50-0.85)
    LOW = "low"       # Hedged, comparative, synthetic (<0.50)
```

### CorrectionType (NEW)

```python
class CorrectionType(str, Enum):
    KEEP = "keep"              # Citation is correct
    REPLACE = "replace"        # Find better citation from pool
    REMOVE = "remove"          # No valid citation exists
    ADD_ALTERNATE = "add_alternate" # Multiple valid citations
```

### ClaimType

```python
class ClaimType(str, Enum):
    GENERAL = "general"
    NUMERIC = "numeric"
```

### DerivationType

```python
class DerivationType(str, Enum):
    DIRECT = "direct"     # Quoted directly from source
    COMPUTED = "computed" # Calculated from source values
```

---

## Pydantic Models (API Layer)

### Request/Response Schemas

```python
from datetime import datetime
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field

class ClaimType(str, Enum):
    GENERAL = "general"
    NUMERIC = "numeric"

class VerificationVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"  # NEW

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class CorrectionType(str, Enum):
    KEEP = "keep"
    REPLACE = "replace"
    REMOVE = "remove"
    ADD_ALTERNATE = "add_alternate"

class DerivationType(str, Enum):
    DIRECT = "direct"
    COMPUTED = "computed"

# Response Models
class EvidenceSpanResponse(BaseModel):
    id: UUID
    source_id: UUID
    quote_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    relevance_score: float | None = None  # NEW
    has_numeric_content: bool = False  # NEW
    # Denormalized source metadata for convenience
    source_title: str | None = None
    source_url: str | None = None
    source_author: str | None = None
    source_date: str | None = None

class CitationResponse(BaseModel):
    evidence_span: EvidenceSpanResponse
    confidence_score: float | None = None
    is_primary: bool = True  # NEW

class CitationCorrectionResponse(BaseModel):  # NEW
    id: UUID
    correction_type: CorrectionType
    original_evidence: EvidenceSpanResponse | None = None
    corrected_evidence: EvidenceSpanResponse | None = None
    reasoning: str | None = None

class QAVerificationResult(BaseModel):  # NEW
    question: str
    claim_answer: str
    evidence_answer: str
    match: bool
    normalized_comparison: dict | None = None

class NumericClaimDetail(BaseModel):
    raw_value: str
    normalized_value: float | None = None
    unit: str | None = None
    entity_reference: str | None = None
    derivation_type: DerivationType
    computation_details: dict | None = None
    assumptions: dict | None = None
    qa_verification: list[QAVerificationResult] | None = None  # NEW

class ClaimResponse(BaseModel):
    id: UUID
    claim_text: str
    claim_type: ClaimType
    confidence_level: ConfidenceLevel | None = None  # NEW
    position_start: int
    position_end: int
    verification_verdict: VerificationVerdict | None = None
    verification_reasoning: str | None = None
    abstained: bool = False  # NEW
    citations: list[CitationResponse] = []
    corrections: list[CitationCorrectionResponse] = []  # NEW
    numeric_detail: NumericClaimDetail | None = None

class VerificationSummary(BaseModel):
    total_claims: int
    supported_count: int
    partial_count: int
    unsupported_count: int
    contradicted_count: int = 0  # NEW
    abstained_count: int = 0  # NEW
    unsupported_rate: float = Field(
        description="Percentage of claims that are unsupported (0.0-1.0)"
    )
    contradicted_rate: float = Field(  # NEW
        default=0.0,
        description="Percentage of claims that are contradicted (0.0-1.0)"
    )
    warning: bool = Field(
        description="True if unsupported_rate > 0.20 or contradicted_rate > 0.05"
    )

class MessageClaimsResponse(BaseModel):
    message_id: UUID
    claims: list[ClaimResponse]
    verification_summary: VerificationSummary
```

---

## SSE Event Types (UPDATED)

New streaming events for real-time claim verification:

```python
from typing import Literal

class ClaimGeneratedEvent(BaseModel):  # NEW
    """Emitted when a claim is generated during interleaved synthesis."""
    event_type: Literal["claim_generated"] = "claim_generated"
    timestamp: datetime
    claim_text: str
    position_start: int
    position_end: int
    evidence_preview: str
    confidence_level: ConfidenceLevel

class ClaimVerifiedEvent(BaseModel):
    """Emitted when a claim is verified during synthesis."""
    event_type: Literal["claim_verified"] = "claim_verified"
    timestamp: datetime
    claim_id: UUID
    claim_text: str
    position_start: int
    position_end: int
    verdict: VerificationVerdict
    confidence_level: ConfidenceLevel  # NEW
    evidence_preview: str  # Truncated quote for UI display
    reasoning: str | None = None  # NEW

class CitationCorrectedEvent(BaseModel):  # NEW
    """Emitted when a citation is corrected."""
    event_type: Literal["citation_corrected"] = "citation_corrected"
    timestamp: datetime
    claim_id: UUID
    correction_type: CorrectionType
    reasoning: str | None = None

class NumericClaimDetectedEvent(BaseModel):
    """Emitted when a numeric claim is detected."""
    event_type: Literal["numeric_claim_detected"] = "numeric_claim_detected"
    timestamp: datetime
    claim_id: UUID
    raw_value: str
    normalized_value: str | None
    unit: str | None
    derivation_type: DerivationType
    qa_verified: bool = False  # NEW: Whether QA verification passed

class VerificationSummaryEvent(BaseModel):  # UPDATED
    """Emitted when verification is complete for a message."""
    event_type: Literal["verification_summary"] = "verification_summary"
    timestamp: datetime
    message_id: UUID
    total_claims: int
    supported: int
    partial: int
    unsupported: int
    contradicted: int  # NEW
    abstained_count: int  # NEW
    citation_corrections: int  # NEW: Number of corrections made
    warning: bool
```

---

## SQLAlchemy Models

### Claim Model (UPDATED)

```python
class Claim(Base, UUIDMixin):
    """Atomic factual assertion extracted from agent response."""

    __tablename__ = "claims"

    message_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence_level: Mapped[str | None] = mapped_column(  # NEW
        String(20), nullable=True
    )
    position_start: Mapped[int] = mapped_column(nullable=False)
    position_end: Mapped[int] = mapped_column(nullable=False)
    verification_verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)
    verification_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)  # NEW

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="claims")
    citations: Mapped[list["Citation"]] = relationship(
        "Citation", back_populates="claim", cascade="all, delete-orphan"
    )
    corrections: Mapped[list["CitationCorrection"]] = relationship(  # NEW
        "CitationCorrection", back_populates="claim", cascade="all, delete-orphan"
    )
    numeric_detail: Mapped["NumericClaim | None"] = relationship(
        "NumericClaim", back_populates="claim", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "verification_verdict IN ('supported', 'partial', 'unsupported', 'contradicted')",
            name="ck_claims_verdict"
        ),
        CheckConstraint(
            "confidence_level IN ('high', 'medium', 'low')",
            name="ck_claims_confidence"
        ),
    )
```

### EvidenceSpan Model (UPDATED)

```python
class EvidenceSpan(Base, UUIDMixin):
    """Minimal text passage from source supporting claims."""

    __tablename__ = "evidence_spans"

    source_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quote_text: Mapped[str] = mapped_column(Text, nullable=False)
    start_offset: Mapped[int | None] = mapped_column(nullable=True)
    end_offset: Mapped[int | None] = mapped_column(nullable=True)
    section_heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # NEW
    has_numeric_content: Mapped[bool] = mapped_column(Boolean, default=False)  # NEW

    # Relationships
    source: Mapped["Source"] = relationship("Source", back_populates="evidence_spans")
    citations: Mapped[list["Citation"]] = relationship(
        "Citation", back_populates="evidence_span", cascade="all, delete-orphan"
    )
```

### Citation Model (UPDATED)

```python
class Citation(Base, UUIDMixin):
    """Link between claim and supporting evidence."""

    __tablename__ = "citations"

    claim_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evidence_span_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_spans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True)  # NEW

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="citations")
    evidence_span: Mapped["EvidenceSpan"] = relationship(
        "EvidenceSpan", back_populates="citations"
    )

    __table_args__ = (
        UniqueConstraint("claim_id", "evidence_span_id", name="uq_claim_evidence"),
    )
```

### NumericClaim Model (UPDATED)

```python
class NumericClaim(Base):
    """Extended metadata for numeric claims."""

    __tablename__ = "numeric_claims"

    claim_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        primary_key=True,
    )
    raw_value: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    entity_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    derivation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    computation_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    assumptions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    qa_verification: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # NEW

    # Relationship
    claim: Mapped["Claim"] = relationship("Claim", back_populates="numeric_detail")
```

### CitationCorrection Model (NEW)

```python
class CitationCorrection(Base, UUIDMixin):
    """Tracks corrections made to citations during post-processing."""

    __tablename__ = "citation_corrections"

    claim_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_evidence_span_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_spans.id", ondelete="SET NULL"),
        nullable=True,
    )
    corrected_evidence_span_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("evidence_spans.id", ondelete="SET NULL"),
        nullable=True,
    )
    correction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="corrections")
    original_evidence: Mapped["EvidenceSpan | None"] = relationship(
        "EvidenceSpan",
        foreign_keys=[original_evidence_span_id],
    )
    corrected_evidence: Mapped["EvidenceSpan | None"] = relationship(
        "EvidenceSpan",
        foreign_keys=[corrected_evidence_span_id],
    )

    __table_args__ = (
        CheckConstraint(
            "correction_type IN ('keep', 'replace', 'remove', 'add_alternate')",
            name="ck_citation_corrections_type"
        ),
    )
```

---

## Migration Script (UPDATED)

```sql
-- Migration: Add claim-level citation tables with 2024-2025 improvements
-- Depends on: messages, sources tables

-- Claims extracted from agent responses (UPDATED with new fields)
CREATE TABLE claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    claim_type VARCHAR(20) NOT NULL CHECK (claim_type IN ('general', 'numeric')),
    confidence_level VARCHAR(20) CHECK (confidence_level IN ('high', 'medium', 'low')),  -- NEW
    position_start INT NOT NULL,
    position_end INT NOT NULL,
    verification_verdict VARCHAR(20) CHECK (verification_verdict IN ('supported', 'partial', 'unsupported', 'contradicted')),  -- UPDATED: added 'contradicted'
    verification_reasoning TEXT,
    abstained BOOLEAN NOT NULL DEFAULT FALSE,  -- NEW
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_claims_message_id ON claims(message_id);
CREATE INDEX idx_claims_verdict ON claims(verification_verdict);
CREATE INDEX idx_claims_confidence ON claims(confidence_level);  -- NEW

-- Evidence spans from sources (UPDATED with new fields)
CREATE TABLE evidence_spans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    quote_text TEXT NOT NULL,
    start_offset INT,
    end_offset INT,
    section_heading VARCHAR(500),
    relevance_score FLOAT,  -- NEW
    has_numeric_content BOOLEAN NOT NULL DEFAULT FALSE,  -- NEW
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_spans_source_id ON evidence_spans(source_id);
CREATE INDEX idx_evidence_spans_relevance ON evidence_spans(relevance_score);  -- NEW

-- Claim-to-evidence links (many-to-many) (UPDATED with new field)
CREATE TABLE citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    evidence_span_id UUID NOT NULL REFERENCES evidence_spans(id) ON DELETE CASCADE,
    confidence_score FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    is_primary BOOLEAN NOT NULL DEFAULT TRUE,  -- NEW
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(claim_id, evidence_span_id)
);

CREATE INDEX idx_citations_claim_id ON citations(claim_id);
CREATE INDEX idx_citations_evidence_span_id ON citations(evidence_span_id);

-- Numeric claim metadata (1:1 extension) (UPDATED with new field)
CREATE TABLE numeric_claims (
    claim_id UUID PRIMARY KEY REFERENCES claims(id) ON DELETE CASCADE,
    raw_value TEXT NOT NULL,
    normalized_value NUMERIC,
    unit VARCHAR(50),
    entity_reference TEXT,
    derivation_type VARCHAR(20) NOT NULL CHECK (derivation_type IN ('direct', 'computed')),
    computation_details JSONB,
    assumptions JSONB,
    qa_verification JSONB  -- NEW: QAFactEval verification results
);

-- Citation corrections tracking (NEW TABLE)
CREATE TABLE citation_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    original_evidence_span_id UUID REFERENCES evidence_spans(id) ON DELETE SET NULL,
    corrected_evidence_span_id UUID REFERENCES evidence_spans(id) ON DELETE SET NULL,
    correction_type VARCHAR(20) NOT NULL CHECK (correction_type IN ('keep', 'replace', 'remove', 'add_alternate')),
    reasoning TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_citation_corrections_claim_id ON citation_corrections(claim_id);
CREATE INDEX idx_citation_corrections_type ON citation_corrections(correction_type);
```

---

## Data Integrity Rules

1. **Claim-Message Relationship**: Claims are deleted when their parent message is deleted (CASCADE)
2. **EvidenceSpan-Source Relationship**: Evidence spans are deleted when their parent source is deleted (CASCADE)
3. **Citation Uniqueness**: A claim cannot cite the same evidence span twice
4. **NumericClaim Extension**: Only claims with `claim_type = 'numeric'` should have a NumericClaim record
5. **Position Validity**: `position_start < position_end` and both must be within message content bounds
6. **Verification Caching**: Verification results are computed once during synthesis and cached in the claim record
7. **Confidence Level Assignment**: Must be set during Stage 3 (confidence classification) before verification
8. **Citation Corrections**: Only created when correction_type != 'keep'
9. **QA Verification**: Only stored for numeric claims that underwent QAFactEval verification

---

## Query Patterns

### Get all claims for a message with evidence and corrections

```sql
SELECT
    c.id, c.claim_text, c.claim_type, c.confidence_level, c.verification_verdict, c.abstained,
    es.quote_text, es.section_heading, es.relevance_score,
    s.title as source_title, s.url as source_url,
    cc.correction_type, cc.reasoning as correction_reasoning
FROM claims c
LEFT JOIN citations ct ON ct.claim_id = c.id
LEFT JOIN evidence_spans es ON es.id = ct.evidence_span_id
LEFT JOIN sources s ON s.id = es.source_id
LEFT JOIN citation_corrections cc ON cc.claim_id = c.id
WHERE c.message_id = :message_id
ORDER BY c.position_start;
```

### Get verification summary for a message (UPDATED for four-tier)

```sql
SELECT
    COUNT(*) as total_claims,
    COUNT(*) FILTER (WHERE verification_verdict = 'supported') as supported,
    COUNT(*) FILTER (WHERE verification_verdict = 'partial') as partial,
    COUNT(*) FILTER (WHERE verification_verdict = 'unsupported') as unsupported,
    COUNT(*) FILTER (WHERE verification_verdict = 'contradicted') as contradicted,  -- NEW
    COUNT(*) FILTER (WHERE abstained = TRUE) as abstained  -- NEW
FROM claims
WHERE message_id = :message_id;
```

### Get citation correction metrics

```sql
SELECT
    correction_type,
    COUNT(*) as count
FROM citation_corrections cc
JOIN claims c ON c.id = cc.claim_id
WHERE c.message_id = :message_id
GROUP BY correction_type;
```

### Get numeric claims with QA verification results

```sql
SELECT
    c.id, c.claim_text, c.position_start, c.position_end, c.confidence_level,
    nc.raw_value, nc.normalized_value, nc.unit, nc.derivation_type,
    nc.qa_verification,
    es.quote_text
FROM claims c
JOIN numeric_claims nc ON nc.claim_id = c.id
LEFT JOIN citations ct ON ct.claim_id = c.id AND ct.is_primary = TRUE
LEFT JOIN evidence_spans es ON es.id = ct.evidence_span_id
WHERE c.message_id = :message_id
  AND c.claim_type = 'numeric';
```

### Get claims by confidence level for routing analysis

```sql
SELECT
    confidence_level,
    COUNT(*) as claim_count,
    AVG(CASE WHEN verification_verdict = 'supported' THEN 1.0 ELSE 0.0 END) as support_rate
FROM claims
WHERE message_id = :message_id
GROUP BY confidence_level;
```
