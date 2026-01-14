"""Citation verification prompts for the 6-stage pipeline.

This module contains LLM prompts for each stage of the citation verification pipeline:
- evidence_selection.py - Stage 1: Evidence Pre-Selection
- interleaved_generation.py - Stage 2: Interleaved Generation
- confidence.py - Stage 3: Confidence Classification
- verification.py - Stage 4: Isolated Verification
- correction.py - Stage 5: Citation Correction
- numeric_qa.py - Stage 6: Numeric QA Verification
- claim_extraction.py - Atomic claim decomposition
"""

# Prompts will be imported here as they are implemented
# from src.agent.prompts.citation.evidence_selection import EVIDENCE_PRESELECTION_PROMPT
# from src.agent.prompts.citation.interleaved_generation import INTERLEAVED_GENERATION_PROMPT
# from src.agent.prompts.citation.confidence import CONFIDENCE_CLASSIFICATION_PROMPT
# from src.agent.prompts.citation.verification import ISOLATED_VERIFICATION_PROMPT
# from src.agent.prompts.citation.correction import CITATION_CORRECTION_PROMPT
# from src.agent.prompts.citation.numeric_qa import NUMERIC_DETECTION_PROMPT, QA_GENERATION_PROMPT
# from src.agent.prompts.citation.claim_extraction import ATOMIC_CLAIM_EXTRACTION_PROMPT

__all__: list[str] = [
    # Will export prompts as they are implemented
]
