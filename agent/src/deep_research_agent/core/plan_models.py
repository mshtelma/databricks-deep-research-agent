"""
Plan and Step data models for multi-agent research system.

Based on deer-flow implementation patterns for structured planning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Iterable, Mapping, MutableMapping, Sequence, Set
from pydantic import BaseModel, Field
from datetime import datetime
from . import get_logger
from .id_generator import PlanIDGenerator
from .template_generator import DynamicSection


logger = get_logger(__name__)




class StepType(str, Enum):
    """Types of steps in a research plan."""
    RESEARCH = "research"
    VALIDATION = "validation"


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(BaseModel):
    """Individual step in a research plan."""
    
    step_id: str = Field(description="Unique identifier for the step")
    title: str = Field(description="Brief title of the step")
    description: str = Field(description="Detailed description of what to accomplish")
    step_type: StepType = Field(description="Type of the step")
    status: StepStatus = Field(default=StepStatus.PENDING)
    
    # Execution details
    need_search: bool = Field(default=True, description="Whether this step requires web search")
    search_queries: Optional[List[str]] = Field(default=None, description="Specific search queries for this step")
    
    # Dependencies and context
    depends_on: Optional[List[str]] = Field(default=None, description="Step IDs this step depends on")
    required_context: Optional[List[str]] = Field(default=None, description="Required context from previous steps")
    requirement_mapping: Optional[List[str]] = Field(default=None, description="Requirements this step addresses")
    
    # Step ordering and section linkage (new fields for fixing infinite loop)
    execution_order: Optional[int] = Field(default=None, description="Order in which this step should be executed")
    template_section_title: Optional[str] = Field(default=None, description="Dynamic section title associated with this step, if any")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the step")
    
    # Results
    execution_result: Optional[str] = Field(default=None, description="Result from executing this step")
    observations: Optional[List[str]] = Field(default=None, description="Observations made during execution")
    citations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Citations collected in this step")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(default=None, description="Confidence in step completion (0-1)")
    grounding_score: Optional[float] = Field(default=None, description="How well grounded the results are (0-1)")
    
    # Timing
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "step_id": "step_001",
                    "title": "Market Analysis Research",
                    "description": "Gather comprehensive data on current AI market trends, size, and growth projections",
                    "step_type": "research",
                    "need_search": True,
                    "search_queries": [
                        "AI market size 2024",
                        "artificial intelligence industry growth rate",
                        "top AI companies market share"
                    ]
                }
            ]
        }


class PlanQuality(BaseModel):
    """Quality assessment of a research plan."""
    
    completeness_score: float = Field(description="How complete the plan is (0-1)")
    feasibility_score: float = Field(description="How feasible the plan is (0-1)")
    clarity_score: float = Field(description="How clear and well-defined the plan is (0-1)")
    coverage_score: float = Field(description="How well the plan covers the research topic (0-1)")
    
    overall_score: float = Field(description="Overall quality score (0-1)")
    issues: List[str] = Field(default_factory=list, description="Identified issues with the plan")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    
    def calculate_overall_score(self) -> float:
        """Calculate overall score from individual metrics."""
        scores = [
            self.completeness_score,
            self.feasibility_score,
            self.clarity_score,
            self.coverage_score
        ]
        return sum(scores) / len(scores)


class Plan(BaseModel):
    """Research plan with structured steps."""
    
    # Core plan information
    plan_id: str = Field(description="Unique identifier for the plan")
    title: str = Field(description="Title of the research plan")
    research_topic: str = Field(description="Original research topic/question")
    
    # Planning metadata
    thought: str = Field(description="Planning reasoning and approach")
    has_enough_context: bool = Field(default=False, description="Whether existing context is sufficient")
    needs_background_investigation: bool = Field(default=True, description="Whether background investigation is needed")
    
    # Steps
    steps: List[Step] = Field(default_factory=list, description="Ordered list of plan steps")
    
    # Plan metadata
    iteration: int = Field(default=0, description="Plan iteration number")
    quality_assessment: Optional[PlanQuality] = Field(default=None, description="Quality assessment of the plan")
    
    # Report structure (for adaptive styling)
    suggested_report_structure: Optional[List[str]] = Field(default=None, description="Suggested section names for the report")
    structure_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata about structure generation")
    
    dynamic_sections: List[DynamicSection] = Field(default_factory=list, description="Dynamic section descriptors for template generation")
    report_template: Optional[str] = Field(default=None, description="Pre-rendered markdown template to guide report generation")
    
    # Presentation requirements (for intelligent table/format decisions)
    presentation_requirements: Optional[Dict[str, Any]] = Field(default=None, description="Requirements for optimal presentation format")
    
    # Entity validation
    requested_entities: List[str] = Field(default_factory=list, description="Entities (countries, organizations) that should be mentioned in the research")
    
    # Execution tracking
    current_step_index: int = Field(default=0, description="Index of currently executing step")
    completed_steps: int = Field(default=0, description="Number of completed steps")
    failed_steps: int = Field(default=0, description="Number of failed steps")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Requirement integration (new)
    extracted_requirements: Optional[Any] = Field(default=None, description="Extracted RequirementSet from instructions")
    requirement_confidence: Optional[float] = Field(default=None, description="Confidence in requirement extraction")
    success_criteria: Optional[List[str]] = Field(default=None, description="Success criteria from requirements")
    complexity_assessment: Optional[str] = Field(default=None, description="Complexity level: simple/moderate/complex")
    estimated_total_steps: Optional[int] = Field(default=None, description="Estimated steps from requirement analysis")
    output_requirements: Optional[Dict[str, Any]] = Field(default=None, description="Legacy output requirements (deprecated)")
    
    def get_next_step(self) -> Optional[Step]:
        """Get the next pending step to execute using dependency-aware selection."""
        # Get completed step IDs for dependency checking
        completed_step_ids = {step.step_id for step in self.steps if step.status == StepStatus.COMPLETED}
        
        # Also get permanently failed step IDs to skip them
        failed_step_ids = {step.step_id for step in self.steps if step.status == StepStatus.FAILED}
        
        for step in self.steps:
            # Skip permanently failed steps
            if step.status == StepStatus.FAILED:
                # Check if this is marked as permanently failed (after max retries)
                if hasattr(step, 'metadata') and step.metadata and step.metadata.get('permanent_failure', False):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Skipping permanently failed step {step.step_id}")
                    continue
                    
                # Check retry count from metadata
                retry_count = step.metadata.get('retry_count', 0) if hasattr(step, 'metadata') and step.metadata else 0
                max_retries = step.metadata.get('max_retries', 3) if hasattr(step, 'metadata') and step.metadata else 3
                
                if retry_count >= max_retries:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Step {step.step_id} reached max retries ({retry_count}/{max_retries}), skipping")
                    continue
                    
                # Otherwise, retry the failed step if not at max retries yet
                # This allows retrying failed steps up to the circuit breaker limit
                
            if step.status == StepStatus.PENDING:
                # Check if dependencies are met using proper title-to-ID mapping
                if hasattr(step, 'depends_on') and step.depends_on:
                    deps_met = True

                    for dependency in step.depends_on:
                        dep_step = self.get_step_by_id(dependency)
                        # Accept both COMPLETED and FAILED as "done" - allow proceeding with partial data
                        if dep_step and dep_step.status in (StepStatus.COMPLETED, StepStatus.FAILED):
                            continue

                        # Allow title-based dependency matching for backward compatibility
                        matched = False
                        for candidate_step in self.steps:
                            # Accept both COMPLETED and FAILED as "done"
                            if candidate_step.status in (StepStatus.COMPLETED, StepStatus.FAILED) and self._titles_similar(dependency, candidate_step.title):
                                matched = True
                                break

                        if not matched:
                            deps_met = False
                            break

                    if not deps_met:
                        continue

                return step
        return None
    
    def get_step_by_id(self, step_id: str) -> Optional[Step]:
        """Get a step by its ID."""
        normalized_target = PlanIDGenerator.normalize_id(step_id)
        for step in self.steps:
            if PlanIDGenerator.normalize_id(step.step_id) == normalized_target:
                return step
        return None

    def mark_step_completed(
        self,
        step_id: str,
        *,
        execution_result: Optional[str] = None,
        observations: Optional[Iterable[str]] = None,
        citations: Optional[Iterable[Any]] = None,
        event_emitter: Optional[Any] = None,
    ) -> Step:
        """Mark the specified step as completed and update execution metadata."""

        step = self.get_step_by_id(step_id)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found in plan {self.plan_id}")

        now = datetime.now()
        if step.started_at is None:
            step.started_at = now
        step.completed_at = now
        step.status = StepStatus.COMPLETED

        if execution_result is not None:
            step.execution_result = execution_result

        if observations is not None:
            step.observations = list(observations)

        if citations is not None:
            step.citations = [
                citation.to_dict() if hasattr(citation, "to_dict") else dict(citation)
                for citation in citations
            ]

        self._refresh_step_counters()
        
        # Emit STEP_COMPLETED event for real-time UI updates
        logger.info(f"🔍 DEBUG: mark_step_completed called for {step_id}, event_emitter={event_emitter is not None}, has_emit={hasattr(event_emitter, 'emit') if event_emitter else False}")
        if event_emitter and hasattr(event_emitter, 'emit'):
            try:
                from .event_emitter import IntermediateEventType
                logger.info(f"🔍 DEBUG: About to emit STEP_COMPLETED for {step_id}")
                event_emitter.emit(
                    event_type=IntermediateEventType.STEP_COMPLETED,
                    data={
                        "step_id": step_id,
                        "status": step.status,
                        "result": step.execution_result or "",
                        "description": step.description,
                        "progress": {
                            "completed": self.completed_count,
                            "total": len(self.steps),
                            "percentage": (self.completed_count / len(self.steps)) * 100 if self.steps else 0
                        }
                    },
                    correlation_id=f"step_{step_id}",
                    stage_id="researcher"
                )
                logger.info(f"✅ Successfully emitted STEP_COMPLETED event for step {step_id}")
            except Exception as e:
                logger.error(f"❌ Failed to emit STEP_COMPLETED event for step {step_id}: {e}")
        else:
            logger.warning(f"⚠️ No event emitter available for step {step_id} completion")
        
        return step

    def mark_step_failed(self, step_id: str, *, reason: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None, event_emitter: Optional[Any] = None) -> Step:
        """Mark the specified step as failed with optional metadata.
        
        Args:
            step_id: ID of the step to mark as failed
            reason: Optional failure reason
            metadata: Optional metadata including retry info and permanent_failure flag
            event_emitter: Optional event emitter for real-time UI updates
        """

        step = self.get_step_by_id(step_id)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found in plan {self.plan_id}")

        now = datetime.now()
        if step.started_at is None:
            step.started_at = now
        step.completed_at = now
        step.status = StepStatus.FAILED
        if reason:
            step.execution_result = reason

        # Store metadata including permanent_failure flag and retry info
        if metadata:
            if not hasattr(step, 'metadata') or step.metadata is None:
                step.metadata = {}
            step.metadata.update(metadata)
            
            # Log retry information if present
            import logging
            logger = logging.getLogger(__name__)
            if 'permanent_failure' in metadata:
                logger.info(f"Step {step_id} marked with permanent_failure={metadata['permanent_failure']}")
            if 'retry_count' in metadata:
                logger.info(f"Step {step_id} retry count: {metadata['retry_count']}")

        self._refresh_step_counters()
        
        # Emit STEP_FAILED event for real-time UI updates
        if event_emitter and hasattr(event_emitter, 'emit'):
            try:
                from .event_emitter import IntermediateEventType
                event_emitter.emit(
                    event_type=IntermediateEventType.STEP_FAILED,
                    data={
                        "step_id": step_id,
                        "status": step.status,
                        "result": reason or "",
                        "description": step.description,
                        "progress": {
                            "failed": self.failed_count,
                            "total": len(self.steps),
                            "percentage": (self.completed_count / len(self.steps)) * 100 if self.steps else 0
                        }
                    },
                    correlation_id=f"step_{step_id}",
                    stage_id="researcher"
                )
                logger.info(f"Emitted STEP_FAILED event for step {step_id}")
            except Exception as e:
                logger.warning(f"Failed to emit STEP_FAILED event for step {step_id}: {e}")
        
        return step

    def mark_step_activated(self, step_id: str, *, event_emitter: Optional[Any] = None) -> Step:
        """Mark the specified step as activated (in progress) and emit event for real-time UI updates."""
        
        step = self.get_step_by_id(step_id)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found in plan {self.plan_id}")

        # Only update if not already in progress or completed
        if step.status == StepStatus.PENDING:
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.now()
            
            # Emit STEP_ACTIVATED event for real-time UI updates
            if event_emitter and hasattr(event_emitter, 'emit'):
                try:
                    from .event_emitter import IntermediateEventType
                    event_emitter.emit(
                        event_type=IntermediateEventType.STEP_ACTIVATED,
                        data={
                            "step_id": step_id,
                            "status": step.status,
                            "result": "",
                            "description": step.description,
                            "progress": {
                                "in_progress": len([s for s in self.steps if s.status == StepStatus.IN_PROGRESS]),
                                "total": len(self.steps)
                            }
                        },
                        correlation_id=f"step_{step_id}",
                        stage_id="researcher"
                    )
                    logger.info(f"Emitted STEP_ACTIVATED event for step {step_id}")
                except Exception as e:
                    logger.warning(f"Failed to emit STEP_ACTIVATED event for step {step_id}: {e}")
        
        return step

    _METADATA_ID_LIST_KEYS: Set[str] = {
        "depends_on",
        "required_context",
        "requirement_mapping",
        "section_dependencies",
        "source_step_ids",
        "upstream_steps",
        "downstream_steps",
    }

    _METADATA_ID_STRING_KEYS: Set[str] = {
        "source_step_id",
        "target_step_id",
        "section_id",
        "origin_step_id",
        "destination_step_id",
        "next_step_id",
        "previous_step_id",
    }

    def renumber_steps(self, id_mapping: Mapping[str, str]) -> Dict[str, str]:
        """Apply a step ID remapping and keep plan invariants consistent."""

        if not id_mapping:
            return {}

        normalized_mapping: Dict[str, str] = {}
        seen_new_ids: Set[str] = set()

        for old_id, new_id in id_mapping.items():
            if not old_id or not new_id:
                continue

            normalized_old = PlanIDGenerator.normalize_id(old_id)
            normalized_new = PlanIDGenerator.normalize_id(new_id)

            normalized_mapping[old_id] = normalized_new
            normalized_mapping[normalized_old] = normalized_new

            if normalized_new in seen_new_ids:
                raise ValueError(
                    f"Duplicate target step ID '{normalized_new}' detected while renumbering plan {self.plan_id}"
                )
            seen_new_ids.add(normalized_new)

        if not normalized_mapping:
            return {}

        for step in self.steps:
            old_id = step.step_id
            normalized_old_id = PlanIDGenerator.normalize_id(old_id) if old_id else old_id

            metadata_map: MutableMapping[str, Any]
            if step.metadata is None:
                step.metadata = {}

            if isinstance(step.metadata, MutableMapping):
                metadata_map = step.metadata
            else:
                metadata_map = dict(step.metadata)
                step.metadata = metadata_map

            is_section_step = bool(metadata_map.get("is_section_step", False))

            if not is_section_step:
                new_id = normalized_mapping.get(old_id) or normalized_mapping.get(normalized_old_id)
                if new_id and new_id != old_id:
                    metadata_map.setdefault("original_step_id", old_id)
                    logger.debug(
                        "Renumbering plan step",
                        plan_id=self.plan_id,
                        old_id=old_id,
                        new_id=new_id,
                        step_title=step.title,
                    )
                    step.step_id = new_id
                    normalized_old_id = PlanIDGenerator.normalize_id(new_id)

            self._remap_step_references(step, normalized_mapping)

        self._refresh_step_counters()

        # Return the effective mapping using normalized keys for callers that need it
        return {
            PlanIDGenerator.normalize_id(old_id): PlanIDGenerator.normalize_id(new_id)
            for old_id, new_id in id_mapping.items()
            if old_id and new_id
        }

    def _refresh_step_counters(self) -> None:
        """Recalculate aggregate counters after step mutations."""
        self.completed_steps = sum(
            1 for candidate in self.steps if candidate.status == StepStatus.COMPLETED
        )
        self.failed_steps = sum(
            1 for candidate in self.steps if candidate.status == StepStatus.FAILED
        )

    def _remap_step_references(
        self,
        step: Step,
        mapping: Mapping[str, str],
    ) -> None:
        """Update step references that may point to renamed IDs."""

        if step.depends_on:
            step.depends_on = self._remap_id_list(step.depends_on, mapping)

        if step.required_context:
            step.required_context = self._remap_id_list(step.required_context, mapping)

        if step.requirement_mapping:
            step.requirement_mapping = self._remap_id_list(step.requirement_mapping, mapping)

        if step.metadata:
            self._remap_metadata_dict(step.metadata, mapping)

    def _remap_id(self, candidate: str, mapping: Mapping[str, str]) -> str:
        if not candidate:
            return candidate

        normalized_candidate = PlanIDGenerator.normalize_id(candidate)
        return (
            mapping.get(candidate)
            or mapping.get(normalized_candidate)
            or candidate
        )

    def _remap_id_list(
        self,
        values: Iterable[str],
        mapping: Mapping[str, str],
    ) -> List[str]:
        return [self._remap_id(value, mapping) for value in values]

    def _remap_metadata_dict(
        self,
        metadata: MutableMapping[str, Any],
        mapping: Mapping[str, str],
    ) -> None:
        for key in list(metadata.keys()):
            value = metadata[key]

            if key in self._METADATA_ID_LIST_KEYS and isinstance(value, list):
                metadata[key] = [
                    self._remap_id(item, mapping) if isinstance(item, str) else item
                    for item in value
                ]
                continue

            if key in self._METADATA_ID_STRING_KEYS and isinstance(value, str):
                metadata[key] = self._remap_id(value, mapping)
                continue

            if isinstance(value, MutableMapping):
                self._remap_metadata_dict(value, mapping)
                continue

            if isinstance(value, list):
                updated_items: List[Any] = []
                for item in value:
                    if isinstance(item, MutableMapping):
                        self._remap_metadata_dict(item, mapping)
                        updated_items.append(item)
                    elif isinstance(item, str) and key in self._METADATA_ID_LIST_KEYS:
                        updated_items.append(self._remap_id(item, mapping))
                    else:
                        updated_items.append(item)
                metadata[key] = updated_items
    
    def get_completed_context(self) -> List[str]:
        """Get accumulated context from completed steps."""
        context = []
        for step in self.steps:
            if step.status == StepStatus.COMPLETED and step.observations:
                context.extend(step.observations)
        return context
    
    def is_complete(self) -> bool:
        """Check if the plan execution is complete."""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED]
            for step in self.steps
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of plan execution."""
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "total_steps": len(self.steps),
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "completion_percentage": (self.completed_steps / len(self.steps) * 100) if self.steps else 0,
            "is_complete": self.is_complete(),
            "quality_score": self.quality_assessment.overall_score if self.quality_assessment else None
        }
    
    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar enough to be considered a match."""
        # Normalize titles for comparison
        norm1 = title1.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
        norm2 = title2.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
        
        # Exact match after normalization
        if norm1 == norm2:
            return True
        
        # Substring match (one contains the other, minimum 4 chars)
        if len(norm1) >= 4 and len(norm2) >= 4:
            if norm1 in norm2 or norm2 in norm1:
                return True
        
        # Word overlap similarity (at least 50% common words)
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if words1 and words2:
            overlap = len(words1 & words2)
            total_unique_words = len(words1 | words2)
            similarity = overlap / total_unique_words if total_unique_words > 0 else 0
            return similarity >= 0.5
        
        return False

    # ========================================================================
    # Dynamic Step Management Methods for Incremental Research Loops
    # ========================================================================

    def add_step_after_index(self, index: int, new_step: Step, *, event_emitter: Optional[Any] = None) -> bool:
        """Insert a new step at the specified index for incremental research loops."""
        try:
            # Validate index
            if index < 0 or index > len(self.steps):
                logger.warning(f"Invalid index {index} for adding step to plan with {len(self.steps)} steps")
                return False

            # Ensure step has unique ID
            if any(step.step_id == new_step.step_id for step in self.steps):
                # Generate new unique ID
                new_step.step_id = PlanIDGenerator.generate_step_id()

            # Insert the new step
            self.steps.insert(index, new_step)

            # Update step counters
            self._refresh_step_counters()

            logger.info(f"[DYNAMIC PLAN] Added new step '{new_step.title}' at index {index}")

            # Emit step addition event with proper enum type
            if event_emitter:
                from .event_emitter import IntermediateEventType
                event_emitter.emit(
                    event_type=IntermediateEventType.STEP_ADDED,
                    data={
                        "step_id": new_step.step_id,
                        "step_title": new_step.title,
                        "index": index,
                        "reason": "incremental_research_loop",
                        "description": new_step.description
                    },
                    reasoning=f"Added new step for deeper investigation: {new_step.title}",
                    stage_id="planner"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to add step at index {index}: {e}")
            return False

    def modify_pending_step(self, step_id: str, updates: Dict[str, Any], *, event_emitter: Optional[Any] = None) -> bool:
        """Modify an unexecuted step based on new insights from research loops."""
        step = self.get_step_by_id(step_id)

        if not step:
            logger.warning(f"Step {step_id} not found for modification")
            return False

        if step.status not in [StepStatus.PENDING]:
            logger.warning(f"Cannot modify step {step_id} with status {step.status} - only pending steps can be modified")
            return False

        try:
            # Apply updates
            original_values = {}
            for field, value in updates.items():
                if hasattr(step, field):
                    original_values[field] = getattr(step, field)
                    setattr(step, field, value)

            logger.info(f"[DYNAMIC PLAN] Modified pending step '{step.title}' - updated {list(updates.keys())}")

            # Emit step modification event
            if event_emitter:
                event_emitter.emit(
                    event_type="step_modified",
                    data={
                        "step_id": step_id,
                        "step_title": step.title,
                        "modifications": list(updates.keys()),
                        "reason": "incremental_research_insights"
                    },
                    reasoning=f"Modified step based on new research insights: {step.title}",
                    stage_id="planner"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to modify step {step_id}: {e}")
            return False

    def mark_step_obsolete(self, step_id: str, reason: str, *, event_emitter: Optional[Any] = None) -> bool:
        """Mark a pending step as obsolete and skip it."""
        step = self.get_step_by_id(step_id)

        if not step:
            logger.warning(f"Step {step_id} not found for marking obsolete")
            return False

        if step.status not in [StepStatus.PENDING]:
            logger.warning(f"Cannot mark step {step_id} as obsolete - status is {step.status}")
            return False

        try:
            step.status = StepStatus.SKIPPED

            # Add obsolete reason to metadata
            if not hasattr(step, 'metadata') or step.metadata is None:
                step.metadata = {}
            step.metadata['obsolete_reason'] = reason
            step.metadata['marked_obsolete_at'] = datetime.now().isoformat()

            self._refresh_step_counters()

            logger.info(f"[DYNAMIC PLAN] Marked step '{step.title}' as obsolete: {reason}")

            # Emit step obsolete event
            if event_emitter:
                event_emitter.emit(
                    event_type="step_marked_obsolete",
                    data={
                        "step_id": step_id,
                        "step_title": step.title,
                        "reason": reason
                    },
                    reasoning=f"Step no longer needed due to research insights: {reason}",
                    stage_id="planner"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to mark step {step_id} as obsolete: {e}")
            return False

    def get_execution_context(self, current_step_id: str) -> Dict[str, Any]:
        """Get context from completed steps for current step execution."""
        completed_steps = [s for s in self.steps if s.status == StepStatus.COMPLETED]

        context = {
            "completed_count": len(completed_steps),
            "total_steps": len(self.steps),
            "key_findings": [],
            "accumulated_observations": [],
            "completed_step_titles": [s.title for s in completed_steps],
            "entity_mentions": set(),
            "source_types": set()
        }

        # Extract findings and observations from completed steps
        for step in completed_steps:
            if hasattr(step, 'observations') and step.observations:
                context["accumulated_observations"].extend(step.observations)

                # Extract key findings (first sentence of each observation)
                for obs in step.observations:
                    obs_text = str(obs)
                    first_sentence = obs_text.split('.')[0]
                    if len(first_sentence) > 20:  # Meaningful findings
                        context["key_findings"].append(first_sentence + '.')

            # Extract execution results as findings too
            if hasattr(step, 'execution_result') and step.execution_result:
                result_text = str(step.execution_result)
                if len(result_text) > 50:
                    context["key_findings"].append(result_text[:200] + '...')

        # Limit findings to most important ones
        context["key_findings"] = context["key_findings"][-10:]  # Last 10 findings
        context["accumulated_observations"] = context["accumulated_observations"][-20:]  # Last 20 observations

        return context

    def add_incremental_steps_from_gaps(self, gap_analysis: Dict[str, Any], *, event_emitter: Optional[Any] = None) -> int:
        """Add new steps based on gap analysis from research loops."""
        steps_added = 0

        # Add steps for knowledge gaps
        for gap in gap_analysis.get("knowledge_gaps", [])[:3]:  # Top 3 gaps
            if "Missing" in gap:
                aspect = gap.split("Missing ")[1].split(" perspective")[0]
                new_step = Step(
                    step_id=PlanIDGenerator.generate_step_id(),
                    title=f"Research {aspect} aspects",
                    description=f"Investigate {aspect} perspective on {self.research_topic}",
                    step_type=StepType.RESEARCH,
                    need_search=True,
                    search_queries=[f"{self.research_topic} {aspect} analysis", f"{aspect} impact {self.research_topic}"]
                )

                # Add after existing research steps
                insert_index = len([s for s in self.steps if s.step_type == StepType.RESEARCH])
                if self.add_step_after_index(insert_index, new_step, event_emitter=event_emitter):
                    steps_added += 1

        # Add steps for deep dive topics
        for topic in gap_analysis.get("deep_dive_topics", [])[:2]:  # Top 2 deep dives
            if "Deeper analysis of" in topic:
                subject = topic.replace("Deeper analysis of ", "")
                new_step = Step(
                    step_id=PlanIDGenerator.generate_step_id(),
                    title=f"Deep dive: {subject}",
                    description=f"Comprehensive analysis of {subject} including technical details and recent developments",
                    step_type=StepType.RESEARCH,
                    need_search=True,
                    search_queries=[f"{subject} technical specifications", f"{subject} recent developments 2024"]
                )

                # Add near the end of research steps
                insert_index = len([s for s in self.steps if s.step_type == StepType.RESEARCH])
                if self.add_step_after_index(insert_index, new_step, event_emitter=event_emitter):
                    steps_added += 1

        # Add verification step if needed
        if gap_analysis.get("verification_needed") and steps_added > 0:
            verification_step = Step(
                step_id=PlanIDGenerator.generate_step_id(),
                title="Fact verification",
                description="Verify controversial claims and cross-reference findings with authoritative sources",
                step_type=StepType.VALIDATION,
                need_search=True,
                search_queries=["scientific evidence", "peer reviewed sources", "expert consensus"]
            )

            # Add at the end (SYNTHESIS steps no longer exist per Phase 1)
            # Verification should happen after all research but before reporter generates final report
            if self.add_step_after_index(len(self.steps), verification_step, event_emitter=event_emitter):
                steps_added += 1

        logger.info(f"[DYNAMIC PLAN] Added {steps_added} new steps based on gap analysis")
        return steps_added

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "plan_id": "plan_001",
                    "title": "Comprehensive AI Market Research",
                    "research_topic": "What are the current trends in the AI market?",
                    "thought": "To understand AI market trends, we need to gather data on market size, key players, emerging technologies, and future projections.",
                    "has_enough_context": False,
                    "steps": [
                        {
                            "step_id": "step_001",
                            "title": "Current Market Analysis",
                            "description": "Research current AI market size and growth",
                            "step_type": "research",
                            "need_search": True
                        },
                        {
                            "step_id": "step_002",
                            "title": "Technology Trends",
                            "description": "Identify emerging AI technologies and innovations",
                            "step_type": "research",
                            "need_search": True
                        }
                    ]
                }
            ]
        }


class PlanFeedback(BaseModel):
    """Human or automatic feedback on a plan."""
    
    feedback_type: str = Field(description="Type of feedback: human, automatic, quality_check")
    feedback: str = Field(description="The feedback content")
    suggestions: List[str] = Field(default_factory=list, description="Specific suggestions for improvement")
    requires_revision: bool = Field(default=False, description="Whether the plan needs revision")
    approved: bool = Field(default=False, description="Whether the plan is approved for execution")
    timestamp: datetime = Field(default_factory=datetime.now)
