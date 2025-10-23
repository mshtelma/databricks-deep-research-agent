"""
State capture infrastructure integrated into workflow.

This module provides a singleton StateCapture class that agents can use to
save their state to fixtures during test capture mode.

Usage:
    # In workflow nodes:
    from deep_research_agent.core.state_capture import state_capture

    state_capture.capture_if_enabled("reporter", state, "before")

    # To enable capture:
    CAPTURE_STATE=true python tests/capture/run_capture.py
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class StateCapture:
    """
    Singleton state capturer that agents can use.

    When CAPTURE_STATE=true, captures lightweight state snapshots at agent boundaries.
    Saves only essential fields to keep fixtures manageable (10-50KB each).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.capture_enabled = os.getenv("CAPTURE_STATE", "false").lower() == "true"
            self.capture_dir = Path(os.getenv("CAPTURE_DIR", "tests/fixtures/states"))
            self.current_prompt_id = None
            self.current_prompt = None
            self.captured_states = {}
            self.initialized = True

    def set_prompt_context(self, prompt_id: str, prompt: str):
        """
        Set the current prompt being processed.

        Args:
            prompt_id: Unique identifier for this prompt (e.g., "simple_fact")
            prompt: The actual user prompt text
        """
        self.current_prompt_id = prompt_id
        self.current_prompt = prompt
        self.captured_states = {}

    def capture_if_enabled(
        self,
        agent_name: str,
        state: Dict[str, Any],
        phase: str = "after"
    ) -> None:
        """
        Capture state if capture mode is enabled.

        Args:
            agent_name: Name of the agent (coordinator, planner, researcher, fact_checker, reporter)
            state: Current state dict
            phase: "before" or "after" the agent runs, or "error" for error states
        """
        if not self.capture_enabled:
            return

        if not self.current_prompt_id:
            print(f"⚠️ Warning: No prompt_id set for state capture")
            return

        # Create a lightweight copy (only essential fields for this agent)
        captured = self._extract_essential_state(agent_name, state)

        # Store in memory first
        key = f"{agent_name}_{phase}"
        self.captured_states[key] = captured

        # Save to file
        self._save_state(agent_name, phase, captured)

    def _extract_essential_state(
        self,
        agent_name: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract only the fields this agent needs.

        This is CRITICAL for keeping fixtures manageable.
        We only save what's necessary for testing each specific agent.

        Args:
            agent_name: Name of the agent
            state: Full state dict

        Returns:
            Filtered state dict with only essential fields
        """
        # Common fields all agents need
        essential = {
            "research_topic": state.get("research_topic"),
            "current_agent": agent_name,
            "errors": state.get("errors", [])[-5:],  # Last 5 errors only
            "warnings": state.get("warnings", [])[-5:],  # Last 5 warnings only
        }

        # Agent-specific fields
        if agent_name == "reporter":
            essential.update({
                "observations": self._extract_observations(state.get("observations", [])),
                "current_plan": self._simplify_plan(state.get("current_plan")),
                "completed_steps": self._simplify_steps(state.get("completed_steps", [])),
                "citations": state.get("citations", [])[:20],  # Limit to 20 citations
                "factuality_report": self._simplify_factuality_report(state.get("factuality_report")),
                "factuality_score": state.get("factuality_score"),
                "report_style": str(state.get("report_style")) if state.get("report_style") else None,
                "section_research_results": state.get("section_research_results"),
                "enable_grounding": state.get("enable_grounding"),
                "citation_style": state.get("citation_style"),
                "query_constraints": self._simplify_query_constraints(state.get("query_constraints")),  # CRITICAL: Needed for hybrid planner decision
                "unified_plan": self._simplify_unified_plan(state.get("unified_plan")),  # CRITICAL FIX: Needed for Tier 1 Hybrid mode
                "calculation_results": state.get("calculation_results"),  # CRITICAL FIX: Output from executing unified_plan
            })

        elif agent_name == "planner":
            essential.update({
                "background_investigation_results": state.get("background_investigation_results"),  # Full content
                "current_plan": self._simplify_plan(state.get("current_plan")),
                "plan_iterations": state.get("plan_iterations"),
                "plan_feedback": state.get("plan_feedback"),
                "enable_iterative_planning": state.get("enable_iterative_planning"),
                "max_plan_iterations": state.get("max_plan_iterations"),
                "query_constraints": self._simplify_query_constraints(state.get("query_constraints")),  # CRITICAL: Created by planner, needed by reporter
            })

        elif agent_name == "researcher":
            essential.update({
                "current_plan": self._simplify_plan(state.get("current_plan")),
                "current_step": state.get("current_step"),
                "current_step_index": state.get("current_step_index"),
                "observations": self._extract_observations(state.get("observations", [])),  # ALL observations
                "search_results": self._simplify_search_results(state.get("search_results", [])),
                "search_queries": state.get("search_queries", []),  # ALL queries
                "research_loops": state.get("research_loops"),
                "max_research_loops": state.get("max_research_loops"),
            })

        elif agent_name == "fact_checker":
            essential.update({
                "observations": self._extract_observations(state.get("observations", [])),
                "enable_grounding": state.get("enable_grounding"),
                "verification_level": str(state.get("verification_level")) if state.get("verification_level") else None,
                "grounding_results": state.get("grounding_results"),
                "factuality_report": self._simplify_factuality_report(state.get("factuality_report")),
                "contradictions": state.get("contradictions"),
            })

        elif agent_name == "coordinator":
            # Coordinator needs minimal state
            essential.update({
                "messages": self._extract_recent_messages(state.get("messages", [])),
            })

        return essential

    def _extract_observations(self, observations: List) -> List[Dict]:
        """Extract observations, converting objects to dicts - NO TRUNCATION."""
        extracted = []
        for obs in observations:
            if isinstance(obs, dict):
                # Already a dict, keep ALL fields including full_content
                extracted.append({
                    "content": obs.get("content"),  # Full content
                    "full_content": obs.get("full_content"),  # CRITICAL: Preserve enriched web content
                    "entity_tags": obs.get("entity_tags", []),  # All tags
                    "metric_values": obs.get("metric_values", {}),
                    "confidence": obs.get("confidence"),
                    "step_id": obs.get("step_id"),
                    "section_title": obs.get("section_title"),
                })
            elif hasattr(obs, 'to_dict'):
                # StructuredObservation object
                obs_dict = obs.to_dict()
                extracted.append({
                    "content": obs_dict.get("content"),  # Full content
                    "full_content": obs_dict.get("full_content"),  # CRITICAL: Preserve enriched web content
                    "entity_tags": obs_dict.get("entity_tags", []),  # All tags
                    "metric_values": obs_dict.get("metric_values", {}),
                    "confidence": obs_dict.get("confidence"),
                    "step_id": obs_dict.get("step_id"),
                    "section_title": obs_dict.get("section_title"),
                })
            else:
                # String observation
                extracted.append({
                    "content": str(obs),  # Full content
                    "full_content": None,  # No enriched content for string observations
                    "entity_tags": [],
                    "metric_values": {},
                    "confidence": 1.0,
                })

        return extracted

    def _simplify_plan(self, plan: Any) -> Optional[Dict]:
        """Simplify plan object to essential fields."""
        if not plan:
            return None

        # Convert to dict if it's an object
        if hasattr(plan, '__dict__'):
            plan_dict = plan.__dict__
        elif hasattr(plan, 'dict'):
            plan_dict = plan.dict()
        elif isinstance(plan, dict):
            plan_dict = plan
        else:
            return {"_raw": str(plan)[:200]}

        steps = plan_dict.get("steps", [])

        # CRITICAL FIX: Include full steps array with search_queries
        # This was causing test fixtures to lose steps, making it impossible to debug
        # the research workflow. We now include the full steps with all fields.
        simplified_steps = []
        for step in steps:
            if hasattr(step, '__dict__'):
                step_dict = step.__dict__
            elif hasattr(step, 'dict'):
                step_dict = step.dict()
            elif isinstance(step, dict):
                step_dict = step
            else:
                simplified_steps.append({"title": str(step)[:100]})
                continue

            simplified_steps.append({
                "step_id": step_dict.get("step_id"),
                "title": step_dict.get("title"),
                "description": step_dict.get("description", "")[:200],  # Limit description length
                "step_type": str(step_dict.get("step_type")),
                "status": str(step_dict.get("status")),
                "need_search": step_dict.get("need_search"),
                "search_queries": step_dict.get("search_queries", []),  # CRITICAL: Include search queries!
                "depends_on": step_dict.get("depends_on"),
            })

        return {
            "plan_id": plan_dict.get("plan_id"),
            "title": plan_dict.get("title"),
            "research_topic": plan_dict.get("research_topic"),
            "num_steps": len(steps),
            "step_titles": [self._get_step_title(s) for s in steps[:5]],  # First 5 steps
            "steps": simplified_steps,  # CRITICAL FIX: Include full steps array!
            "iteration": plan_dict.get("iteration"),
            "suggested_report_structure": (plan_dict.get("suggested_report_structure") or [])[:10],
        }

    def _simplify_steps(self, steps: List) -> List[Dict]:
        """Simplify completed steps."""
        simplified = []
        for step in steps[:10]:  # Max 10 steps
            if hasattr(step, '__dict__'):
                step_dict = step.__dict__
            elif hasattr(step, 'dict'):
                step_dict = step.dict()
            elif isinstance(step, dict):
                step_dict = step
            else:
                continue

            simplified.append({
                "step_id": step_dict.get("step_id"),
                "title": step_dict.get("title"),
                "status": str(step_dict.get("status")),
                "step_type": str(step_dict.get("step_type")),
            })

        return simplified

    def _get_step_title(self, step: Any) -> str:
        """Extract title from step object or dict."""
        if hasattr(step, 'title'):
            return step.title
        elif isinstance(step, dict):
            return step.get('title', 'Unknown')
        else:
            return str(step)[:50]

    def _simplify_search_results(self, results: List) -> List[Dict]:
        """Keep only essential search result fields."""
        simplified = []
        for r in results[:20]:  # Keep max 20
            if isinstance(r, dict):
                simplified.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "score": r.get("score"),
                })
            elif hasattr(r, '__dict__'):
                simplified.append({
                    "title": getattr(r, 'title', None),
                    "url": getattr(r, 'url', None),
                    "score": getattr(r, 'score', None),
                })
        return simplified

    def _simplify_factuality_report(self, report: Any) -> Optional[Dict]:
        """Simplify factuality report to essential metrics."""
        if not report:
            return None

        if hasattr(report, '__dict__'):
            report_dict = report.__dict__
        elif hasattr(report, 'dict'):
            report_dict = report.dict()
        elif isinstance(report, dict):
            report_dict = report
        else:
            return {"_raw": str(report)[:200]}

        return {
            "total_claims": report_dict.get("total_claims"),
            "grounded_claims": report_dict.get("grounded_claims"),
            "partially_grounded_claims": report_dict.get("partially_grounded_claims"),
            "ungrounded_claims": report_dict.get("ungrounded_claims"),
            "contradicted_claims": report_dict.get("contradicted_claims"),
            "overall_factuality_score": report_dict.get("overall_factuality_score"),
            "confidence_score": report_dict.get("confidence_score"),
        }

    def _simplify_query_constraints(self, constraints: Any) -> Optional[Dict]:
        """Simplify QueryConstraints to essential fields."""
        if not constraints:
            return None

        # Convert to dict if it's an object
        # CRITICAL: Use dataclasses.asdict() for dataclasses to recursively convert nested dataclasses
        from dataclasses import is_dataclass, asdict

        if is_dataclass(constraints) and not isinstance(constraints, type):
            # It's a dataclass instance - use asdict() for recursive conversion
            constraints_dict = asdict(constraints)
        elif hasattr(constraints, '__dict__'):
            constraints_dict = constraints.__dict__
        elif hasattr(constraints, 'dict'):
            constraints_dict = constraints.dict()
        elif isinstance(constraints, dict):
            constraints_dict = constraints
        else:
            return {"_raw": str(constraints)[:200]}

        # Scenarios should already be dicts after asdict(), but double-check
        scenarios = constraints_dict.get("scenarios", [])
        if scenarios and not isinstance(scenarios[0], dict):
            print(f"⚠️ Warning: Scenarios not converted to dicts by asdict(): {type(scenarios[0])}")

        return {
            "entities": constraints_dict.get("entities", []),
            "metrics": constraints_dict.get("metrics", []),
            "scenarios": scenarios,  # Already converted by asdict()
            "time_period": constraints_dict.get("time_period"),
            "data_quality_requirements": constraints_dict.get("data_quality_requirements"),
        }

    def _simplify_unified_plan(self, plan: Any) -> Optional[Dict]:
        """
        Simplify UnifiedPlan to essential fields for testing.

        UnifiedPlan is a complex Pydantic model with data_sources, table_specs, etc.
        We serialize it properly to enable Tier 1 Hybrid mode testing.
        """
        if not plan:
            return None

        # Convert to dict if it's a Pydantic model
        if hasattr(plan, 'model_dump'):
            # Pydantic v2
            plan_dict = plan.model_dump()
        elif hasattr(plan, 'dict'):
            # Pydantic v1 or custom object
            plan_dict = plan.dict()
        elif isinstance(plan, dict):
            # Already a dict
            plan_dict = plan
        else:
            return {"_raw": str(plan)[:500]}

        # Return the full plan dict for complete testing
        # UnifiedPlan is already well-structured and necessary for calculation testing
        return plan_dict

    def _extract_recent_messages(self, messages: List) -> List[Dict]:
        """Extract ALL messages for coordinator - NO TRUNCATION."""
        extracted = []
        for msg in messages:  # ALL messages
            if hasattr(msg, 'content'):
                extracted.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content,  # Full content
                })
            elif isinstance(msg, dict):
                extracted.append({
                    "type": msg.get("type", "unknown"),
                    "content": str(msg.get("content", "")),  # Full content
                })
        return extracted

    def _save_state(self, agent_name: str, phase: str, state: Dict[str, Any]):
        """
        Save state to file.

        Args:
            agent_name: Name of the agent
            phase: "before", "after", or "error"
            state: State dict to save
        """
        # Create directory
        dir_path = self.capture_dir / agent_name
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create filename
        filename = f"{self.current_prompt_id}_{phase}.json"
        filepath = dir_path / filename

        # Add metadata
        state_with_metadata = {
            **state,
            "_metadata": {
                "prompt_id": self.current_prompt_id,
                "prompt": self.current_prompt if self.current_prompt else "",  # Full prompt
                "agent": agent_name,
                "phase": phase,
                "captured_at": datetime.now().isoformat(),
            }
        }

        # Calculate size before adding to metadata
        state_json = json.dumps(state_with_metadata, default=str)
        state_with_metadata["_metadata"]["state_size_bytes"] = len(state_json)
        state_with_metadata["_metadata"]["state_size_kb"] = len(state_json) / 1024

        # Save
        with open(filepath, 'w') as f:
            json.dump(state_with_metadata, f, indent=2, default=str)

        # Display path (handle both relative and absolute paths)
        try:
            display_path = filepath.resolve().relative_to(Path.cwd())
        except ValueError:
            # Path is outside cwd - show full path
            display_path = filepath
        print(f"   💾 Saved: {display_path} ({len(state_json)/1024:.1f}KB)")


# Global singleton instance
state_capture = StateCapture()
