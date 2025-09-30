#!/usr/bin/env python3
"""
Databricks-compliant response builder.

This builds responses that match the EXACT schema requirements from Databricks serving endpoints,
based on the retrieved OpenAPI schema.

IMPORTANT SCHEMA COMPLIANCE:
- All responses must follow MLflow ResponsesAgent or OpenAI ChatCompletion formats
- Delta events contain ONLY plain text, never JSON objects
- Done events have structured message format with content arrays
- See SCHEMA_REQUIREMENTS.md for complete specifications
"""

import time
from typing import Dict, List, Any, Optional

from .core import get_logger

logger = get_logger(__name__)


class DatabricksResponseBuilder:
    """Simplified response builder focused on event emission (unused methods removed)."""
    
    def __init__(self, schema_format: str = "auto"):
        """Initialize response builder (simplified)."""
        self.schema_format = schema_format

    # ===============================
    # Event Emission Helper Functions (only used methods kept)
    # ===============================
    
    def emit_plan_structure_event(
        self, 
        plan_id: str, 
        plan_name: str, 
        plan_description: str, 
        steps: List[Dict[str, Any]],
        complexity: str = "moderate",
        estimated_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a plan_structure intermediate event that matches the UI's expected format.
        
        Args:
            plan_id: Unique identifier for the plan
            plan_name: Human-readable name for the plan
            plan_description: Description of what the plan will accomplish
            steps: List of step dictionaries with step_id, name, description, status
            complexity: Plan complexity ("simple", "moderate", "complex")
            estimated_time: Estimated execution time in minutes
            
        Returns:
            Dict formatted as MLflow ResponsesAgentStreamEvent intermediate_event
        """
        event_data = {
            "type": "intermediate_event",
            "event_type": "plan_structure", 
            "data": {
                "plan_id": plan_id,
                "plan": {
                    "name": plan_name,
                    "description": plan_description,
                    "complexity": complexity,
                    "estimated_time": estimated_time or len(steps) * 5,  # 5 minutes per step default
                    "steps": [
                        {
                            "step_id": step.get("step_id", f"step_{i+1:03d}"),
                            "name": step.get("name", f"Step {i+1}"),
                            "description": step.get("description", ""),
                            "status": step.get("status", "pending"),
                            "type": step.get("type", "research"),
                            "estimated_time": step.get("estimated_time", 5)
                        }
                        for i, step in enumerate(steps)
                    ]
                }
            }
        }
        
        logger.info(
            "Created plan_structure event",
            plan_id=plan_id,
            steps_count=len(steps),
            complexity=complexity
        )
        
        return event_data

    def emit_step_started_event(
        self,
        step_id: str,
        step_index: int,
        step_name: str,
        step_type: str = "research"
    ) -> Dict[str, Any]:
        """
        Create a step_started intermediate event.
        This is an alias for step_activated that some UI components expect.

        Args:
            step_id: Unique identifier for the step
            step_index: Zero-based index of the step in the plan
            step_name: Human-readable name for the step
            step_type: Type of step (research, analysis, etc.)

        Returns:
            Dict formatted as step_started event
        """
        return {
            "event_type": "step_started",
            "data": {
                "step_id": step_id,
                "step_index": step_index,
                "step_name": step_name,
                "step_type": step_type,
                "status": "in_progress",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }

    def emit_step_activated_event(
        self,
        step_id: str,
        step_index: int,
        step_name: str,
        step_type: str = "research"
    ) -> Dict[str, Any]:
        """
        Create a step_activated intermediate event.
        
        Args:
            step_id: Unique identifier for the step
            step_index: Zero-based index of the step in the plan
            step_name: Human-readable name for the step
            step_type: Type of step (research, analysis, etc.)
            
        Returns:
            Dict formatted as MLflow ResponsesAgentStreamEvent intermediate_event
        """
        event_data = {
            "type": "intermediate_event", 
            "event_type": "step_activated",
            "data": {
                "step_id": step_id,
                "step_index": step_index,
                "step_name": step_name,
                "step_type": step_type,
                "status": "active",
                "timestamp": int(time.time())
            }
        }
        
        logger.info(
            "Created step_activated event", 
            step_id=step_id,
            step_index=step_index,
            step_name=step_name
        )
        
        return event_data

    def emit_step_completed_event(
        self, 
        step_id: str, 
        step_index: int, 
        step_name: str,
        step_type: str = "research",
        success: bool = True,
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a step_completed intermediate event.
        
        Args:
            step_id: Unique identifier for the step
            step_index: Zero-based index of the step in the plan
            step_name: Human-readable name for the step
            step_type: Type of step (research, analysis, etc.)
            success: Whether the step completed successfully
            summary: Optional summary of what was accomplished
            
        Returns:
            Dict formatted as MLflow ResponsesAgentStreamEvent intermediate_event
        """
        event_data = {
            "type": "intermediate_event",
            "event_type": "step_completed", 
            "data": {
                "step_id": step_id,
                "step_index": step_index,
                "step_name": step_name,
                "step_type": step_type,
                "status": "completed" if success else "failed",
                "success": success,
                "summary": summary,
                "timestamp": int(time.time())
            }
        }
        
        logger.info(
            "Created step_completed event",
            step_id=step_id, 
            step_index=step_index,
            step_name=step_name,
            success=success
        )
        
        return event_data


# Single instance for event emission (simplified)
databricks_response_builder = DatabricksResponseBuilder()