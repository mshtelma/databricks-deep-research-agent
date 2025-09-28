"""
Event transformer module for converting technical events to user-presentable format.

This module transforms internal agent events into human-readable format suitable 
for display in the UI, while filtering out purely technical events that aren't 
relevant to users.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .types import IntermediateEventType

logger = logging.getLogger(__name__)


@dataclass
class UserFriendlyEvent:
    """User-friendly representation of an intermediate event."""
    
    id: str
    timestamp: float
    icon: str
    title: str
    description: str
    category: str  # "search", "analysis", "planning", "writing", etc.
    progress: Optional[float] = None  # 0.0 to 1.0 for progress events
    metadata: Optional[Dict[str, Any]] = None


class UserFriendlyEventTransformer:
    """
    Transform technical events into user-presentable format.
    
    This class converts internal agent events into human-readable messages
    that can be displayed in the UI to show research progress.
    """
    
    # Event transformation mappings
    EVENT_TRANSFORMATIONS = {
        # Agent events
        IntermediateEventType.AGENT_START: {
            "icon": "🤖",
            "category": "agent",
            "transform": lambda data: {
                "title": f"Starting {data.get('agent_name', 'Agent')}",
                "description": _get_agent_description(data.get('agent_name', ''))
            }
        },
        
        IntermediateEventType.AGENT_COMPLETE: {
            "icon": "✅", 
            "category": "agent",
            "transform": lambda data: {
                "title": f"{data.get('agent_name', 'Agent')} Complete",
                "description": f"Finished {data.get('agent_name', '').lower()} phase"
            }
        },
        
        IntermediateEventType.AGENT_THINKING: {
            "icon": "💭",
            "category": "analysis", 
            "transform": lambda data: {
                "title": "Analyzing",
                "description": data.get('thought_summary', 'Processing information...')
            }
        },
        
        # Search events
        IntermediateEventType.SEARCH_RESULTS_FOUND: {
            "icon": "🔍",
            "category": "search",
            "transform": lambda data: {
                "title": f"Found {data.get('count', 0)} search results",
                "description": f"Searching for: {data.get('query', '')[:60]}..."
            }
        },
        
        IntermediateEventType.QUERY_GENERATED: {
            "icon": "🔎", 
            "category": "search",
            "transform": lambda data: {
                "title": "Generated search query",
                "description": f"Query: \"{data.get('query', '')}\""
            }
        },
        
        IntermediateEventType.SOURCE_ANALYZED: {
            "icon": "📄",
            "category": "analysis",
            "transform": lambda data: {
                "title": "Analyzing source",
                "description": data.get('title', 'Processing content...')[:80] + "..."
            }
        },
        
        # Planning events
        IntermediateEventType.PLAN_CREATED: {
            "icon": "📋",
            "category": "planning",
            "transform": lambda data: {
                "title": "Research plan created",
                "description": f"Generated {data.get('steps_count', 0)} research steps"
            }
        },
        
        IntermediateEventType.PLAN_UPDATED: {
            "icon": "📝",
            "category": "planning", 
            "transform": lambda data: {
                "title": "Plan updated",
                "description": f"Refined research approach (iteration {data.get('iteration', 1)})"
            }
        },
        
        # Verification events
        IntermediateEventType.GROUNDING_START: {
            "icon": "🔍",
            "category": "verification",
            "transform": lambda data: {
                "title": "Fact-checking started",
                "description": "Verifying claims and checking for contradictions"
            }
        },
        
        IntermediateEventType.GROUNDING_COMPLETE: {
            "icon": "✓", 
            "category": "verification",
            "transform": lambda data: {
                "title": "Fact-checking complete", 
                "description": f"Verified {data.get('claims_checked', 0)} claims"
            }
        },
        
        IntermediateEventType.CONTRADICTION_FOUND: {
            "icon": "⚠️",
            "category": "verification",
            "transform": lambda data: {
                "title": "Potential contradiction found",
                "description": data.get('claim', 'Conflicting information detected')[:80] + "..."
            }
        },
        
        # Report generation
        IntermediateEventType.REPORT_GENERATION: {
            "icon": "📊",
            "category": "writing",
            "transform": lambda data: {
                "title": "Generating report",
                "description": f"Creating {data.get('style', 'default')} report"
            }
        },
        
        # Progress events
        IntermediateEventType.ACTION_PROGRESS: {
            "icon": "⏳",
            "category": "progress",
            "transform": lambda data: {
                "title": data.get('action', 'Processing'),
                "description": data.get('progress_description', ''),
                "progress": data.get('progress', 0) / 100.0 if data.get('progress') else None
            }
        },
    }
    
    # Events that should be hidden from users (purely technical)
    HIDDEN_EVENTS = {
        IntermediateEventType.LLM_PROMPT_SENT,
        IntermediateEventType.LLM_STREAMING,
        IntermediateEventType.LLM_RESPONSE_COMPLETE,
        IntermediateEventType.TOOL_CALL_START,  # Unless specifically marked for display
        IntermediateEventType.TOOL_CALL_COMPLETE,  # Unless specifically marked for display
    }
    
    def transform_event(self, event: Dict[str, Any]) -> Optional[UserFriendlyEvent]:
        """
        Transform a single intermediate event to user-friendly format.
        
        Args:
            event: Raw intermediate event data
            
        Returns:
            UserFriendlyEvent object or None if event should be hidden
        """
        try:
            event_type_str = event.get('event_type', '')
            event_id = event.get('id', '')
            timestamp = event.get('timestamp', datetime.now().timestamp())
            data = event.get('data', {})
            meta = event.get('meta', {})
            
            # Parse event type
            try:
                event_type = IntermediateEventType(event_type_str)
            except ValueError:
                logger.warning(f"Unknown event type: {event_type_str}")
                return None
            
            # Check if event should be hidden
            if event_type in self.HIDDEN_EVENTS:
                # Unless explicitly marked as user_presentable
                if not meta.get('user_presentable', False):
                    return None
            
            # Check for transformation mapping
            if event_type not in self.EVENT_TRANSFORMATIONS:
                # Generic fallback for unmapped events
                return self._create_generic_event(event_id, timestamp, event_type_str, data)
            
            transform_config = self.EVENT_TRANSFORMATIONS[event_type]
            transform_func = transform_config['transform']
            
            # Apply transformation
            transform_result = transform_func(data)
            
            return UserFriendlyEvent(
                id=event_id,
                timestamp=timestamp,
                icon=transform_config['icon'],
                title=transform_result['title'],
                description=transform_result['description'],
                category=transform_config['category'],
                progress=transform_result.get('progress'),
                metadata={
                    'original_event_type': event_type_str,
                    'raw_data': data if meta.get('include_raw_data') else None
                }
            )
            
        except Exception as e:
            logger.error(f"Error transforming event {event.get('id', 'unknown')}: {e}")
            return None
    
    def transform_events(self, events: List[Dict[str, Any]]) -> List[UserFriendlyEvent]:
        """
        Transform a list of intermediate events to user-friendly format.
        
        Args:
            events: List of raw intermediate event data
            
        Returns:
            List of UserFriendlyEvent objects
        """
        transformed = []
        
        for event in events:
            user_event = self.transform_event(event)
            if user_event:
                transformed.append(user_event)
        
        # Sort by timestamp 
        transformed.sort(key=lambda e: e.timestamp)
        
        logger.debug(f"Transformed {len(events)} events to {len(transformed)} user-friendly events")
        
        return transformed
    
    def _create_generic_event(
        self, 
        event_id: str, 
        timestamp: float, 
        event_type: str, 
        data: Dict[str, Any]
    ) -> UserFriendlyEvent:
        """Create a generic user-friendly event for unmapped event types."""
        
        # Try to extract useful information from data
        title = data.get('title', '') or data.get('action', '') or event_type.replace('_', ' ').title()
        description = data.get('description', '') or data.get('summary', '') or 'Processing...'
        
        # Choose appropriate icon based on event type
        icon = "🔧"  # Default
        category = "other"
        
        if 'search' in event_type.lower():
            icon = "🔍"
            category = "search"
        elif 'plan' in event_type.lower():
            icon = "📋" 
            category = "planning"
        elif 'report' in event_type.lower() or 'write' in event_type.lower():
            icon = "📝"
            category = "writing"
        elif 'analysis' in event_type.lower() or 'synthesis' in event_type.lower():
            icon = "🧠"
            category = "analysis"
        
        return UserFriendlyEvent(
            id=event_id,
            timestamp=timestamp,
            icon=icon,
            title=title,
            description=description[:100] + "..." if len(description) > 100 else description,
            category=category,
            metadata={'original_event_type': event_type}
        )


def _get_agent_description(agent_name: str) -> str:
    """Get human-readable description for agent names."""
    descriptions = {
        "coordinator": "Analyzing request and determining research approach",
        "planner": "Creating structured research plan",
        "researcher": "Executing research steps and gathering information", 
        "fact_checker": "Verifying claims and checking factual accuracy",
        "reporter": "Synthesizing findings into final report"
    }
    
    return descriptions.get(agent_name.lower(), f"Running {agent_name} process")


# Global transformer instance
global_event_transformer = UserFriendlyEventTransformer()


# Convenience functions
def transform_event(event: Dict[str, Any]) -> Optional[UserFriendlyEvent]:
    """Transform a single event using the global transformer."""
    return global_event_transformer.transform_event(event)


def transform_events(events: List[Dict[str, Any]]) -> List[UserFriendlyEvent]:
    """Transform a list of events using the global transformer."""
    return global_event_transformer.transform_events(events)