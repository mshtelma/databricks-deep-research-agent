import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronUp, Search, Brain, Link, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { IntermediateEvent, IntermediateEventType } from '../../types/chat';

interface ActivityFeedProps {
    events: IntermediateEvent[];
    showThoughts?: boolean;
    className?: string;
}

interface ActivityItemProps {
    event: IntermediateEvent;
    showThoughts: boolean;
}

const ActivityItem: React.FC<ActivityItemProps> = ({ event, showThoughts }) => {
    const getEventIcon = () => {
        switch (event.event_type) {
            case IntermediateEventType.ACTION_START:
            case IntermediateEventType.ACTION_PROGRESS:
            case IntermediateEventType.ACTION_COMPLETE:
                return <Clock className="w-4 h-4" />;

            case IntermediateEventType.TOOL_CALL_START:
            case IntermediateEventType.TOOL_CALL_PROGRESS:
            case IntermediateEventType.TOOL_CALL_COMPLETE:
                if (event.data.tool_name?.includes('search')) {
                    return <Search className="w-4 h-4" />;
                }
                return <CheckCircle className="w-4 h-4" />;

            case IntermediateEventType.TOOL_CALL_ERROR:
                return <AlertTriangle className="w-4 h-4 text-red-500" />;

            case IntermediateEventType.THOUGHT_SNAPSHOT:
            case IntermediateEventType.SYNTHESIS_PROGRESS:
                return <Brain className="w-4 h-4" />;

            case IntermediateEventType.CITATION_ADDED:
                return <Link className="w-4 h-4" />;

            // Multi-agent specific event icons
            case IntermediateEventType.AGENT_HANDOFF:
                return <span className="w-4 h-4 text-sm">🔄</span>;
            
            case IntermediateEventType.PLAN_CREATED:
            case IntermediateEventType.PLAN_UPDATED:
            case IntermediateEventType.PLAN_ITERATION:
                return <span className="w-4 h-4 text-sm">📋</span>;
                
            case IntermediateEventType.BACKGROUND_INVESTIGATION:
                return <span className="w-4 h-4 text-sm">🔍</span>;
                
            case IntermediateEventType.GROUNDING_START:
            case IntermediateEventType.GROUNDING_COMPLETE:
                return <span className="w-4 h-4 text-sm">🔎</span>;
                
            case IntermediateEventType.GROUNDING_CONTRADICTION:
                return <span className="w-4 h-4 text-sm">⚠️</span>;
                
            case IntermediateEventType.REPORT_GENERATION:
                return <span className="w-4 h-4 text-sm">📄</span>;
                
            case IntermediateEventType.QUALITY_ASSESSMENT:
                return <span className="w-4 h-4 text-sm">⭐</span>;

            default:
                return <Clock className="w-4 h-4" />;
        }
    };

    const getEventColor = () => {
        switch (event.event_type) {
            case IntermediateEventType.ACTION_START:
            case IntermediateEventType.TOOL_CALL_START:
                return 'text-blue-600 bg-blue-50';

            case IntermediateEventType.ACTION_PROGRESS:
            case IntermediateEventType.TOOL_CALL_PROGRESS:
                return 'text-yellow-600 bg-yellow-50';

            case IntermediateEventType.ACTION_COMPLETE:
            case IntermediateEventType.TOOL_CALL_COMPLETE:
                return 'text-green-600 bg-green-50';

            case IntermediateEventType.TOOL_CALL_ERROR:
                return 'text-red-600 bg-red-50';

            case IntermediateEventType.THOUGHT_SNAPSHOT:
            case IntermediateEventType.SYNTHESIS_PROGRESS:
                return 'text-purple-600 bg-purple-50';

            case IntermediateEventType.CITATION_ADDED:
                return 'text-indigo-600 bg-indigo-50';

            // Multi-agent specific event colors
            case IntermediateEventType.AGENT_HANDOFF:
                return 'text-cyan-600 bg-cyan-50';
            
            case IntermediateEventType.PLAN_CREATED:
            case IntermediateEventType.PLAN_UPDATED:
            case IntermediateEventType.PLAN_ITERATION:
                return 'text-purple-600 bg-purple-50';
                
            case IntermediateEventType.BACKGROUND_INVESTIGATION:
                return 'text-indigo-600 bg-indigo-50';
                
            case IntermediateEventType.GROUNDING_START:
            case IntermediateEventType.GROUNDING_COMPLETE:
                return 'text-teal-600 bg-teal-50';
                
            case IntermediateEventType.GROUNDING_CONTRADICTION:
                return 'text-red-600 bg-red-50';
                
            case IntermediateEventType.REPORT_GENERATION:
                return 'text-green-600 bg-green-50';
                
            case IntermediateEventType.QUALITY_ASSESSMENT:
                return 'text-yellow-600 bg-yellow-50';

            default:
                return 'text-gray-600 bg-gray-50';
        }
    };

    const getEventTitle = () => {
        switch (event.event_type) {
            case IntermediateEventType.ACTION_START:
                return `Starting: ${event.data.action}`;

            case IntermediateEventType.ACTION_PROGRESS:
                return `Progress: ${event.data.action} (${event.data.progress?.percentage?.toFixed(0)}%)`;

            case IntermediateEventType.ACTION_COMPLETE:
                return `Completed: ${event.data.action}`;

            case IntermediateEventType.TOOL_CALL_START:
                return `Starting search: ${event.data.parameters?.query}`;

            case IntermediateEventType.TOOL_CALL_COMPLETE:
                return `Search complete: ${event.data.results_count} results`;

            case IntermediateEventType.TOOL_CALL_ERROR:
                return `Search error: ${event.data.tool_name}`;

            case IntermediateEventType.THOUGHT_SNAPSHOT:
                return 'Agent reasoning...';

            case IntermediateEventType.SYNTHESIS_PROGRESS:
                return `Synthesis: ${event.data.progress_type}`;

            case IntermediateEventType.CITATION_ADDED:
                return `Found source: ${event.data.title}`;

            // Multi-agent specific event titles
            case IntermediateEventType.AGENT_HANDOFF:
                return `Handoff: ${event.data.from_agent} → ${event.data.to_agent}`;
            
            case IntermediateEventType.PLAN_CREATED:
                return `Research plan created with ${event.data.step_count || 'multiple'} steps`;
                
            case IntermediateEventType.PLAN_UPDATED:
                return `Plan updated (iteration ${event.data.iteration || 'unknown'})`;
                
            case IntermediateEventType.PLAN_ITERATION:
                return `Plan refinement (quality: ${event.data.quality ? Math.round(event.data.quality * 100) + '%' : 'assessing'})`;
                
            case IntermediateEventType.BACKGROUND_INVESTIGATION:
                return `Background research: ${event.data.topic || 'investigating context'}`;
                
            case IntermediateEventType.GROUNDING_START:
                return 'Starting fact verification...';
                
            case IntermediateEventType.GROUNDING_COMPLETE:
                return `Fact check complete (${event.data.factuality_score ? Math.round(event.data.factuality_score * 100) + '% factuality' : 'verified'})`;
                
            case IntermediateEventType.GROUNDING_CONTRADICTION:
                return `Contradiction detected: ${event.data.claim || 'fact verification issue'}`;
                
            case IntermediateEventType.REPORT_GENERATION:
                return `Generating ${event.data.report_style || 'professional'} report...`;
                
            case IntermediateEventType.QUALITY_ASSESSMENT:
                return `Quality assessment: ${event.data.score ? Math.round(event.data.score * 100) + '%' : 'evaluating'}`;

            default:
                return 'Activity update';
        }
    };

    const getEventDetails = () => {
        switch (event.event_type) {
            case IntermediateEventType.TOOL_CALL_START:
                if (event.data.parameters?.query) {
                    return event.data.parameters.query;
                }
                break;

            case IntermediateEventType.TOOL_CALL_COMPLETE:
                return event.data.result_summary;

            case IntermediateEventType.TOOL_CALL_ERROR:
                return event.data.error_message;

            case IntermediateEventType.THOUGHT_SNAPSHOT:
                if (showThoughts) {
                    return event.data.content;
                }
                break;

            case IntermediateEventType.SYNTHESIS_PROGRESS:
                return event.data.content_preview;

            case IntermediateEventType.CITATION_ADDED:
                return event.data.snippet;

            // Multi-agent specific event details
            case IntermediateEventType.AGENT_HANDOFF:
                return event.data.reason || `Transferring control to ${event.data.to_agent}`;
            
            case IntermediateEventType.PLAN_CREATED:
                return event.data.description || 'Initial research plan established';
                
            case IntermediateEventType.PLAN_UPDATED:
                return event.data.changes || 'Plan refined based on quality assessment';
                
            case IntermediateEventType.PLAN_ITERATION:
                return event.data.feedback || 'Improving plan quality and coverage';
                
            case IntermediateEventType.BACKGROUND_INVESTIGATION:
                return event.data.findings || 'Gathering initial context and background';
                
            case IntermediateEventType.GROUNDING_START:
                return event.data.verification_level || 'Initiating fact-checking process';
                
            case IntermediateEventType.GROUNDING_COMPLETE:
                const contradictions = event.data.contradictions_found || 0;
                return contradictions > 0 ? 
                    `Found ${contradictions} potential issues to resolve` : 
                    'All facts verified successfully';
                
            case IntermediateEventType.GROUNDING_CONTRADICTION:
                return event.data.evidence || 'Conflicting information detected';
                
            case IntermediateEventType.REPORT_GENERATION:
                return event.data.section || 'Synthesizing research findings';
                
            case IntermediateEventType.QUALITY_ASSESSMENT:
                return event.data.metrics || 'Evaluating research completeness and accuracy';

            default:
                break;
        }
        return null;
    };

    // Don't show thought snapshots if disabled
    if (event.event_type === IntermediateEventType.THOUGHT_SNAPSHOT && !showThoughts) {
        return null;
    }

    const title = getEventTitle();
    const details = getEventDetails();
    const colorClass = getEventColor();
    const icon = getEventIcon();
    const timestamp = new Date(event.timestamp * 1000);

    // Get agent identifier from event data
    const getAgentInfo = () => {
        const agent = event.data.agent || event.data.from_agent || event.data.current_agent
        if (!agent) return null
        
        const agentIcons = {
            'coordinator': '🎯',
            'planner': '📋',
            'researcher': '🔬',
            'fact_checker': '🔎',
            'reporter': '📄'
        }
        
        const agentColors = {
            'coordinator': 'bg-blue-100 text-blue-800',
            'planner': 'bg-purple-100 text-purple-800',
            'researcher': 'bg-orange-100 text-orange-800',
            'fact_checker': 'bg-red-100 text-red-800',
            'reporter': 'bg-green-100 text-green-800'
        }
        
        return {
            name: agent,
            icon: agentIcons[agent] || '🤖',
            colorClass: agentColors[agent] || 'bg-gray-100 text-gray-800'
        }
    }
    
    const agentInfo = getAgentInfo()

    return (
        <div className={`flex items-start gap-3 p-3 rounded-lg ${colorClass}`}>
            <div className="flex items-center gap-2">
                <div className="flex-shrink-0 mt-0.5">
                    {icon}
                </div>
                {agentInfo && (
                    <div className={`px-2 py-0.5 rounded-full text-xs font-medium ${agentInfo.colorClass}`}>
                        <span className="mr-1">{agentInfo.icon}</span>
                        {agentInfo.name}
                    </div>
                )}
            </div>
            <div className="flex-grow min-w-0">
                <div className="flex items-center justify-between">
                    <p className="text-sm font-medium truncate">{title}</p>
                    <span className="text-xs text-gray-500 ml-2">
                        {timestamp.toLocaleTimeString()}
                    </span>
                </div>
                {details && (
                    <p className="text-xs mt-1 text-gray-700 line-clamp-2">
                        {details}
                    </p>
                )}
                {event.event_type === IntermediateEventType.CITATION_ADDED && event.data.url && (
                    <a
                        href={event.data.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-600 hover:underline mt-1 block truncate"
                    >
                        {event.data.url}
                    </a>
                )}
            </div>
        </div>
    );
};

export const ActivityFeed: React.FC<ActivityFeedProps> = ({
    events,
    showThoughts = false,
    className = ''
}) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [newEventCount, setNewEventCount] = useState(0);
    const [userScrolledUp, setUserScrolledUp] = useState(false);
    const feedRef = useRef<HTMLDivElement>(null);
    const lastEventCountRef = useRef(events.length);

    // Group events by correlation_id for better organization
    const groupedEvents = React.useMemo(() => {
        const groups: { [key: string]: IntermediateEvent[] } = {};

        events.forEach(event => {
            const key = event.correlation_id || 'default';
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(event);
        });

        // Sort events within each group by sequence
        Object.keys(groups).forEach(key => {
            groups[key].sort((a, b) => a.sequence - b.sequence);
        });

        return groups;
    }, [events]);

    // Auto-scroll unless user has scrolled up
    useEffect(() => {
        if (feedRef.current && !userScrolledUp && events.length > lastEventCountRef.current) {
            feedRef.current.scrollTop = feedRef.current.scrollHeight;
        }

        // Track new events for notification
        if (events.length > lastEventCountRef.current && !isExpanded) {
            setNewEventCount(prev => prev + (events.length - lastEventCountRef.current));
        }

        lastEventCountRef.current = events.length;
    }, [events, userScrolledUp, isExpanded]);

    // Handle scroll to detect if user scrolled up
    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        const element = e.target as HTMLDivElement;
        const isAtBottom = element.scrollHeight - element.scrollTop <= element.clientHeight + 10;
        setUserScrolledUp(!isAtBottom);
    };

    // Reset new event count when expanded
    useEffect(() => {
        if (isExpanded) {
            setNewEventCount(0);
        }
    }, [isExpanded]);

    const visibleEvents = showThoughts
        ? events
        : events.filter(event => event.event_type !== IntermediateEventType.THOUGHT_SNAPSHOT);

    return (
        <div className={`bg-white border rounded-lg shadow-sm ${className}`}>
            <div
                className="flex items-center justify-between p-3 border-b cursor-pointer hover:bg-gray-50"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="flex items-center gap-2">
                    <h3 className="text-sm font-medium text-gray-900">Activity Feed</h3>
                    {!isExpanded && newEventCount > 0 && (
                        <span className="bg-blue-500 text-white text-xs px-2 py-0.5 rounded-full">
                            {newEventCount} new
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500">
                        {visibleEvents.length} events
                    </span>
                    {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-gray-400" />
                    ) : (
                        <ChevronDown className="w-4 h-4 text-gray-400" />
                    )}
                </div>
            </div>

            {isExpanded && (
                <div
                    ref={feedRef}
                    className="max-h-96 overflow-y-auto p-3 space-y-2"
                    onScroll={handleScroll}
                >
                    {visibleEvents.length === 0 ? (
                        <p className="text-center text-gray-500 text-sm py-8">
                            No activity yet
                        </p>
                    ) : (
                        Object.entries(groupedEvents).map(([correlationId, groupEvents]) => (
                            <div key={correlationId} className="space-y-1">
                                {groupEvents
                                    .filter(event => showThoughts || event.event_type !== IntermediateEventType.THOUGHT_SNAPSHOT)
                                    .map(event => (
                                        <ActivityItem
                                            key={event.id}
                                            event={event}
                                            showThoughts={showThoughts}
                                        />
                                    ))
                                }
                            </div>
                        ))
                    )}

                    {userScrolledUp && (
                        <div className="sticky bottom-0 text-center">
                            <button
                                onClick={() => {
                                    if (feedRef.current) {
                                        feedRef.current.scrollTop = feedRef.current.scrollHeight;
                                        setUserScrolledUp(false);
                                    }
                                }}
                                className="bg-blue-500 text-white text-xs px-3 py-1 rounded-full hover:bg-blue-600 transition-colors"
                            >
                                ↓ New updates
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ActivityFeed;
