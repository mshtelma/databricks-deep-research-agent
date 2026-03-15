# Events

> The typed streaming event system for real-time workflow observability.

## Overview
Every significant action in the framework emits a typed event. Events form a discriminated union (FrameworkEvent) with 40-50+ event types. This enables:
- Real-time UI updates
- Progress tracking
- Debugging and logging
- Token usage monitoring

## Event Design
- All events are Pydantic BaseModel instances
- Discriminated on `event_type` field (literal string)
- Each has `node_id` and `timestamp`
- Events are emitted as async generators during execution
- Type-safe pattern matching via `match event.event_type:`

## StreamEvent Type
`StreamEvent = FrameworkEvent` (the union type). Applications consume `AsyncGenerator[StreamEvent, None]`.

## Event Categories
Group events by category. For each category, list event types with brief descriptions:

### Workflow Lifecycle
- WorkflowStartedEvent — Workflow execution begins
- WorkflowCompletedEvent — Workflow finishes (has total_tokens, duration)

### Node Lifecycle
- NodeStartedEvent — A node begins execution
- NodeCompletedEvent — A node finishes successfully
- NodeErrorEvent — A node encountered an error
- NodeSkippedEvent — A node was skipped (error_handling: skip)

### Agent Output
- AgentOutputEvent — Final agent output (parsed result)
- AgentStreamChunkEvent — Streaming chunk from LLM

### Domain-Specific
- CoordinatorClassifiedEvent — Query classified (complexity, mode)
- PlanCreatedEvent — Research plan with steps
- ReflectionDecisionEvent — Reflector decision (CONTINUE/ADJUST/COMPLETE)
- BackgroundCompletedEvent — Background context ready
- SynthesisStartedEvent — Synthesis phase begins

### Plan-and-Execute
- ItemStartedEvent — Starting a plan item
- ItemCompletedEvent — Plan item finished
- ItemsExtractedEvent — Items parsed from plan
- EvaluationDecisionEvent — Evaluator verdict (continue/replan/done)
- ReplanTriggeredEvent — Plan being revised
- PlanAndExecuteExitEvent — Plan-and-execute loop exits

### Tool Calls
- ToolCallEvent — Tool invoked (name, arguments)
- ToolResultEvent — Tool returned result
- ToolCacheHitEvent — Cached result used

### Loop Control
- LoopIterationEvent — Loop iteration started
- LoopExitEvent — Loop exited (reason)

### Conditional
- BranchSelectedEvent — Which branch was taken

### Token Budget
- TokenUsageEvent — Token usage for a request
- TokenBudgetExceededEvent — Budget limit hit

### Conversation
- ConversationCompactedEvent — Context was compacted

### Checkpoint
- CheckpointSavedEvent — State checkpointed
- CheckpointResumedEvent — State restored from checkpoint

### Citation/Verification
- ClaimGeneratedEvent — Claim extracted
- ClaimVerifiedEvent — Claim verified
- CitationCorrectedEvent — Citation corrected
- NumericClaimDetectedEvent — Numeric claim found
- VerificationSummaryEvent — Verification summary

## Consuming Events
```python
async for event in executor.execute():
    match event.event_type:
        case "workflow_started":
            print(f"Starting workflow: {event.workflow_name}")
        case "plan_created":
            for i, step in enumerate(event.steps, 1):
                print(f"  Step {i}: {step}")
        case "tool_call":
            print(f"Calling {event.tool_name}({event.arguments})")
        case "reflection_decision":
            print(f"Reflector: {event.decision} — {event.reasoning}")
        case "agent_stream_chunk":
            print(event.chunk, end="")
        case "workflow_completed":
            print(f"\nDone! {event.total_tokens} tokens used")
```

## Event Filtering
```python
# Only domain events
domain_events = [e async for e in executor.execute() if e.event_type in {
    "plan_created", "reflection_decision", "coordinator_classified"
}]
```

## See Also
- [Streaming and Events Guide](../guides/streaming-and-events.md)
- [Event Types Reference](../reference/event-types-reference.md)
- [Architecture](architecture.md)
