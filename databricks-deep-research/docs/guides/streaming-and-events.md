# Streaming and Events

> Build real-time UIs and monitoring from the framework's event stream.

## Overview

The framework emits 35+ typed events during execution. These events are the
primary interface for building UIs, logging, and monitoring. Every event is a
Pydantic `BaseModel` with an `event_type` literal discriminator, a `node_id`,
and an ISO-8601 `timestamp`.

Events are produced as an `AsyncGenerator[StreamEvent, None]` by both the
low-level `WorkflowExecutor` and the high-level `WorkflowRunner`.

## Consuming Events

### High-level API (WorkflowRunner)

`WorkflowRunner` is the recommended entry point. It wraps client creation,
workflow loading, and execution into a single object.

```python
from databricks_deep_research import WorkflowRunner

runner = WorkflowRunner.from_databricks(
    model="databricks-claude-haiku-4-5",
)

# Streaming — yields events as they happen
async for event in runner.stream("deep_research.yaml", query="What is quantum computing?"):
    handle_event(event)

# The final result (output text, sources, state) is available after iteration
result = runner.last_result
print(result.output)
print(f"Sources: {len(result.sources)}")
```

### Batch API (run to completion)

If you do not need real-time events, collect them all at once:

```python
result = await runner.run("deep_research.yaml", query="What is quantum computing?")
print(result.output)

# All events are stored in result.events
for event in result.events:
    if event.event_type == "tool_call":
        print(f"Called: {event.tool_name}")
```

### Low-level API (WorkflowExecutor)

For full control over state initialization, tool factories, and execution:

```python
from databricks_deep_research import (
    WorkflowExecutor,
    FrameworkLLMClient,
    WorkflowState,
    load_workflow,
)

definition = load_workflow("deep_research.yaml")
client = FrameworkLLMClient.from_databricks(model="databricks-claude-haiku-4-5")
state = WorkflowState(query="What is quantum computing?")

executor = WorkflowExecutor(definition, client)

async for event in executor.execute(state):
    handle_event(event)
```

## Pattern Matching

Every event carries an `event_type` literal. Use Python's structural pattern
matching to dispatch:

```python
from databricks_deep_research.events.types import StreamEvent

def handle_event(event: StreamEvent) -> None:
    match event.event_type:
        # Workflow lifecycle
        case "workflow_started":
            print(f"Starting: {event.workflow_name}")
        case "workflow_completed":
            print(f"Done in {event.duration_ms:.0f}ms, "
                  f"{event.total_tokens} tokens, "
                  f"{event.total_sources} sources")

        # Planning
        case "plan_created":
            print(f"Plan: {event.title}")
            for i, step in enumerate(event.steps, 1):
                print(f"  Step {i}: {step}")

        # Research progress
        case "item_started":
            print(f"Researching step {event.item_index + 1}/{event.total_items}: "
                  f"{event.item_summary}")
        case "item_completed":
            print(f"  Completed ({event.items_processed} done)")

        # Tool calls
        case "tool_call":
            print(f"Calling {event.tool_name}({event.arguments})")
        case "tool_result":
            status = "OK" if event.tool_success else f"ERROR: {event.tool_error}"
            print(f"  -> {status}, {event.accepted_source_count} sources accepted")

        # Reflection
        case "reflection_decision":
            print(f"Reflector: {event.decision} -- {event.reasoning}")

        # Streaming synthesis output
        case "agent_stream_chunk":
            print(event.chunk, end="", flush=True)

        # Token tracking
        case "token_usage":
            print(f"  Tokens: {event.total_tokens} "
                  f"(budget remaining: {event.budget_remaining})")

        # Errors
        case "node_error":
            print(f"ERROR in {event.node_id}: {event.error_message}")
            if event.will_retry:
                print(f"  Retrying (attempt {event.retry_attempt})")
```

## Building a Progress UI

A practical terminal progress display that tracks each phase of research:

```python
import sys
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """Tracks research progress from the event stream."""

    phase: str = "starting"
    current_step: int = 0
    total_steps: int = 0
    sources_found: int = 0
    total_tokens: int = 0
    plan_title: str = ""
    events_seen: int = 0

    def handle(self, event) -> None:
        self.events_seen += 1

        match event.event_type:
            case "coordinator_classified":
                self.phase = "coordinator"
                self._print_status(
                    f"Query classified: {event.complexity} "
                    f"(depth: {event.recommended_depth})"
                )

            case "background_completed":
                self.phase = "background"
                self.sources_found += event.sources_discovered
                self._print_status(
                    f"Background: {event.sources_discovered} sources discovered"
                )

            case "plan_created":
                self.phase = "planning"
                self.total_steps = len(event.steps)
                self.plan_title = event.title
                self._print_status(f"Plan: {event.title} ({self.total_steps} steps)")

            case "item_started":
                self.phase = "researching"
                self.current_step = event.item_index + 1
                self.total_steps = event.total_items
                self._print_status(
                    f"Step {self.current_step}/{self.total_steps}: "
                    f"{event.item_summary[:60]}"
                )

            case "reflection_decision":
                self.phase = "reflecting"
                self._print_status(
                    f"Reflection: {event.decision.upper()} -- "
                    f"{event.reasoning[:60]}"
                )

            case "tool_result":
                self.sources_found += event.accepted_source_count

            case "synthesis_started":
                self.phase = "synthesizing"
                self._print_status(
                    f"Synthesizing from {event.total_sources} sources, "
                    f"{event.total_observations} observations"
                )

            case "agent_stream_chunk":
                # Stream final output character by character
                sys.stdout.write(event.chunk)
                sys.stdout.flush()

            case "token_usage":
                self.total_tokens += event.total_tokens

            case "workflow_completed":
                print()  # newline after streamed output
                self._print_status(
                    f"Complete! {event.total_tokens} tokens, "
                    f"{event.total_sources} sources, "
                    f"{event.duration_ms / 1000:.1f}s"
                )

    def _print_status(self, message: str) -> None:
        bar = f"[{self.phase.upper():^14}]"
        step_info = (
            f" Step {self.current_step}/{self.total_steps}"
            if self.total_steps > 0
            else ""
        )
        source_info = f" | {self.sources_found} sources"
        token_info = f" | {self.total_tokens} tokens" if self.total_tokens else ""
        print(f"\r{bar}{step_info}{source_info}{token_info} | {message}")


# Usage
tracker = ProgressTracker()

async for event in runner.stream("deep_research.yaml", query="..."):
    tracker.handle(event)
```

## SSE (Server-Sent Events) Integration

Bridge framework events to SSE for web frontends:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from databricks_deep_research import WorkflowRunner

app = FastAPI()

# Create the runner at startup
runner = WorkflowRunner.from_databricks()


@app.post("/api/research")
async def research(query: str):
    async def event_generator():
        async for event in runner.stream("deep_research.yaml", query=query):
            # Every event is a Pydantic model, so serialization is built-in
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

### Frontend consumption (TypeScript)

```typescript
const eventSource = new EventSource("/api/research?query=...");

eventSource.onmessage = (msg) => {
  const event = JSON.parse(msg.data);

  switch (event.event_type) {
    case "plan_created":
      displayPlan(event.steps);
      break;
    case "agent_stream_chunk":
      appendToOutput(event.chunk);
      break;
    case "tool_call":
      showToolActivity(event.tool_name);
      break;
    case "workflow_completed":
      showComplete(event.total_tokens, event.duration_ms);
      eventSource.close();
      break;
  }
};
```

### Named SSE events

You can use named SSE event types for more targeted frontend handlers:

```python
async def event_generator():
    async for event in runner.stream("deep_research.yaml", query=query):
        # Use event_type as the SSE event name
        data = event.model_dump_json()
        yield f"event: {event.event_type}\ndata: {data}\n\n"
```

```typescript
// Frontend listens to specific event types
source.addEventListener("plan_created", (e) => {
  const plan = JSON.parse(e.data);
  displayPlan(plan.steps);
});

source.addEventListener("agent_stream_chunk", (e) => {
  const chunk = JSON.parse(e.data);
  appendToOutput(chunk.chunk);
});
```

## Event Filtering

Common filter patterns for different use cases:

### Domain events only

High-level research flow decisions:

```python
DOMAIN_EVENTS = {
    "coordinator_classified",
    "plan_created",
    "reflection_decision",
    "background_completed",
    "synthesis_started",
}

async for event in runner.stream(workflow, query=query):
    if event.event_type in DOMAIN_EVENTS:
        log_domain_event(event)
```

### Tool events only

Monitor external calls (search, crawl, vector search):

```python
TOOL_EVENTS = {"tool_call", "tool_result", "tool_cache_hit"}

async for event in runner.stream(workflow, query=query):
    if event.event_type in TOOL_EVENTS:
        log_tool_activity(event)
```

### Progress events

Track node-level execution for dashboards:

```python
PROGRESS_EVENTS = {
    "node_started",
    "node_completed",
    "item_started",
    "item_completed",
    "loop_iteration",
}

async for event in runner.stream(workflow, query=query):
    if event.event_type in PROGRESS_EVENTS:
        update_dashboard(event)
```

### Error events

Alert on failures and budget exhaustion:

```python
ERROR_EVENTS = {
    "node_error",
    "node_skipped",
    "token_budget_exceeded",
}

async for event in runner.stream(workflow, query=query):
    if event.event_type in ERROR_EVENTS:
        alert(event)
```

### Verification events (citation pipeline)

Track citation verification in detail:

```python
VERIFICATION_EVENTS = {
    "claim_generated",
    "claim_verified",
    "citation_corrected",
    "numeric_claim_detected",
    "verification_summary",
}

async for event in runner.stream(workflow, query=query):
    if event.event_type in VERIFICATION_EVENTS:
        track_verification(event)
```

### Combined filter helper

```python
def filter_events(event_types: set[str]):
    """Create an async filter for specific event types."""
    async def filtered(stream):
        async for event in stream:
            if event.event_type in event_types:
                yield event
    return filtered

# Usage
domain_only = filter_events(DOMAIN_EVENTS)
async for event in domain_only(runner.stream(workflow, query=query)):
    print(event)
```

## Token Usage Tracking

Aggregate `token_usage` events for cost monitoring:

```python
from dataclasses import dataclass


@dataclass
class TokenTracker:
    """Aggregate token usage across the entire workflow."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def track(self, event) -> None:
        if event.event_type == "token_usage":
            self.prompt_tokens += event.prompt_tokens
            self.completion_tokens += event.completion_tokens
            self.total_tokens += event.total_tokens
            self.call_count += 1

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate (adjust rates per model)."""
        # Example rates -- adjust for your model pricing
        prompt_rate = 0.003 / 1000   # $0.003 per 1K prompt tokens
        completion_rate = 0.015 / 1000  # $0.015 per 1K completion tokens
        return (
            self.prompt_tokens * prompt_rate
            + self.completion_tokens * completion_rate
        )

    def summary(self) -> str:
        return (
            f"LLM calls: {self.call_count} | "
            f"Prompt: {self.prompt_tokens:,} | "
            f"Completion: {self.completion_tokens:,} | "
            f"Total: {self.total_tokens:,} | "
            f"Est. cost: ${self.cost_estimate_usd:.4f}"
        )


# Usage
tracker = TokenTracker()

async for event in runner.stream(workflow, query=query):
    tracker.track(event)

print(tracker.summary())
```

### Budget monitoring

React to budget limits before they cause failures:

```python
async for event in runner.stream(workflow, query=query):
    if event.event_type == "token_usage":
        if event.budget_remaining != -1 and event.budget_remaining < 5000:
            print(f"WARNING: Only {event.budget_remaining} tokens remaining")

    if event.event_type == "token_budget_exceeded":
        print(f"BUDGET EXCEEDED: used {event.used} of {event.limit}")
```

## Event Serialization

All events are Pydantic models, so serialization is built-in and type-safe:

```python
# To dict
data = event.model_dump()

# To JSON string
json_str = event.model_dump_json()

# From JSON (type-safe via discriminated union)
from pydantic import TypeAdapter
from databricks_deep_research import FrameworkEvent

adapter = TypeAdapter(FrameworkEvent)
restored_event = adapter.validate_json(json_str)

# The discriminator ensures the correct type is instantiated
assert type(restored_event).__name__ == "PlanCreatedEvent"
```

### Storing events for replay

```python
import json

# Collect events during a run
all_events = []
async for event in runner.stream(workflow, query=query):
    all_events.append(event.model_dump_json())
    handle_event(event)

# Save to file
with open("events.jsonl", "w") as f:
    for event_json in all_events:
        f.write(event_json + "\n")

# Replay later
adapter = TypeAdapter(FrameworkEvent)
with open("events.jsonl") as f:
    for line in f:
        event = adapter.validate_json(line)
        handle_event(event)
```

## See Also

- [Events Concept](../concepts/events.md) -- Event categories and design rationale
- [Event Types Reference](../reference/event-types-reference.md) -- Full field-level reference for every event type
- [Architecture](../concepts/architecture.md) -- How events fit into the execution model
