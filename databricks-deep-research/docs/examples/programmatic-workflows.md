# Programmatic Workflows

> Build workflows in Python code instead of YAML.

## Overview
While YAML is the primary way to define workflows, you can also construct WorkflowDefinition objects directly in Python. This is useful for dynamic workflow generation, testing, and programmatic customization.

## Basic Example
```python
from databricks_deep_research import (
    WorkflowDefinition, WorkflowNode, NodeType,
)
from databricks_deep_research.workflow.definition import ToolDeclaration

workflow = WorkflowDefinition(
    id="programmatic-research",
    name="programmatic-research",
    version=1,
    tools=[
        ToolDeclaration(name="web_search", kind="web_search"),
        ToolDeclaration(name="web_crawl", kind="web_crawl"),
    ],
    pools=[
        {"name": "sources", "item_type": "source", "dedup_key": "url"},
        {"name": "observations", "item_type": "text", "dedup_content_hash": True},
    ],
    root=WorkflowNode(
        id="pipeline",
        type=NodeType.sequence,
        label="Main pipeline",
        children=[
            WorkflowNode(
                id="plan",
                type=NodeType.agent,
                label="Plan research",
                config={"subtype": "planner"},
            ),
            WorkflowNode(
                id="research",
                type=NodeType.agent,
                label="Execute research",
                config={
                    "subtype": "researcher",
                    "tools": ["web_search", "web_crawl"],
                    "pool_writes": [
                        {"pool": "sources", "extract": "sources"},
                        {"pool": "observations", "extract": "findings"},
                    ],
                },
            ),
            WorkflowNode(
                id="synthesize",
                type=NodeType.agent,
                label="Synthesize findings",
                config={
                    "subtype": "synthesizer",
                    "pool_inject": [
                        {"pool": "sources", "format": "markdown"},
                        {"pool": "observations", "format": "text"},
                    ],
                },
            ),
        ],
    ),
)
```

## Dynamic Workflow Generation
Show how to build workflows dynamically:
```python
def build_workflow(tools: list[str], depth: str) -> WorkflowDefinition:
    # Build tool declarations
    tool_decls = [ToolDeclaration(name=t, kind=t) for t in tools]

    # Adjust iterations based on depth
    max_iter = {"light": 3, "standard": 5, "deep": 10}[depth]

    # Build the workflow
    return WorkflowDefinition(
        id=f"{depth}-research",
        name=f"{depth}-research",
        root=WorkflowNode(
            id="root",
            type=NodeType.sequence,
            label=f"{depth.title()} research pipeline",
            children=[...],
        ),
        tools=tool_decls,
    )
```

## Saving to YAML
```python
from databricks_deep_research import save_workflow

save_workflow(workflow, "my_workflow.yaml")
```

## Hybrid Approach
Load a YAML template, modify it programmatically:
```python
from databricks_deep_research import load_workflow

workflow = load_workflow("examples/simple_research.yaml")
# Modify the workflow object...
workflow.token_budget = 50000
# Use the modified workflow
```

## When to Use Programmatic Workflows
- Dynamic tool selection based on user config
- Parameterized workflows (depth, model tier)
- Testing (construct minimal workflows for unit tests)
- Workflow generators (build from UI input)

## When to Use YAML
- Standard research pipelines
- Shared team workflows
- Version-controlled configurations
- Documentation and readability

## See Also
- [Workflow Engine](../concepts/workflow-engine.md)
- [YAML Workflow Authoring](../guides/yaml-workflow-authoring.md)
- [Workflow Definition Schema](../reference/workflow-definition-schema.md)
- [Testing](../guides/testing.md)
