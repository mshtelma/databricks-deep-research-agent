# Templates and Prompts

> Customize agent prompts using the safe template renderer.

## Overview
Agent prompts support template syntax via `SafeTemplateRenderer`. Template variables are populated from workflow state and pool injections.

**Important:** The renderer is a custom restricted template engine -- it is _not_ Jinja2. It deliberately omits expression evaluation, attribute traversal, and method calls to enforce security invariants on user-provided templates.

## Template Syntax
Based on the actual `renderer.py`:
```
{variable}                              — Insert state value (empty string if missing)
{%if variable%}...{%endif%}            — Conditional block (truthy check)
{%for item in items%}...{%endfor%}     — Iteration over a list
{{                                       — Literal { (escaped)
}}                                       — Literal } (escaped)
```

### What is NOT supported
The renderer rejects any template that contains the following patterns inside braces:
- `__` (dunder access)
- `.` (attribute traversal)
- `[` (subscript access)
- `(` or `)` (method/function calls)

Attempting to use these raises a `TemplateSecurityError`.

## Available Variables
- `query`: The user's research query (always available)
- Any key written to `WorkflowState` by previous nodes
- Pool-injected content (via `pool_inject` config)
- Custom variables from `required_inputs`
- Loop variables defined in `{%for item in items%}` blocks

## SafeTemplateRenderer
The renderer at `databricks_deep_research/templates/renderer.py` provides:

- **No Jinja2 dependency** -- pure regex-based parsing with no external template engine
- **Sandboxed execution** -- no file system access, no `exec`, no attribute traversal
- **Missing variables become empty strings** -- undefined variables silently produce `""`
- **Thread-safe** -- no mutable instance state; `render()` is safe to call concurrently
- **Nesting limit** -- maximum depth of 3 for nested control-flow blocks
- **Loop limit** -- maximum of 1000 iterations per `for` block
- **Variable extraction** -- `extract_variables(template)` returns the set of variable names referenced in a template, useful for validation before rendering

### Error Types
| Error | Cause |
|-------|-------|
| `TemplateSecurityError` | Template contains forbidden patterns (`__`, `.`, `[`, `(`, `)` inside braces) |
| `TemplateRenderError` | Unbalanced block tags or nesting depth exceeded |

### Programmatic Usage
```python
from databricks_deep_research.templates.renderer import SafeTemplateRenderer

renderer = SafeTemplateRenderer()

# Simple variable substitution
result = renderer.render(
    "Research topic: {query}",
    {"query": "quantum computing"},
)
# => "Research topic: quantum computing"

# Conditional blocks
result = renderer.render(
    "{%if observations%}Previous findings:\n{observations}{%endif%}",
    {"observations": "Found 3 relevant papers."},
)
# => "Previous findings:\nFound 3 relevant papers."

# Iteration
result = renderer.render(
    "Steps:\n{%for step in steps%}- {step}\n{%endfor%}",
    {"steps": ["Search web", "Analyze results", "Synthesize"]},
)
# => "Steps:\n- Search web\n- Analyze results\n- Synthesize\n"

# Extract referenced variables (useful for validation)
vars = renderer.extract_variables("Hello {name}, topic: {query}")
# => {"name", "query"}
```

### Literal Braces
To include literal `{` and `}` characters (e.g., in JSON schema examples embedded in prompts), use doubled braces:
```
Output as JSON: {{"key": "value"}}
```
This renders as:
```
Output as JSON: {"key": "value"}
```

## Customizing Prompts in YAML
```yaml
- id: researcher
  type: agent
  config:
    subtype: researcher
    system_prompt: |
      You are a {domain} research specialist.
      Focus on quantitative evidence.
      Avoid opinion pieces.
    user_prompt_template: |
      ## Research Step
      {current_step}

      ## Query
      {query}

      {%if observations%}
      ## Previous Findings
      {observations}
      {%endif%}
```

### Conditional Sections
Use `{%if var%}` to include prompt sections only when a variable is truthy:
```yaml
user_prompt_template: |
  {query}

  {%if constraints%}
  ## Constraints
  {constraints}
  {%endif%}

  {%if sources%}
  ## Required Sources
  {%for source in sources%}- {source}
  {%endfor%}{%endif%}
```

## Best Practices
- Keep prompts concise and focused
- Use `system_prompt` for persona and rules
- Use `user_prompt_template` for dynamic content
- Use `{%if var%}` blocks for optional sections rather than relying on empty-string substitution
- Use `extract_variables()` to verify your template references the variables you expect
- Use `{{` and `}}` to embed literal braces (e.g., JSON examples) in prompts
- Avoid deeply nested blocks -- the renderer enforces a maximum depth of 3

## Prompt Debugging
If a template is not rendering as expected:

1. **Check variable names** -- variable names must match `[a-zA-Z_][a-zA-Z0-9_]*`. No dots, no subscripts.
2. **Use `extract_variables()`** -- compare the returned set against your context dict to find mismatches:
   ```python
   renderer = SafeTemplateRenderer()
   expected = renderer.extract_variables(template)
   provided = set(variables.keys())
   missing = expected - provided
   if missing:
       print(f"Missing variables: {missing}")
   ```
3. **Watch for TemplateSecurityError** -- if your template includes JSON or code samples with `.` or `()`, wrap them in literal brace escapes (`{{` / `}}`).
4. **Check block balance** -- every `{%if ...%}` needs `{%endif%}` and every `{%for ...%}` needs `{%endfor%}`. Unbalanced blocks raise `TemplateRenderError`.
5. **Nesting depth** -- if you hit a depth error, flatten your template by extracting inner blocks into separate variables rendered beforehand.

## See Also
- [Agent System](../concepts/agent-system.md)
- [YAML Workflow Authoring](yaml-workflow-authoring.md)
- [Builtin Agents](builtin-agents.md)
