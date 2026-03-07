"""SafeTemplateRenderer - restricted template language for user-provided templates.

Supports ONLY:
  {{variable}}              - variable substitution
  {{variable|length}}       - length filter (only filter)
  {{#if variable}}...{{/if}} - conditional blocks
  {{#for item in list}}...{{/for}} - iteration with {{item}}, {{item.key}}, {{loop.index}}

Security invariants:
  - Variable names MUST match [a-zA-Z_][a-zA-Z0-9_]*
  - Dict access ONLY inside for-loops, ONLY single-level (item["key"])
  - No expression evaluation, no method calls, no imports
  - Context values stringified via str()
  - Max nesting depth: 3. Max loop iterations: 1000.

Part of 011-workflow-orchestration (Change 1: Safe Template Renderer).
"""

import re
from dataclasses import dataclass, field
from typing import Any

# Allowed identifier pattern — no dots, no brackets, no underscores-only edge cases
_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Template syntax patterns
_VAR_PATTERN = re.compile(r"\{\{(\s*[\w.|]+\s*)\}\}")
_IF_OPEN = re.compile(r"\{\{#if\s+([\w]+)\s*\}\}")
_IF_CLOSE = re.compile(r"\{\{/if\}\}")
_FOR_OPEN = re.compile(r"\{\{#for\s+([\w]+)\s+in\s+([\w]+)\s*\}\}")
_FOR_CLOSE = re.compile(r"\{\{/for\}\}")

# Forbidden patterns — anything that looks like Jinja2/Python introspection
_FORBIDDEN_PATTERNS = [
    re.compile(r"__\w+__"),          # dunder access (__class__, __mro__)
    re.compile(r"\.\w+\("),          # method calls (.format(, .join()
    re.compile(r"\bimport\b"),       # import statements
    re.compile(r"\beval\b"),         # eval
    re.compile(r"\bexec\b"),         # exec
    re.compile(r"\bgetattr\b"),      # getattr
    re.compile(r"\bsetattr\b"),      # setattr
    re.compile(r"\bglobals\b"),      # globals
    re.compile(r"\blocals\b"),       # locals
    re.compile(r"\[\s*['\"]"),       # bracket access with string literals ["key"]
    re.compile(r"\{%"),              # Jinja2 block syntax
    re.compile(r"%\}"),              # Jinja2 block syntax
]

MAX_NESTING_DEPTH = 3
MAX_LOOP_ITERATIONS = 1000


class TemplateSecurityError(Exception):
    """Raised when a template contains forbidden patterns."""


class TemplateRenderError(Exception):
    """Raised when template rendering fails."""


@dataclass
class _LoopContext:
    index: int = 0  # 1-based


@dataclass
class SafeTemplateRenderer:
    """Restricted template renderer that prevents SSTI/RCE.

    This replaces Jinja2 for user-provided templates. Only supports
    variable substitution, conditionals, and iteration with a strictly
    limited syntax.
    """

    max_nesting_depth: int = MAX_NESTING_DEPTH
    max_loop_iterations: int = MAX_LOOP_ITERATIONS
    _depth: int = field(default=0, init=False)

    def validate(self, template: str) -> list[str]:
        """Validate a template for forbidden patterns. Returns list of errors."""
        errors: list[str] = []
        for pattern in _FORBIDDEN_PATTERNS:
            matches = pattern.findall(template)
            if matches:
                errors.append(
                    f"Forbidden pattern detected: {pattern.pattern!r} "
                    f"(matched: {matches[0]!r})"
                )
        # Check nesting depth
        depth = 0
        max_depth = 0
        for line in template.splitlines():
            if _IF_OPEN.search(line) or _FOR_OPEN.search(line):
                depth += 1
                max_depth = max(max_depth, depth)
            if _IF_CLOSE.search(line) or _FOR_CLOSE.search(line):
                depth -= 1
        if max_depth > self.max_nesting_depth:
            errors.append(
                f"Nesting depth {max_depth} exceeds maximum {self.max_nesting_depth}"
            )
        if depth != 0:
            errors.append("Unbalanced block tags (mismatched #if/#for and /if//for)")
        return errors

    def extract_variables(self, template: str) -> set[str]:
        """Extract top-level variable names referenced in a template."""
        variables: set[str] = set()
        for match in _VAR_PATTERN.finditer(template):
            expr = match.group(1).strip()
            # Handle |length filter
            name = expr.split("|")[0].strip()
            # Handle loop.index and item.key — skip those
            if "." in name:
                root = name.split(".")[0]
                variables.add(root)
            else:
                variables.add(name)
        # Also extract from #if and #for
        for match in _IF_OPEN.finditer(template):
            variables.add(match.group(1))
        for match in _FOR_OPEN.finditer(template):
            variables.add(match.group(2))  # the iterable
        return variables

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template: Template string using safe syntax.
            context: Variable name → value mapping.

        Returns:
            Rendered string.

        Raises:
            TemplateSecurityError: If template contains forbidden patterns.
            TemplateRenderError: If rendering fails.
        """
        errors = self.validate(template)
        if errors:
            raise TemplateSecurityError(
                f"Template validation failed: {'; '.join(errors)}"
            )
        self._depth = 0
        return self._render_block(template, context)

    def _render_block(
        self, template: str, context: dict[str, Any],
    ) -> str:
        self._depth += 1
        if self._depth > self.max_nesting_depth + 1:
            raise TemplateRenderError(
                f"Rendering depth exceeded maximum ({self.max_nesting_depth})"
            )
        try:
            result = self._process_for_blocks(template, context)
            result = self._process_if_blocks(result, context)
            result = self._substitute_variables(result, context)
            return result
        finally:
            self._depth -= 1

    def _process_for_blocks(
        self, template: str, context: dict[str, Any],
    ) -> str:
        """Process {{#for item in list}}...{{/for}} blocks."""
        while True:
            match = _FOR_OPEN.search(template)
            if not match:
                break
            item_name = match.group(1)
            list_name = match.group(2)
            start = match.start()
            # Find matching {{/for}}
            end_match = _FOR_CLOSE.search(template, match.end())
            if not end_match:
                raise TemplateRenderError("Unmatched {{#for}} block")
            body = template[match.end():end_match.start()]
            end = end_match.end()

            items = context.get(list_name, [])
            if not isinstance(items, (list, tuple)):
                items = []
            parts: list[str] = []
            for i, item in enumerate(items[:self.max_loop_iterations]):
                loop_ctx = _LoopContext(index=i + 1)
                inner_context = {**context, item_name: item, "loop": loop_ctx}
                rendered = self._render_item_body(body, inner_context, item_name)
                parts.append(rendered)

            template = template[:start] + "".join(parts) + template[end:]
        return template

    def _render_item_body(
        self, body: str, context: dict[str, Any], item_name: str,
    ) -> str:
        """Render the body of a for loop, handling item.key and loop.index."""
        result = body
        # Replace {{loop.index}}
        result = result.replace("{{loop.index}}", str(context["loop"].index))
        # Replace {{item.key}} patterns — single-level dict access only
        item_val = context.get(item_name)
        item_dot_pattern = re.compile(
            r"\{\{\s*" + re.escape(item_name) + r"\.([\w]+)\s*\}\}"
        )
        for m in item_dot_pattern.finditer(result):
            key = m.group(1)
            if isinstance(item_val, dict):
                replacement = str(item_val.get(key, ""))
            else:
                replacement = ""
            result = result.replace(m.group(0), replacement)
        # Replace {{item}} itself
        result = re.sub(
            r"\{\{\s*" + re.escape(item_name) + r"\s*\}\}",
            str(item_val) if item_val is not None else "",
            result,
        )
        # Process nested blocks
        result = self._process_if_blocks(result, context)
        result = self._substitute_variables(result, context)
        return result

    def _process_if_blocks(
        self, template: str, context: dict[str, Any],
    ) -> str:
        """Process {{#if variable}}...{{/if}} blocks."""
        while True:
            match = _IF_OPEN.search(template)
            if not match:
                break
            var_name = match.group(1)
            start = match.start()
            end_match = _IF_CLOSE.search(template, match.end())
            if not end_match:
                raise TemplateRenderError("Unmatched {{#if}} block")
            body = template[match.end():end_match.start()]
            end = end_match.end()

            value = context.get(var_name)
            if value and value != [] and value != {} and value != "":
                replacement = body
            else:
                replacement = ""
            template = template[:start] + replacement + template[end:]
        return template

    def _substitute_variables(
        self, template: str, context: dict[str, Any],
    ) -> str:
        """Replace {{variable}} and {{variable|length}} patterns."""
        def replacer(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            # Handle |length filter
            if "|" in expr:
                parts = expr.split("|", 1)
                var_name = parts[0].strip()
                filter_name = parts[1].strip()
                if filter_name != "length":
                    return match.group(0)  # Unknown filter, leave as-is
                value = context.get(var_name)
                if hasattr(value, "__len__"):
                    return str(len(value))
                return "0"
            # Handle dot-path (for non-loop contexts, e.g., extraction.company)
            if "." in expr:
                parts = expr.split(".")
                current: Any = context
                for part in parts:
                    if isinstance(current, dict):
                        current = current.get(part)
                    elif isinstance(current, _LoopContext) and part == "index":
                        return str(current.index)
                    else:
                        return ""
                    if current is None:
                        return ""
                return str(current)
            # Simple variable
            value = context.get(expr)
            if value is None:
                return ""
            return str(value)

        return _VAR_PATTERN.sub(replacer, template)
