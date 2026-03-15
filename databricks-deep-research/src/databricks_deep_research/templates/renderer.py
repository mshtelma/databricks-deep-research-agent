"""SafeTemplateRenderer — restricted template language for user-provided templates.

Supports ONLY:
  {variable}                      - variable substitution
  {%if variable%}...{%endif%}     - conditional blocks (truthy check)
  {%for item in items%}...{%endfor%} - iteration over lists

Security invariants:
  - Variable names MUST match [a-zA-Z_][a-zA-Z0-9_]*
  - No expression evaluation, no attribute traversal, no method calls
  - Rejects templates containing __, ., [, (, ) inside braces
  - Context values stringified via str()
  - Max nesting depth: 3. Max loop iterations: 1000.
"""

from __future__ import annotations

import re
from typing import Any

# Variable reference: {name}
_VAR = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Control flow blocks
_IF_OPEN = re.compile(r"\{%\s*if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*%\}")
_IF_CLOSE = re.compile(r"\{%\s*endif\s*%\}")
_FOR_OPEN = re.compile(r"\{%\s*for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*%\}")
_FOR_CLOSE = re.compile(r"\{%\s*endfor\s*%\}")

# Dangerous patterns inside { ... } — attribute traversal, method calls, dunders
_DANGEROUS_BRACE = re.compile(
    r"\{[^}]*(__|\.|\[|\(|\))[^}]*\}"
)

MAX_NESTING_DEPTH = 3
MAX_LOOP_ITERATIONS = 1000


class TemplateSecurityError(Exception):
    """Raised when a template contains forbidden patterns."""


class TemplateRenderError(Exception):
    """Raised when template rendering fails."""


class SafeTemplateRenderer:
    """Simple template renderer with security guarantees.

    No Jinja2. No expression evaluation. No attribute traversal.
    Only simple variable substitution and basic control flow.
    """

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render a template with variable substitution and control flow.

        Args:
            template: Template string using safe syntax.
            variables: Variable name to value mapping.

        Returns:
            Rendered string.

        Raises:
            TemplateSecurityError: If template contains forbidden patterns.
            TemplateRenderError: If rendering fails (unbalanced blocks, excess nesting).
        """
        self._validate(template)
        return self._render_block(template, variables, depth=0)

    def extract_variables(self, template: str) -> set[str]:
        """Extract variable names referenced in a template.

        Strips ``{{`` / ``}}`` literal-brace escapes before scanning,
        consistent with ``render()`` and ``_validate()``.
        """
        # Strip literal-brace escapes before scanning (same as _validate)
        cleaned = template.replace("{{", "").replace("}}", "")
        names: set[str] = set()
        for m in _VAR.finditer(cleaned):
            names.add(m.group(1))
        for m in _IF_OPEN.finditer(cleaned):
            names.add(m.group(1))
        for m in _FOR_OPEN.finditer(cleaned):
            names.add(m.group(2))  # the iterable, not the loop var
        return names

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(self, template: str) -> None:
        """Reject templates with dangerous patterns inside braces.

        ``{{`` and ``}}`` are treated as literal brace escapes (standard in
        many template languages) and stripped before validation so that JSON
        schema examples embedded in prompts are not flagged.
        """
        # Strip literal-brace escapes before checking for dangerous patterns.
        sanitised = template.replace("{{", "").replace("}}", "")
        m = _DANGEROUS_BRACE.search(sanitised)
        if m:
            raise TemplateSecurityError(
                f"Forbidden pattern in template: {m.group(0)!r}"
            )

    def _render_block(
        self,
        template: str,
        variables: dict[str, Any],
        depth: int,
    ) -> str:
        if depth > MAX_NESTING_DEPTH:
            raise TemplateRenderError(
                f"Nesting depth exceeds maximum ({MAX_NESTING_DEPTH})"
            )
        result = self._process_for_blocks(template, variables, depth)
        result = self._process_if_blocks(result, variables, depth)
        result = self._substitute_variables(result, variables)
        return result

    def _process_for_blocks(
        self,
        template: str,
        variables: dict[str, Any],
        depth: int,
    ) -> str:
        """Process {%for item in items%}...{%endfor%} blocks."""
        while True:
            match = _FOR_OPEN.search(template)
            if not match:
                break
            item_name = match.group(1)
            list_name = match.group(2)
            start = match.start()

            end_match = self._find_matching_close(
                template, match.end(), _FOR_OPEN, _FOR_CLOSE, "for",
            )
            body = template[match.end() : end_match.start()]
            end = end_match.end()

            items = variables.get(list_name, [])
            if not isinstance(items, (list, tuple)):
                items = []

            parts: list[str] = []
            for item in items[:MAX_LOOP_ITERATIONS]:
                inner = {**variables, item_name: item}
                parts.append(self._render_block(body, inner, depth + 1))

            template = template[:start] + "".join(parts) + template[end:]
        return template

    def _process_if_blocks(
        self,
        template: str,
        variables: dict[str, Any],
        depth: int,
    ) -> str:
        """Process {%if variable%}...{%endif%} blocks."""
        while True:
            match = _IF_OPEN.search(template)
            if not match:
                break
            var_name = match.group(1)
            start = match.start()

            end_match = self._find_matching_close(
                template, match.end(), _IF_OPEN, _IF_CLOSE, "if",
            )
            body = template[match.end() : end_match.start()]
            end = end_match.end()

            value = variables.get(var_name)
            replacement = self._render_block(body, variables, depth + 1) if value else ""

            template = template[:start] + replacement + template[end:]
        return template

    @staticmethod
    def _find_matching_close(
        template: str,
        search_start: int,
        open_pattern: re.Pattern[str],
        close_pattern: re.Pattern[str],
        block_name: str,
    ) -> re.Match[str]:
        """Find the matching close tag, respecting nesting."""
        nesting = 1
        pos = search_start
        while nesting > 0:
            next_open = open_pattern.search(template, pos)
            next_close = close_pattern.search(template, pos)
            if next_close is None:
                raise TemplateRenderError(f"Unmatched {{%{block_name}%}} block")
            if next_open is not None and next_open.start() < next_close.start():
                nesting += 1
                pos = next_open.end()
            else:
                nesting -= 1
                if nesting == 0:
                    return next_close
                pos = next_close.end()
        # Unreachable, but satisfies type checker
        raise TemplateRenderError(f"Unmatched {{%{block_name}%}} block")  # pragma: no cover

    @staticmethod
    def _substitute_variables(template: str, variables: dict[str, Any]) -> str:
        """Replace {variable} with str(value), or empty string if missing.

        ``{{`` and ``}}`` are converted to literal ``{`` and ``}`` after
        variable substitution (standard escape convention).
        """
        # Temporarily replace {{ / }} with placeholders so _VAR doesn't match them.
        _OPEN_ESC = "\x00LBRACE\x00"
        _CLOSE_ESC = "\x00RBRACE\x00"
        template = template.replace("{{", _OPEN_ESC).replace("}}", _CLOSE_ESC)

        def _replacer(match: re.Match[str]) -> str:
            name = match.group(1)
            value = variables.get(name)
            if value is None:
                return ""
            return str(value)

        result = _VAR.sub(_replacer, template)
        return result.replace(_OPEN_ESC, "{").replace(_CLOSE_ESC, "}")
