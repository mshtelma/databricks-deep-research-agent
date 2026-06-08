"""Unit tests for the inline-header normalizer (#6 fix).

Synthesizer LLMs occasionally emit ``### Header`` mid-paragraph without the
leading ``\n\n``. The frontend markdown renderer treats this as a paragraph
continuation. ``_ensure_header_breaks`` inserts the missing newlines so the
header reflows correctly.

The helper MUST skip fenced code blocks so example code that mentions
``# inside`` (e.g., a YAML config snippet, Python comment) is preserved
verbatim.
"""

from databricks_deep_research.citation.pipeline import _ensure_header_breaks


def test_inserts_double_newline_before_inline_header() -> None:
    text = "... margin of 8% [0]. ## Market Opportunity and TAM\n"
    fixed = _ensure_header_breaks(text)
    assert "\n\n## Market Opportunity" in fixed
    assert "8% [0]." in fixed


def test_handles_multiple_inline_headers() -> None:
    text = "Body text. ## Heading 1 More body. ### Heading 2 end."
    fixed = _ensure_header_breaks(text)
    assert "\n\n## Heading 1" in fixed
    assert "\n\n### Heading 2" in fixed


def test_idempotent_for_already_well_formed_headers() -> None:
    text = "Body text.\n\n## Heading 1\n\nMore body.\n"
    fixed = _ensure_header_breaks(text)
    assert fixed == text


def test_preserves_hash_inside_code_block() -> None:
    text = "Body text.\n\n```python\n# comment with hash\nx = 1  # inline\n```\nAfter."
    fixed = _ensure_header_breaks(text)
    # The hash inside the fenced block must NOT have extra newlines inserted.
    assert "```python\n# comment with hash\nx = 1  # inline\n```" in fixed


def test_handles_h1_to_h6() -> None:
    for level in range(1, 7):
        marker = "#" * level
        text = f"Body. {marker} Title at level {level}"
        fixed = _ensure_header_breaks(text)
        assert f"\n\n{marker} Title at level {level}" in fixed


def test_no_op_on_empty_string() -> None:
    assert _ensure_header_breaks("") == ""


def test_no_op_when_text_is_only_a_header() -> None:
    text = "## Standalone header\n"
    fixed = _ensure_header_breaks(text)
    assert fixed == text
