"""Utilities for generating dynamic report templates from lightweight sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Sequence


class SectionContentType(str, Enum):
    """Guidance for how a section should be rendered in the final report."""

    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    BULLET_LIST = "bullet_list"
    CASE_STUDIES = "case_studies"
    DATA_DEEP_DIVE = "data_deep_dive"


@dataclass(frozen=True)
class DynamicSection:
    """Minimal section descriptor used to build markdown templates."""

    title: str
    purpose: str
    priority: int = 100
    content_type: SectionContentType = SectionContentType.ANALYSIS
    hints: Sequence[str] = field(default_factory=tuple)
    step_ids: Sequence[str] = field(default_factory=tuple)  # Direct step references for observation filtering

    def prompt_block(self) -> str:
        """Render this section into a markdown block with inline instructions."""

        guidance: List[str] = []

        if self.purpose:
            guidance.append(self.purpose)

        if self.content_type == SectionContentType.COMPARISON:
            guidance.append(
                "Use properly formatted markdown tables with pipe delimiters and separator row. "
                "Example format: | Column 1 | Column 2 |\\n| --- | --- |\\n| Data 1 | Data 2 |"
            )
        elif self.content_type == SectionContentType.TIMELINE:
            guidance.append("Lay out events chronologically, highlighting inflection points.")
        elif self.content_type == SectionContentType.BULLET_LIST:
            guidance.append("Summarise the key items using concise bullet points.")
        elif self.content_type == SectionContentType.CASE_STUDIES:
            guidance.append("Highlight specific case studies with context and outcomes.")
        elif self.content_type == SectionContentType.DATA_DEEP_DIVE:
            guidance.append("Surface quantitative metrics and supporting methodology details.")

        guidance.extend(self.hints)

        if not guidance:
            guidance.append("Provide a well-supported narrative for this section.")

        rendered_guidance = " ".join(f"{item.strip()}" for item in guidance if item and item.strip())

        return f"## {self.title}\n[{rendered_guidance}]\n"


class ReportTemplateGenerator:
    """Builds markdown templates from dynamic sections and global guidance."""

    _STANDARD_HEADER = (
        "# {title}\n\n"
        "## Key Points\n"
        "[Bullet the 3-5 most important takeaways with explicit numbers where available.]\n\n"
        "## Overview\n"
        "[Issue a concise overview that frames why this topic matters now.]\n\n"
    )

    _STANDARD_FOOTER = (
        "\n## Key Citations\n"
        "[List every referenced source using '- [Title](URL)' format, leave a blank line between items.]\n"
    )

    def build_template(
        self,
        *,
        title: str,
        sections: Iterable[DynamicSection],
        include_appendix: bool = False,
    ) -> str:
        """Create a markdown template string covering standard and dynamic sections."""

        sorted_sections = sorted(sections, key=lambda section: section.priority)
        body_blocks = [section.prompt_block() for section in sorted_sections]

        appendix_block = (
            "\n## Appendix\n"
            "[Add extended notes, raw data excerpts, or methodology clarifications here when useful.]\n"
        ) if include_appendix else ""

        template = (
            self._STANDARD_HEADER.format(title=title.strip() or "Research Report")
            + "\n".join(body_blocks)
            + appendix_block
            + self._STANDARD_FOOTER
        )

        return template

