"""
Report style definitions and templates for different output formats.

Based on deer-flow's multi-style report generation approach.
"""

from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class ReportStyle(str, Enum):
    """Available report styles for research output."""
    ACADEMIC = "academic"
    POPULAR_SCIENCE = "popular_science"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"


class CitationStyle(str, Enum):
    """Available citation styles for references."""
    APA = "APA"
    MLA = "MLA"
    CHICAGO = "Chicago"
    IEEE = "IEEE"


class StyleConfig(BaseModel):
    """Configuration for a specific report style."""
    
    style: ReportStyle
    tone: str = Field(description="Writing tone for this style")
    structure: List[str] = Field(description="Expected sections in order")
    citation_format: CitationStyle = Field(description="Citation format to use")
    length_guideline: str = Field(description="Target length for the report")
    language_complexity: str = Field(description="Language complexity level")
    use_technical_terms: bool = Field(description="Whether to use technical terminology")
    include_visuals: bool = Field(description="Whether to suggest visual elements")
    key_features: List[str] = Field(description="Key features of this style")


# Style configurations
STYLE_CONFIGS: Dict[ReportStyle, StyleConfig] = {
    ReportStyle.ACADEMIC: StyleConfig(
        style=ReportStyle.ACADEMIC,
        tone="Formal, objective, and analytical",
        structure=[
            "Abstract",
            "Introduction",
            "Literature Review",
            "Methodology",
            "Findings",
            "Discussion",
            "Conclusion",
            "References"
        ],
        citation_format=CitationStyle.APA,
        length_guideline="3000-5000 words",
        language_complexity="High - academic vocabulary",
        use_technical_terms=True,
        include_visuals=True,
        key_features=[
            "Extensive citations",
            "Peer-reviewed sources preferred",
            "Critical analysis",
            "Theoretical framework",
            "Methodological rigor"
        ]
    ),
    
    ReportStyle.POPULAR_SCIENCE: StyleConfig(
        style=ReportStyle.POPULAR_SCIENCE,
        tone="Engaging, accessible, and enthusiastic",
        structure=[
            "Hook/Introduction",
            "Background Context",
            "Main Discoveries",
            "Why It Matters",
            "What's Next",
            "Key Takeaways"
        ],
        citation_format=CitationStyle.APA,  # Popular science uses inline mentions
        length_guideline="1500-2500 words",
        language_complexity="Medium - accessible to general audience",
        use_technical_terms=False,
        include_visuals=True,
        key_features=[
            "Analogies and metaphors",
            "Real-world applications",
            "Story-telling elements",
            "Expert quotes",
            "Visual explanations"
        ]
    ),
    
    ReportStyle.NEWS: StyleConfig(
        style=ReportStyle.NEWS,
        tone="Objective, concise, and informative",
        structure=[
            "Headline",
            "Lead paragraph (who, what, when, where, why)",
            "Supporting details",
            "Context and background",
            "Expert opinions",
            "Implications",
            "Conclusion"
        ],
        citation_format=CitationStyle.APA,  # News uses AP style inline
        length_guideline="800-1200 words",
        language_complexity="Medium - newspaper reading level",
        use_technical_terms=False,
        include_visuals=True,
        key_features=[
            "Inverted pyramid structure",
            "Multiple sources",
            "Balanced reporting",
            "Timely information",
            "Clear attribution"
        ]
    ),
    
    ReportStyle.SOCIAL_MEDIA: StyleConfig(
        style=ReportStyle.SOCIAL_MEDIA,
        tone="Conversational, engaging, and shareable",
        structure=[
            "Hook/Attention grabber",
            "Key points (bullet format)",
            "Main insight",
            "Call to action",
            "Hashtags/Links"
        ],
        citation_format=CitationStyle.APA,  # Social media uses links
        length_guideline="280-500 characters per section",
        language_complexity="Low - highly accessible",
        use_technical_terms=False,
        include_visuals=True,
        key_features=[
            "Bite-sized information",
            "Emojis and formatting",
            "Shareable quotes",
            "Thread structure",
            "Visual content priority"
        ]
    ),
    
    ReportStyle.PROFESSIONAL: StyleConfig(
        style=ReportStyle.PROFESSIONAL,
        tone="Professional, clear, and actionable",
        structure=[
            "Executive Summary",
            "Introduction",
            "Analysis",
            "Key Findings",
            "Recommendations",
            "Conclusion",
            "Appendices"
        ],
        citation_format=CitationStyle.CHICAGO,  # Professional reports use Chicago
        length_guideline="2000-3000 words",
        language_complexity="Medium-High - professional audience",
        use_technical_terms=True,
        include_visuals=True,
        key_features=[
            "Data-driven insights",
            "Clear recommendations",
            "Risk assessment",
            "Implementation guidance",
            "Executive-friendly format"
        ]
    ),
    
    ReportStyle.TECHNICAL: StyleConfig(
        style=ReportStyle.TECHNICAL,
        tone="Precise, detailed, and systematic",
        structure=[
            "Overview",
            "Technical Background",
            "Detailed Analysis",
            "Implementation Details",
            "Performance Metrics",
            "Technical Specifications",
            "Conclusion"
        ],
        citation_format=CitationStyle.IEEE,  # Technical reports use IEEE
        length_guideline="2500-4000 words",
        language_complexity="High - technical audience",
        use_technical_terms=True,
        include_visuals=True,
        key_features=[
            "Code examples",
            "Technical diagrams",
            "Performance benchmarks",
            "Architecture details",
            "API documentation"
        ]
    ),
    
    ReportStyle.EXECUTIVE: StyleConfig(
        style=ReportStyle.EXECUTIVE,
        tone="Strategic, concise, and decision-focused",
        structure=[
            "Executive Brief (1 page)",
            "Strategic Context",
            "Key Insights",
            "Business Impact",
            "Recommendations",
            "Next Steps"
        ],
        citation_format=CitationStyle.APA,  # Executive summaries minimal citations
        length_guideline="1000-1500 words",
        language_complexity="Medium - C-suite audience",
        use_technical_terms=False,
        include_visuals=True,
        key_features=[
            "Bottom-line focus",
            "Strategic implications",
            "ROI considerations",
            "Risk-opportunity balance",
            "Action-oriented"
        ]
    )
}


class StyleTemplate:
    """Templates for generating reports in different styles."""
    
    def __init__(self, **kwargs):
        """Initialize a style template with custom configuration."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Set defaults if not provided
        if not hasattr(self, 'citation_style'):
            self.citation_style = CitationStyle.APA
    
    @staticmethod
    def get_style_prompt(style: ReportStyle) -> str:
        """Get the prompt template for a specific style."""
        config = STYLE_CONFIGS[style]
        
        return f"""
You are writing a {config.style.value.replace('_', ' ')} style report.

TONE: {config.tone}

STRUCTURE:
{chr(10).join(f"- {section}" for section in config.structure)}

CITATION FORMAT: {config.citation_format}
LENGTH: {config.length_guideline}
LANGUAGE: {config.language_complexity}
TECHNICAL TERMS: {'Use' if config.use_technical_terms else 'Avoid'} technical terminology

KEY FEATURES TO INCLUDE:
{chr(10).join(f"- {feature}" for feature in config.key_features)}

Remember to:
1. Maintain the appropriate tone throughout
2. Follow the expected structure
3. Use citations in the specified format
4. Keep within the length guidelines
5. Match the language complexity to the target audience
"""

    @staticmethod
    def get_section_template(style: ReportStyle, section: str) -> str:
        """Get template for a specific section of a style."""
        templates = {
            (ReportStyle.ACADEMIC, "Abstract"): """
Write a 150-250 word abstract that includes:
- Research objective
- Methodology overview
- Key findings
- Implications
- Keywords (3-5)
""",
            (ReportStyle.POPULAR_SCIENCE, "Hook/Introduction"): """
Start with an attention-grabbing opening that:
- Poses an intriguing question OR
- Shares a surprising fact OR
- Tells a brief story
- Connects to readers' everyday experience
- Sets up the "wow factor"
""",
            (ReportStyle.NEWS, "Lead paragraph"): """
Write a lead paragraph (50-75 words) that answers:
- WHO is involved?
- WHAT happened/was discovered?
- WHEN did it occur?
- WHERE did it take place?
- WHY is it significant?
- HOW does it impact readers?
""",
            (ReportStyle.PROFESSIONAL, "Executive Summary"): """
Provide a one-page executive summary with:
- Context (2-3 sentences)
- Key findings (3-5 bullet points)
- Business impact (2-3 sentences)
- Recommended actions (3-5 bullet points)
- Timeline and resources needed
""",
            (ReportStyle.TECHNICAL, "Technical Background"): """
Provide technical background including:
- Relevant technologies and frameworks
- Technical prerequisites
- System architecture overview
- Key technical concepts
- Performance requirements
""",
            (ReportStyle.EXECUTIVE, "Executive Brief"): """
Create a one-page executive brief with:
- Strategic question addressed (1 sentence)
- Bottom-line answer (1-2 sentences)
- 3 key insights (bullet points)
- Primary recommendation (1 sentence)
- Critical success factors (3 bullet points)
""",
            (ReportStyle.SOCIAL_MEDIA, "Hook/Attention grabber"): """
Create a social media hook:
- Start with a question, statistic, or bold statement
- Use emoji strategically 
- Keep under 280 characters
- Make it shareable and discussion-worthy
- Include relevant hashtags
"""
        }
        
        return templates.get((style, section), f"Write the {section} section appropriate for {style.value} style.")


class ReportFormatter:
    """Utility class for formatting reports in different styles."""
    
    def __init__(self):
        """Initialize the report formatter."""
        self.style_configs = STYLE_CONFIGS
    
    def get_style_template(self, style: ReportStyle) -> StyleTemplate:
        """Get the style template for a report style."""
        config = self.style_configs[style]
        return StyleTemplate(
            name=style.value.replace('_', ' ').title(),
            tone=config.tone,
            language_complexity=config.language_complexity,
            use_technical_terms=config.use_technical_terms,
            paragraph_length="long" if style == ReportStyle.ACADEMIC else "medium",
            sentence_structure="complex" if style == ReportStyle.ACADEMIC else "simple",
            citation_style=config.citation_format,
            include_abstract=style == ReportStyle.ACADEMIC,
            include_methodology=style == ReportStyle.ACADEMIC,
            include_references=True
        )
    
    def format_section(self, title: str, content: str, style: ReportStyle) -> str:
        """Format a section of the report."""
        header = self.format_section_header(title, style)
        return f"{header}\n\n{content}\n"
    
    @staticmethod
    def format_citation(citation: Dict, style: ReportStyle) -> str:
        """Format a citation according to the style guide."""
        title = citation.get('title', '')
        url = citation.get('url', '')
        author = citation.get('author', 'Unknown')
        date = citation.get('date', 'n.d.')
        
        if style == ReportStyle.ACADEMIC:
            # APA format
            return f"{author} ({date}). {title}. Retrieved from {url}"
        elif style == ReportStyle.NEWS:
            # News attribution
            return f'According to {author}, "{title}"'
        elif style == ReportStyle.SOCIAL_MEDIA:
            # Social media format
            return f"[{title}]({url})"
        elif style in [ReportStyle.PROFESSIONAL, ReportStyle.EXECUTIVE]:
            # Footnote style
            return f"{title} ({author}, {date})"
        elif style == ReportStyle.TECHNICAL:
            # IEEE style
            return f"[{citation.get('number', '1')}] {author}, \"{title},\" {date}. [Online]. Available: {url}"
        else:
            # Default inline format
            return f"{title} - {author}"
    
    @staticmethod
    def format_section_header(section: str, style: ReportStyle) -> str:
        """Format a section header according to style."""
        if style == ReportStyle.ACADEMIC:
            return f"\n## {section}\n"
        elif style == ReportStyle.SOCIAL_MEDIA:
            return f"\n🔹 {section.upper()}\n"
        elif style == ReportStyle.NEWS:
            return f"\n**{section}**\n"
        else:
            return f"\n### {section}\n"
    
    @staticmethod
    def apply_style_formatting(content: str, style: ReportStyle) -> str:
        """Apply style-specific formatting to content."""
        if style == ReportStyle.SOCIAL_MEDIA:
            # Break into tweet-sized chunks
            lines = content.split('\n')
            formatted = []
            for line in lines:
                if len(line) > 280:
                    # Break long lines
                    words = line.split()
                    current = []
                    for word in words:
                        if sum(len(w) + 1 for w in current) + len(word) <= 280:
                            current.append(word)
                        else:
                            formatted.append(' '.join(current))
                            current = [word]
                    if current:
                        formatted.append(' '.join(current))
                else:
                    formatted.append(line)
            return '\n\n'.join(formatted)
        
        return content
    
    @staticmethod
    def generate_style_metadata(style: ReportStyle) -> Dict:
        """Generate metadata for the report style."""
        config = STYLE_CONFIGS[style]
        return {
            "style": style.value,
            "tone": config.tone,
            "target_length": config.length_guideline,
            "citation_format": config.citation_format,
            "complexity": config.language_complexity,
            "sections": config.structure
        }