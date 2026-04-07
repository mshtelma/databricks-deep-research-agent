"""Tests for answer extraction."""

from benchmarks.core.answer_extract import XMLTagExtractor


class TestXMLTagExtractor:
    def setup_method(self) -> None:
        self.extractor = XMLTagExtractor()

    def test_basic_extraction(self) -> None:
        text = "Some analysis here. <FINAL_ANSWER>42</FINAL_ANSWER> Done."
        assert self.extractor.extract(text) == "42"

    def test_with_commas(self) -> None:
        text = "<FINAL_ANSWER>12,345</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "12,345"

    def test_percentage(self) -> None:
        text = "<FINAL_ANSWER>45.2%</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "45.2%"

    def test_case_insensitive(self) -> None:
        text = "<final_answer>42</final_answer>"
        assert self.extractor.extract(text) == "42"

    def test_mixed_case(self) -> None:
        text = "<Final_Answer>hello</Final_Answer>"
        assert self.extractor.extract(text) == "hello"

    def test_multiline(self) -> None:
        text = "<FINAL_ANSWER>\n42.5\n</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "42.5"

    def test_multiple_takes_last(self) -> None:
        text = "<FINAL_ANSWER>wrong</FINAL_ANSWER> ... <FINAL_ANSWER>correct</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "correct"

    def test_missing_returns_none(self) -> None:
        assert self.extractor.extract("The answer is 42") is None

    def test_empty_returns_none(self) -> None:
        assert self.extractor.extract("") is None

    def test_bold_markdown_fallback(self) -> None:
        text = "Analysis...\n\n**FINAL ANSWER**: 543 million\n\nEnd."
        assert self.extractor.extract(text) == "543 million"

    def test_bold_markdown_with_underscore(self) -> None:
        text = "**FINAL_ANSWER**: March 1977\n\nDone."
        assert self.extractor.extract(text) == "March 1977"

    def test_prefix_pattern_fallback(self) -> None:
        text = "Analysis complete.\n\nFinal Answer: 2,602\n\nSources:"
        assert self.extractor.extract(text) == "2,602"

    def test_text_answer_preserved(self) -> None:
        text = "<FINAL_ANSWER>March 1977</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "March 1977"

    def test_whitespace_stripped(self) -> None:
        text = "<FINAL_ANSWER>  42  </FINAL_ANSWER>"
        assert self.extractor.extract(text) == "42"

    def test_custom_tag(self) -> None:
        extractor = XMLTagExtractor(tag="ANSWER")
        text = "<ANSWER>yes</ANSWER>"
        assert extractor.extract(text) == "yes"

    def test_negative_number(self) -> None:
        text = "<FINAL_ANSWER>-1,234</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "-1,234"

    def test_decimal(self) -> None:
        text = "<FINAL_ANSWER>1608.80%</FINAL_ANSWER>"
        assert self.extractor.extract(text) == "1608.80%"
