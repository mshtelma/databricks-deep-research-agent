"""Unit tests for FileUploadService text extraction helpers."""

import builtins
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deep_research.models.uploaded_file import FileProcessingStatus, UploadedFile
from deep_research.services.file_upload_service import FileUploadService


def test_sanitize_text_content_removes_invalid_control_bytes() -> None:
    """Sanitization should strip PostgreSQL-invalid text bytes."""
    service = FileUploadService(session=MagicMock())
    raw = "Hello\x00World\x1f\nLine\tTwo\r\nDone"

    sanitized = service._sanitize_text_content(raw)

    assert sanitized == "HelloWorld\nLine\tTwo\nDone"


def test_extract_pdf_text_uses_docling_converter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PDF extraction should use Docling markdown output."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    service = FileUploadService(session=MagicMock())

    docling_module = types.ModuleType("docling")
    converter_module = types.ModuleType("docling.document_converter")

    class _FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Parsed PDF"

    class _FakeResult:
        document = _FakeDocument()

    class _FakeConverter:
        def convert(self, file_path: str) -> _FakeResult:
            assert file_path == str(pdf_path)
            return _FakeResult()

    converter_module.DocumentConverter = _FakeConverter  # type: ignore[attr-defined]
    docling_module.document_converter = converter_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "docling", docling_module)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)

    text = service._extract_pdf_text(pdf_path)

    assert text == "# Parsed PDF"


def test_extract_pdf_text_returns_none_when_docling_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PDF extraction should fail cleanly without Docling."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    service = FileUploadService(session=MagicMock())

    original_import = builtins.__import__

    def _import_with_docling_failure(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "docling.document_converter":
            raise ImportError("No module named 'docling'")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_with_docling_failure)

    text = service._extract_pdf_text(pdf_path)

    assert text is None


# =========================================================================
# UploadedFile model: total_extracted_chars
# =========================================================================


def test_mark_ready_stores_total_extracted_chars() -> None:
    """mark_ready() with total_extracted_chars stores it in metadata_."""
    f = UploadedFile(
        id=uuid4(),
        owner_id="u1",
        filename="test.txt",
        file_type="txt",
        file_size=1000,
        storage_path="/tmp/test.txt",
        processing_status=FileProcessingStatus.PENDING.value,
        metadata_={},
    )
    f.mark_ready(5, total_extracted_chars=4200)

    assert f.processing_status == FileProcessingStatus.READY.value
    assert f.chunk_count == 5
    assert f.metadata_["total_extracted_chars"] == 4200
    assert f.total_extracted_chars == 4200


def test_mark_ready_without_chars_preserves_metadata() -> None:
    """mark_ready() without total_extracted_chars doesn't touch metadata_."""
    f = UploadedFile(
        id=uuid4(),
        owner_id="u1",
        filename="test.txt",
        file_type="txt",
        file_size=1000,
        storage_path="/tmp/test.txt",
        processing_status=FileProcessingStatus.PENDING.value,
        metadata_={"existing_key": "value"},
    )
    f.mark_ready(5)

    assert f.metadata_ == {"existing_key": "value"}
    assert f.total_extracted_chars is None


def test_total_extracted_chars_property_returns_none_for_legacy() -> None:
    """total_extracted_chars returns None when not in metadata."""
    f = UploadedFile(
        id=uuid4(),
        owner_id="u1",
        filename="test.txt",
        file_type="txt",
        file_size=1000,
        storage_path="/tmp/test.txt",
        processing_status=FileProcessingStatus.READY.value,
        metadata_={},
    )
    assert f.total_extracted_chars is None

    # Also test with None metadata_
    f.metadata_ = None  # type: ignore[assignment]
    assert f.total_extracted_chars is None
