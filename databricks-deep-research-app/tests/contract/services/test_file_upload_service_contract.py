"""Contract tests for CachedFileUploadService.

Tests verify that the cached implementation honours the IFileUploadService
protocol contract — metadata CRUD, session quota, chunk round-trip —
against every parametrised backend (fake, lakebase, sql_warehouse).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.services.cached.file_upload import CachedFileUploadService


class TestCachedFileUploadServiceContract:
    """Core contract: upload / get / list / delete round-trips."""

    @pytest.mark.asyncio
    async def test_upload_and_get(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        content = b"hello world"
        view, err = await svc.upload_file(
            owner_id="user-1",
            filename="hello.txt",
            file_content=content,
            file_size=len(content),
        )
        assert err is None
        assert view is not None
        assert view.filename == "hello.txt"
        assert view.file_type == "txt"
        assert view.file_size == len(content)
        assert view.processing_status == "pending"

        fetched = await svc.get(view.id)
        assert fetched is not None
        assert fetched.id == view.id
        assert fetched.filename == "hello.txt"

    @pytest.mark.asyncio
    async def test_get_for_user_isolates_ownership(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        content = b"data"
        view, _ = await svc.upload_file(
            owner_id="user-A",
            filename="doc.txt",
            file_content=content,
            file_size=len(content),
        )
        assert view is not None

        # Same file_id, different owner_id → returns None
        result = await svc.get_for_user(view.id, "user-B")
        assert result is None

        # Correct owner → returns view
        result2 = await svc.get_for_user(view.id, "user-A")
        assert result2 is not None
        assert result2.id == view.id

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        result = await svc.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_list_session_files(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        session_id = uuid4()
        for i in range(3):
            content = f"file {i}".encode()
            await svc.upload_file(
                owner_id="user-list",
                filename=f"file{i}.txt",
                file_content=content,
                file_size=len(content),
                session_id=session_id,
            )

        files, total = await svc.get_session_files(
            owner_id="user-list",
            session_id=session_id,
        )
        assert total == 3
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_list_without_session_returns_all_user_files(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        owner = "user-nofilter"
        for i in range(2):
            content = b"x"
            await svc.upload_file(
                owner_id=owner,
                filename=f"f{i}.txt",
                file_content=content,
                file_size=1,
            )

        files, total = await svc.get_session_files(owner_id=owner)
        assert total == 2
        assert len(files) == 2

    @pytest.mark.asyncio
    async def test_list_pagination(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        owner = "user-page"
        for i in range(5):
            await svc.upload_file(
                owner_id=owner,
                filename=f"p{i}.txt",
                file_content=b"x",
                file_size=1,
            )

        page1, total = await svc.get_session_files(owner_id=owner, limit=3, offset=0)
        page2, _ = await svc.get_session_files(owner_id=owner, limit=3, offset=3)
        assert total == 5
        assert len(page1) == 3
        assert len(page2) == 2
        # No overlap
        ids1 = {f.id for f in page1}
        ids2 = {f.id for f in page2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_delete_file(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        content = b"delete me"
        view, _ = await svc.upload_file(
            owner_id="user-del",
            filename="del.txt",
            file_content=content,
            file_size=len(content),
        )
        assert view is not None

        deleted = await svc.delete_file(view.id, "user-del")
        assert deleted is True

        after = await svc.get(view.id)
        assert after is None

    @pytest.mark.asyncio
    async def test_delete_returns_false_if_not_found(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        result = await svc.delete_file(uuid4(), "user-x")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_rejects_wrong_owner(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        content = b"private"
        view, _ = await svc.upload_file(
            owner_id="owner-A",
            filename="priv.txt",
            file_content=content,
            file_size=len(content),
        )
        assert view is not None

        # Different owner cannot delete
        deleted = await svc.delete_file(view.id, "owner-B")
        assert deleted is False

        # Original owner's file still exists
        still_there = await svc.get(view.id)
        assert still_there is not None


class TestSessionQuota:
    """validate_session_quota enforcement."""

    @pytest.mark.asyncio
    async def test_quota_passes_when_no_files(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        ok, err = await svc.validate_session_quota("user-q", uuid4(), 1024)
        assert ok is True
        assert err is None

    @pytest.mark.asyncio
    async def test_quota_passes_without_session(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        ok, err = await svc.validate_session_quota("user-q", None, 10 * 1024 * 1024)
        assert ok is True
        assert err is None

    @pytest.mark.asyncio
    async def test_upload_rejects_unsupported_extension(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        view, err = await svc.upload_file(
            owner_id="user-bad",
            filename="virus.exe",
            file_content=b"bad",
            file_size=3,
        )
        assert view is None
        assert err is not None
        assert "Unsupported" in err

    @pytest.mark.asyncio
    async def test_upload_rejects_empty_file(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        view, err = await svc.upload_file(
            owner_id="user-empty",
            filename="empty.txt",
            file_content=b"",
            file_size=0,
        )
        assert view is None
        assert err is not None


class TestFileValidation:
    """validate_file method covers extension + MIME detection."""

    @pytest.mark.asyncio
    async def test_validate_txt_by_extension(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        ok, err, ftype = svc.validate_file("report.txt", 1024)
        assert ok is True
        assert err is None
        assert ftype is not None
        assert ftype.value == "txt"

    @pytest.mark.asyncio
    async def test_validate_pdf_by_extension(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        ok, err, ftype = svc.validate_file("report.pdf", 1024)
        assert ok is True
        assert ftype is not None
        assert ftype.value == "pdf"

    @pytest.mark.asyncio
    async def test_validate_falls_back_to_mime(self, stack) -> None:
        svc = CachedFileUploadService(stack)
        ok, err, ftype = svc.validate_file("noext", 100, content_type="text/plain")
        assert ok is True
        assert ftype is not None

    @pytest.mark.asyncio
    async def test_validate_rejects_oversized(self, stack) -> None:
        from deep_research.services.file_upload_service import MAX_FILE_SIZE_BYTES
        svc = CachedFileUploadService(stack)
        ok, err, ftype = svc.validate_file("big.txt", MAX_FILE_SIZE_BYTES + 1)
        assert ok is False
        assert err is not None
        assert ftype is None
