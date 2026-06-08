"""Verify the shell-app templates are wired correctly at import time.

These tests would have failed against the AIS wheel before T1 fixed the
``parents[4]`` packaging bug. They are layout-aware: ``_TEMPLATE_DIR`` is
computed at module import via the shared resolver, so just importing
``shell_app`` proves the helper found something. The per-file assertions
catch the "directory exists but is empty" failure mode.
"""
from __future__ import annotations

from deep_research.services.deployment import shell_app


class TestTemplateDirResolves:
    def test_template_dir_exists(self) -> None:
        assert shell_app._TEMPLATE_DIR.is_dir(), (
            f"_TEMPLATE_DIR does not resolve to a directory: "
            f"{shell_app._TEMPLATE_DIR}"
        )


class TestAllKnownTemplateFilesPresent:
    def test_verbatim_files_present(self) -> None:
        """Every source name in _VERBATIM_FILES must exist as a file."""
        for src, _dst in shell_app._VERBATIM_FILES:
            path = shell_app._TEMPLATE_DIR / src
            assert path.is_file(), f"verbatim template missing: {path}"

    def test_jinja_files_present(self) -> None:
        """Every source name in _JINJA_FILES must exist as a file."""
        for src, _dst in shell_app._JINJA_FILES:
            path = shell_app._TEMPLATE_DIR / src
            assert path.is_file(), f"jinja template missing: {path}"

    def test_no_orphan_files_referenced(self) -> None:
        """All file names in _VERBATIM_FILES + _JINJA_FILES are reachable —
        sanity check on the tuples themselves."""
        all_srcs = [s for s, _ in shell_app._VERBATIM_FILES + shell_app._JINJA_FILES]
        # Each source name should be a relative path with no '..' escapes.
        for s in all_srcs:
            assert ".." not in s, f"template source contains '..': {s!r}"
            assert not s.startswith("/"), f"template source is absolute: {s!r}"


class TestExecutableEntries:
    def test_entrypoint_sh_is_in_exec_set(self) -> None:
        """entrypoint.sh must be tagged for +x mode bits in the generated zip."""
        assert "entrypoint.sh" in shell_app._EXEC_ENTRIES

    def test_zip_mode_bits_executable_for_entrypoint(self) -> None:
        bits = shell_app._zip_mode_bits("entrypoint.sh")
        # high 16 bits encode the unix file mode; expect 0o755.
        assert bits >> 16 == 0o755

    def test_zip_mode_bits_non_executable_for_regular(self) -> None:
        bits = shell_app._zip_mode_bits("app.py")
        assert bits >> 16 == 0o644
