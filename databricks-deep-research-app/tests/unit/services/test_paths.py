"""Unit tests for ``services/deployment/_paths.py``.

Covers the shared package-data resolver that powers both
``ShellAppExporter`` and ``BatchTranslator``. The helper must pick the
bundled location when present, fall back to the source-tree location
otherwise, and raise a typed ``PackageDataNotFound`` exception that
carries every path it tried.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.services.deployment._paths import (
    PackageDataNotFound,
    resolve_package_data_dir,
)


def _make_bundled_layout(tmp_path: Path, name: str) -> Path:
    """Create ``<tmp>/pkg/templates/<name>/`` and return the caller_file path
    that should resolve to it."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "templates" / name
    target.mkdir(parents=True)
    (target / "marker.txt").write_text("bundled\n")
    caller = pkg / "shell_app.py"
    caller.write_text("# stub\n")
    return caller


def _make_source_tree_layout(tmp_path: Path, name: str) -> Path:
    """Create a deep tree mimicking
    ``<app>/src/deep_research/services/deployment/shell_app.py`` so
    ``parents[4]`` resolves to ``<app>/`` and the source-tree fallback
    finds ``<app>/templates/<name>/``."""
    app = tmp_path / "app"
    services = app / "src" / "deep_research" / "services" / "deployment"
    services.mkdir(parents=True)
    templates = app / "templates" / name
    templates.mkdir(parents=True)
    (templates / "marker.txt").write_text("source-tree\n")
    caller = services / "shell_app.py"
    caller.write_text("# stub\n")
    return caller


class TestResolvePrefersBundled:
    def test_prefers_bundled_when_present(self, tmp_path: Path) -> None:
        caller = _make_bundled_layout(tmp_path, "agent-shell-app")
        resolved = resolve_package_data_dir(caller, "agent-shell-app")
        assert resolved == caller.parent / "templates" / "agent-shell-app"
        assert (resolved / "marker.txt").read_text() == "bundled\n"

    def test_prefers_bundled_over_source_tree(self, tmp_path: Path) -> None:
        """Both layouts present — bundled must win."""
        # Build source-tree layout first, then add the bundled marker.
        caller = _make_source_tree_layout(tmp_path, "agent-shell-app")
        bundled = caller.parent / "templates" / "agent-shell-app"
        bundled.mkdir(parents=True)
        (bundled / "marker.txt").write_text("bundled-wins\n")

        resolved = resolve_package_data_dir(caller, "agent-shell-app")
        assert (resolved / "marker.txt").read_text() == "bundled-wins\n"


class TestResolveFallsBackToSourceTree:
    def test_falls_back_when_only_source_tree_exists(self, tmp_path: Path) -> None:
        caller = _make_source_tree_layout(tmp_path, "agent-shell-app")
        # No bundled location next to caller.
        bundled = caller.parent / "templates"
        assert not bundled.exists()

        resolved = resolve_package_data_dir(caller, "agent-shell-app")
        assert (resolved / "marker.txt").read_text() == "source-tree\n"

    def test_falls_back_when_bundled_dir_missing(self, tmp_path: Path) -> None:
        caller = _make_source_tree_layout(tmp_path, "spark-batch")
        resolved = resolve_package_data_dir(caller, "spark-batch")
        assert resolved.name == "spark-batch"
        assert resolved.is_dir()


class TestResolveRaisesOnMiss:
    def test_raises_when_neither_layout_has_the_dir(self, tmp_path: Path) -> None:
        # Source-tree layout exists but the requested template does NOT.
        caller = _make_source_tree_layout(tmp_path, "agent-shell-app")
        with pytest.raises(PackageDataNotFound) as exc_info:
            resolve_package_data_dir(caller, "does-not-exist")
        err = exc_info.value
        assert err.name == "templates/does-not-exist"
        # The exception should list every path tried, so an operator can
        # see exactly where we looked.
        assert len(err.searched) >= 1
        for searched_path in err.searched:
            assert "does-not-exist" in str(searched_path)

    def test_raises_when_caller_file_is_shallow(self) -> None:
        """If ``__file__`` lives shallower than parents[4], the fallback is
        skipped to avoid an IndexError. Only the bundled path is tried.
        Helper still raises a typed exception.

        We use a synthetic path under `/tmp` rather than ``tmp_path`` because
        macOS test runners place ``tmp_path`` deep enough that ``parents[4]``
        still resolves to a real directory — defeating the "shallow" case.
        """
        caller = Path("/no_parents.py")
        # caller has only one parent (/), so the parents-guard skips the fallback.
        with pytest.raises(PackageDataNotFound) as exc_info:
            resolve_package_data_dir(caller, "agent-shell-app")
        # Only the bundled path was tried (fallback skipped due to parents guard).
        assert len(exc_info.value.searched) == 1


class TestResolveHandlesSymlinks:
    def test_resolves_symlinks_in_caller_file(self, tmp_path: Path) -> None:
        """Symlinking the caller path must not break resolution — the helper
        resolves symlinks upfront so both layouts work transparently."""
        caller = _make_bundled_layout(tmp_path, "agent-shell-app")
        link_dir = tmp_path / "link"
        link_dir.symlink_to(caller.parent)
        linked_caller = link_dir / "shell_app.py"

        resolved = resolve_package_data_dir(linked_caller, "agent-shell-app")
        # The resolved path comes from the real (target) directory, not the
        # symlink itself.
        assert resolved.is_dir()
        assert (resolved / "marker.txt").exists()
