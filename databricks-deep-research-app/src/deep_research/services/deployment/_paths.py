"""Resolve package-bundled data directories that ship with the wheel.

Templates and other static assets are co-located in the source tree at
``<app>/templates/<name>/`` for dev ergonomics, but hatch ``force-include``
rules copy them under the package at build time. This helper resolves to
the right place in both layouts without per-caller fallback logic.
"""
from __future__ import annotations

from pathlib import Path


class PackageDataNotFound(RuntimeError):
    """Raised when neither bundled nor source-tree path holds the asset.

    Carries the searched paths so the operator can see exactly where we
    looked. Subclass of RuntimeError so the app's startup smoke check can
    re-raise it as a fatal config error.
    """

    def __init__(self, name: str, searched: list[Path]) -> None:
        self.name = name
        self.searched = list(searched)
        super().__init__(
            f"Package data {name!r} not found. Searched: "
            + ", ".join(str(p) for p in searched)
        )


def resolve_package_data_dir(caller_file: Path, name: str) -> Path:
    """Resolve a package-bundled data directory.

    Looks first next to ``caller_file`` (wheel/installed layout — files
    are co-located by hatch ``force-include``), then walks up to the app
    root for the editable / source-tree layout.

    Args:
        caller_file: Pass ``Path(__file__)`` from the caller. Resolved
            internally so symlinks don't matter.
        name: Subdirectory name under ``templates/``, e.g.
            ``"agent-shell-app"`` or ``"spark-batch"``. The leading
            ``templates/`` segment is implicit so call sites cannot drift.

    Returns:
        The resolved directory ``Path``. Caller may read files inside it
        with ordinary ``Path`` operations.

    Raises:
        PackageDataNotFound: When neither location has the directory.
            The exception lists every path searched.
    """
    here = Path(caller_file).resolve()
    searched: list[Path] = []

    bundled = here.parent / "templates" / name
    searched.append(bundled)
    if bundled.is_dir():
        return bundled

    if len(here.parents) > 4:
        source_tree = here.parents[4] / "templates" / name
        searched.append(source_tree)
        if source_tree.is_dir():
            return source_tree

    raise PackageDataNotFound(name=f"templates/{name}", searched=searched)
