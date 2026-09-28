"""Smoke check: assert the most-recent dist/*.whl contains every package
data file the deployment translators expect at runtime.

This catches packaging regressions where a template gets added to the source
tree but not to the hatch ``force-include`` rule — the bug that took down
the AIS deployment on 2026-05-12. The expected file list is driven from
the source-of-truth tuples in ``shell_app.py`` so adding a new template
file requires updating one place (the tuple), not two.

Exits 0 on success, 1 on any missing entry. Designed to run from
``make verify-wheel`` after ``uv build --wheel``.
"""
from __future__ import annotations

import glob
import importlib.util
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHELL_APP_PATH = REPO_ROOT / "src/deep_research/services/deployment/shell_app.py"
BATCH_PATH = REPO_ROOT / "src/deep_research/services/deployment/batch.py"


def _load_module(name: str, file_path: Path) -> object:
    """Import a module by file path without running package __init__ chains.

    We need to read the constants from the source files but cannot do a
    normal `from deep_research...` import because that triggers the very
    `_TEMPLATE_DIR` resolution we're trying to verify — which fails if the
    wheel isn't installed. So load the source as a free-standing module
    with a stub helper that returns a sentinel path.
    """
    # Inject a stub for the _paths helper so source files import cleanly.
    import types

    stub = types.ModuleType("deep_research.services.deployment._paths")

    def _stub_resolve(_caller: Path, name: str) -> Path:
        return Path("/__stub__") / "templates" / name

    stub.resolve_package_data_dir = _stub_resolve  # type: ignore[attr-defined]
    sys.modules["deep_research.services.deployment._paths"] = stub

    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    wheels = sorted(
        glob.glob(str(REPO_ROOT / "dist" / "*.whl")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not wheels:
        print("ERROR: no wheel found in dist/. Run `uv build --wheel` first.", file=sys.stderr)
        return 1

    wheel_path = wheels[0]
    with zipfile.ZipFile(wheel_path) as z:
        names = set(z.namelist())

    # Read the source-of-truth constants from shell_app.py.
    shell_mod = _load_module("_shell_app_src", SHELL_APP_PATH)
    verbatim = shell_mod._VERBATIM_FILES  # type: ignore[attr-defined]
    jinja = shell_mod._JINJA_FILES  # type: ignore[attr-defined]
    expected_shell = {
        f"deep_research/services/deployment/templates/agent-shell-app/{src}"
        for src, _ in tuple(verbatim) + tuple(jinja)
    }

    # batch.py exposes a single SQL file path.
    expected_batch = {
        "deep_research/services/deployment/templates/spark-batch/batch_inference.sql",
    }

    expected = expected_shell | expected_batch
    missing = sorted(expected - names)
    if missing:
        print(f"WHEEL {wheel_path} MISSING:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    # Framework wheel must be bundled exactly once under
    # deep_research/services/deployment/_framework_wheel/. shell_app.py reads
    # it from there at deploy-time to embed in generated shell-app zips, so a
    # missing or duplicate file breaks every shell-app deploy.
    framework_wheel_prefix = (
        "deep_research/services/deployment/_framework_wheel/databricks_deep_research-"
    )
    framework_wheels = [
        n for n in names if n.startswith(framework_wheel_prefix) and n.endswith(".whl")
    ]
    if len(framework_wheels) != 1:
        print(
            f"WHEEL {wheel_path} FRAMEWORK WHEEL MISCOUNT: expected exactly 1 "
            f"databricks_deep_research-*.whl under _framework_wheel/, found "
            f"{len(framework_wheels)}: {framework_wheels}",
            file=sys.stderr,
        )
        print(
            "Run `make build-framework` in databricks-deep-research-app/ to stage it.",
            file=sys.stderr,
        )
        return 1

    print(
        f"WHEEL {wheel_path} OK ({len(expected)} template files + "
        f"framework wheel {framework_wheels[0].rsplit('/', 1)[-1]} present)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
