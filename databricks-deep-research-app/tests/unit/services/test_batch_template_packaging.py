"""Verify the spark-batch template is wired correctly at import time.

Companion to ``test_shell_app_template_packaging.py``. Same defect class
as the AIS shell-app failure — ``BatchTranslator`` would have crashed
identically on its first production deploy.
"""
from __future__ import annotations

from deep_research.services.deployment import batch


class TestBatchTemplateResolves:
    def test_template_dir_exists(self) -> None:
        assert batch._BATCH_TEMPLATE_DIR.is_dir(), (
            f"_BATCH_TEMPLATE_DIR does not resolve to a directory: "
            f"{batch._BATCH_TEMPLATE_DIR}"
        )

    def test_sql_template_is_file(self) -> None:
        assert batch._TEMPLATE_PATH.is_file(), (
            f"batch_inference.sql missing: {batch._TEMPLATE_PATH}"
        )

    def test_sql_template_is_non_empty(self) -> None:
        """A 0-byte template would resolve but render empty SQL — guard
        against bundling truncation."""
        assert batch._TEMPLATE_PATH.stat().st_size > 0
