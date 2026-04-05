#!/usr/bin/env python3
"""CLI: Ingest data for a benchmark (clone, chunk, upload, index).

Usage:
    uv run benchmarks/ingest.py officeqa
    uv run benchmarks/ingest.py officeqa --catalog my_catalog --schema my_schema
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databricks_deep_research._fips_compat  # noqa: F401, E402  # FIPS md5 patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = BENCHMARKS_DIR.parent
    for name in (".env.officeqa", ".env", ".env.test"):
        candidate = root / name
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest benchmark data")
    parser.add_argument("benchmark", help="Benchmark name (e.g., officeqa)")
    parser.add_argument("--catalog", type=str, default=None)
    parser.add_argument("--schema", type=str, default=None)
    parser.add_argument("--vs-endpoint", type=str, default=None)
    parser.add_argument("--skip-index", action="store_true", help="Only write Delta table, skip VS index")
    parser.add_argument(
        "--force-recreate", action="store_true",
        help="Delete existing VS index and recreate from scratch (forces re-embedding)",
    )
    return parser.parse_args()


async def ingest_officeqa(args: argparse.Namespace) -> None:
    from benchmarks.core.config_loader import load_config
    from benchmarks.officeqa.ingest import run_ingestion

    config_path = BENCHMARKS_DIR / "officeqa" / "config.yaml"
    overrides: dict[str, str] = {}
    if args.catalog:
        overrides["catalog"] = args.catalog
    if args.schema:
        overrides["schema"] = args.schema
    if args.vs_endpoint:
        overrides["vs_endpoint_name"] = args.vs_endpoint

    config = load_config(config_path, cli_overrides=overrides)

    logger.info(
        "INGEST_START benchmark=officeqa catalog=%s schema=%s",
        config["catalog"],
        config["schema"],
    )

    index_fqn = await run_ingestion(config, force_recreate=args.force_recreate)
    print(f"\nIngestion complete!")
    print(f"Index: {index_fqn}")
    print(f"\nRun benchmark: uv run benchmarks/run.py officeqa")


async def main() -> None:
    args = parse_args()

    if args.benchmark == "officeqa":
        await ingest_officeqa(args)
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        print("Available: officeqa")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
