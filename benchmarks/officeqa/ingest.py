"""OfficeQA ingestion pipeline: clone → chunk → upload → create VS index."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.officeqa.chunker import Chunk, ChunkConfig, chunk_file

logger = logging.getLogger(__name__)


async def _execute_sql(ws_client: Any, sql: str) -> Any:
    """Execute SQL via Statement Execution API with polling."""
    warehouse_id = _get_warehouse_id(ws_client)
    resp = ws_client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
    )
    # Poll if still pending/running (for long operations like COPY INTO)
    state = resp.status.state.value if resp.status and resp.status.state else None
    while state in ("PENDING", "RUNNING"):
        await asyncio.sleep(5)
        resp = ws_client.statement_execution.get_statement(resp.statement_id)
        state = resp.status.state.value if resp.status and resp.status.state else None
    if state == "FAILED":
        error = resp.status.error if resp.status.error else "Unknown SQL error"
        raise RuntimeError(f"SQL execution failed: {error}\nSQL: {sql[:200]}")
    return resp


def _get_warehouse_id(ws_client: Any) -> str:
    """Get a SQL warehouse ID — prefer serverless, fall back to any available."""
    import os

    wh_id = os.environ.get("OFFICEQA_WAREHOUSE_ID", "")
    if wh_id:
        return wh_id

    warehouses = ws_client.warehouses.list()
    for wh in warehouses:
        if wh.warehouse_type and "SERVERLESS" in str(wh.warehouse_type).upper():
            if wh.state and str(wh.state).upper() in ("RUNNING", "STARTING"):
                return wh.id
    # Fall back to any running warehouse
    for wh in ws_client.warehouses.list():
        if wh.state and str(wh.state).upper() in ("RUNNING", "STARTING"):
            return wh.id

    raise RuntimeError(
        "No running SQL warehouse found. Set OFFICEQA_WAREHOUSE_ID or start a warehouse."
    )


def clone_repo(url: str, local_path: Path, branch: str = "main") -> Path:
    """Clone the OfficeQA repository if not already present."""
    if local_path.exists() and (local_path / ".git").exists():
        logger.info("REPO_EXISTS path=%s", local_path)
        # Pull latest
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=local_path,
            capture_output=True,
            check=False,
        )
        return local_path

    # Auto-discover repo cloned next to the benchmark package
    # e.g. benchmarks/data/officeqa/ (sibling of benchmarks/officeqa/)
    benchmarks_dir = Path(__file__).resolve().parent.parent
    sibling_local = benchmarks_dir / local_path.name
    if sibling_local != local_path and sibling_local.exists() and (sibling_local / ".git").exists():
        logger.info("REPO_EXISTS path=%s (auto-discovered)", sibling_local)
        return sibling_local

    logger.info("REPO_CLONE url=%s path=%s", url, local_path)
    for attempt in range(3):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, url, str(local_path)],
                capture_output=True,
                check=True,
                timeout=120,
            )
            return local_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stderr = getattr(exc, "stderr", b"") or b""
            logger.warning(
                "REPO_CLONE_RETRY attempt=%d error=%s stderr=%s",
                attempt + 1, exc, stderr.decode(errors="replace").strip(),
            )
            if attempt == 2:
                raise RuntimeError(
                    f"Failed to clone {url} after 3 attempts: {stderr.decode(errors='replace').strip()}"
                ) from exc
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
                logger.info("REPO_CLONE_CLEANUP path=%s", local_path)
            asyncio.get_event_loop()  # just to verify we're in async context
            import time

            time.sleep(5 * (attempt + 1))

    raise RuntimeError("Unreachable")


def discover_text_files(repo_path: Path) -> list[Path]:
    """Discover transformed text files from the OfficeQA repo.

    Handles both raw directory and zip archives.

    .. note:: Prefer ``transform_json_files()`` which re-generates .txt files
       from JSON sources with our own HTML parser, avoiding ``pd.read_html()``
       corruption (NaN, comma stripping, float coercion).
    """
    transformed_dir = repo_path / "treasury_bulletins_parsed" / "transformed"

    if transformed_dir.exists():
        files = sorted(transformed_dir.glob("*.txt"))
        if files:
            logger.info("DISCOVER_FILES dir=%s count=%d", transformed_dir, len(files))
            return files

    # Try zip fallback
    for zip_path in repo_path.glob("treasury_bulletins_parsed/transformed/*.zip"):
        logger.info("DISCOVER_ZIP path=%s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(transformed_dir)
        files = sorted(transformed_dir.glob("*.txt"))
        if files:
            logger.info("DISCOVER_FILES_FROM_ZIP count=%d", len(files))
            return files

    raise FileNotFoundError(
        f"No text files found in {transformed_dir}. "
        "Ensure OfficeQA repo has the transformed/ directory."
    )


def _discover_json_files(repo_path: Path) -> list[Path]:
    """Discover JSON source files from the OfficeQA repo, extracting zips if needed."""
    jsons_dir = repo_path / "treasury_bulletins_parsed" / "jsons"

    # Extract zips if JSON directory is empty
    json_files = sorted(jsons_dir.glob("treasury_bulletin_*.json"))
    if not json_files:
        for zip_path in sorted(jsons_dir.glob("*.zip")):
            logger.info("JSON_EXTRACT_ZIP path=%s", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(jsons_dir)
        json_files = sorted(jsons_dir.glob("treasury_bulletin_*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {jsons_dir}. "
            "Ensure OfficeQA repo is cloned with treasury_bulletins_parsed/jsons/."
        )

    logger.info("DISCOVER_JSONS dir=%s count=%d", jsons_dir, len(json_files))
    return json_files


def transform_json_files(repo_path: Path, output_dir: Path) -> list[Path]:
    """Transform upstream JSON files to clean Markdown .txt files.

    Reads JSON elements from the OfficeQA repo, parses HTML tables with our
    own parser (preserving exact cell text — no NaN, no comma stripping, no
    float coercion), and writes .txt files to *output_dir*.

    Parameters
    ----------
    repo_path:
        Root of the OfficeQA repo clone.
    output_dir:
        Directory to write output .txt files.  Created if it doesn't exist.

    Returns
    -------
    list[Path]:
        Sorted list of output .txt file paths.
    """
    import json as _json
    import re as _re

    from benchmarks.officeqa.html_table_parser import parse_html_tables

    json_files = _discover_json_files(repo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as exc:
            logger.warning("TRANSFORM_SKIP file=%s error=%s", json_path.name, exc)
            continue

        doc = data.get("document") or {}
        elements = doc.get("elements")
        if not isinstance(elements, list):
            continue

        out_lines: list[str] = []
        for el in elements:
            content = el.get("content") if isinstance(el, dict) else None
            if not isinstance(content, str) or not content.strip():
                continue

            if "<table" in content.lower():
                md_tables = parse_html_tables(content)
                for table_md in md_tables:
                    out_lines.append(table_md)
                    out_lines.append("")
            else:
                text = content.strip()
                text = _re.sub(r"\r\n?", "\n", text)
                for line in text.split("\n"):
                    out_lines.append(line)
                out_lines.append("")

        # Write output with same filename but .txt extension
        out_name = json_path.stem + ".txt"
        out_path = output_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            for line in out_lines:
                f.write(line)
                f.write("\n")

        output_paths.append(out_path)
        logger.info("TRANSFORM_DONE file=%s lines=%d", out_name, len(out_lines))

    logger.info("TRANSFORM_ALL files=%d output_dir=%s", len(output_paths), output_dir)
    return sorted(output_paths)


def chunk_all_files(
    files: list[Path], config: ChunkConfig | None = None
) -> list[Chunk]:
    """Chunk all text files."""
    if config is None:
        config = ChunkConfig()

    all_chunks: list[Chunk] = []
    for fp in files:
        chunks = chunk_file(fp, config)
        all_chunks.extend(chunks)

    logger.info(
        "CHUNK_ALL files=%d total_chunks=%d total_chars=%d",
        len(files),
        len(all_chunks),
        sum(c.char_count for c in all_chunks),
    )
    return all_chunks


async def write_delta_table(
    ws_client: Any,
    chunks: list[Chunk],
    catalog: str,
    schema: str,
    table_name: str,
) -> str:
    """Write chunks to a Delta table via Parquet upload + COPY INTO."""
    table_fqn = f"{catalog}.{schema}.{table_name}"
    volume_fqn = f"{catalog}.{schema}.benchmark_staging"

    logger.info("DELTA_WRITE table=%s chunks=%d", table_fqn, len(chunks))

    # 1. Create schema + volume
    await _execute_sql(ws_client, f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    await _execute_sql(ws_client, f"CREATE VOLUME IF NOT EXISTS {volume_fqn}")

    # 2. Write Parquet locally (cast char_count to int32 to match Delta INT)
    df = pd.DataFrame([asdict(c) for c in chunks])
    df["char_count"] = df["char_count"].astype("int32")
    local_path = Path(tempfile.mktemp(suffix=".parquet"))
    df.to_parquet(local_path)
    logger.info("PARQUET_WRITTEN path=%s size_mb=%.1f", local_path, local_path.stat().st_size / 1e6)

    # 3. Upload to Volume
    volume_path = f"/Volumes/{catalog}/{schema}/benchmark_staging/chunks.parquet"
    with open(local_path, "rb") as f:
        ws_client.files.upload(volume_path, f, overwrite=True)
    logger.info("VOLUME_UPLOAD path=%s", volume_path)
    local_path.unlink(missing_ok=True)

    # 4. Create table + COPY INTO (CDF required for VS delta-sync)
    await _execute_sql(
        ws_client,
        f"""
        CREATE TABLE IF NOT EXISTS {table_fqn} (
            chunk_id STRING,
            file_name STRING,
            bulletin_date STRING,
            page_info STRING,
            content STRING,
            chunk_type STRING,
            char_count INT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """,
    )

    # Enable CDF on existing tables (no-op if already enabled)
    await _execute_sql(
        ws_client,
        f"ALTER TABLE {table_fqn} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
    )

    # Truncate first (idempotent re-ingestion)
    await _execute_sql(ws_client, f"TRUNCATE TABLE {table_fqn}")

    await _execute_sql(
        ws_client,
        f"""
        COPY INTO {table_fqn}
        FROM '{volume_path}'
        FILEFORMAT = PARQUET
        COPY_OPTIONS ('force' = 'true')
        """,
    )

    # Verify
    resp = await _execute_sql(ws_client, f"SELECT COUNT(*) FROM {table_fqn}")
    if resp.result and resp.result.data_array:
        count = resp.result.data_array[0][0]
        logger.info("DELTA_VERIFY table=%s row_count=%s", table_fqn, count)

    return table_fqn


async def create_vs_index(
    ws_client: Any,
    catalog: str,
    schema: str,
    table_name: str,
    index_name: str,
    vs_endpoint_name: str,
    embedding_model: str,
    force_recreate: bool = False,
) -> str:
    """Create a Vector Search index with managed embeddings, poll until ONLINE."""
    index_fqn = f"{catalog}.{schema}.{index_name}"
    table_fqn = f"{catalog}.{schema}.{table_name}"

    # Check if already exists and ready
    try:
        existing = ws_client.vector_search_indexes.get_index(index_fqn)
        if force_recreate:
            logger.info("INDEX_FORCE_DELETE index=%s", index_fqn)
            ws_client.vector_search_indexes.delete_index(index_fqn)
            # Brief pause for cleanup
            await asyncio.sleep(10)
        elif existing.status and existing.status.ready:
            logger.info("INDEX_EXISTS index=%s status=ONLINE", index_fqn)
            return index_fqn
        else:
            logger.info("INDEX_EXISTS_NOT_READY index=%s status=%s", index_fqn, existing.status)
    except Exception:
        # Doesn't exist — create
        pass

    # Create index
    logger.info(
        "INDEX_CREATE index=%s endpoint=%s embedding=%s",
        index_fqn,
        vs_endpoint_name,
        embedding_model,
    )
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingSourceColumn,
        PipelineType,
        VectorIndexType,
    )

    ws_client.vector_search_indexes.create_index(
        name=index_fqn,
        endpoint_name=vs_endpoint_name,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=table_fqn,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name=embedding_model,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
        ),
    )

    # Poll for ONLINE status (timeout: 90 min for large indexes)
    for attempt in range(180):
        await asyncio.sleep(30)
        idx = ws_client.vector_search_indexes.get_index(index_fqn)
        if idx.status and idx.status.ready:
            logger.info("INDEX_READY index=%s attempts=%d", index_fqn, attempt + 1)
            return index_fqn
        logger.info(
            "INDEX_POLL index=%s status=%s attempt=%d/180",
            index_fqn,
            idx.status,
            attempt + 1,
        )

    raise TimeoutError(f"VS index {index_fqn} not ONLINE after 90 minutes")


async def validate_index(
    ws_client: Any, index_fqn: str, test_queries: list[str] | None = None
) -> None:
    """Validate the VS index with test queries."""
    if test_queries is None:
        test_queries = [
            "total federal receipts fiscal year 2023",
            "national debt outstanding 1945",
            "treasury bulletin monthly statement",
        ]

    for query in test_queries:
        try:
            resp = ws_client.vector_search_indexes.query_index(
                index_name=index_fqn,
                columns=["chunk_id", "content", "bulletin_date"],
                query_text=query,
                num_results=3,
            )
            result_count = (
                len(resp.result.data_array) if resp.result and resp.result.data_array else 0
            )
            logger.info(
                "INDEX_VALIDATE query=%s results=%d", query[:50], result_count
            )
        except Exception as exc:
            logger.warning("INDEX_VALIDATE_FAIL query=%s error=%s", query[:50], exc)


# ---------------------------------------------------------------------------
# Structured table records (v9+ — optional, gated on config)
# ---------------------------------------------------------------------------


def extract_parsed_tables(repo_path: Path) -> dict[str, list[Any]]:
    """Extract structured ``ParsedTable`` objects from JSON source files.

    Iterates the same JSON elements as ``transform_json_files`` but calls
    ``parse_html_tables_structured`` to capture both Markdown and structured
    JSON.  Keyed by file stem (e.g., ``"treasury_bulletin_1941_01"``).
    """
    import json as _json

    from benchmarks.officeqa.html_table_parser import parse_html_tables_structured

    json_files = _discover_json_files(repo_path)
    result: dict[str, list[Any]] = {}

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            continue

        doc = data.get("document") or {}
        elements = doc.get("elements")
        if not isinstance(elements, list):
            continue

        file_tables: list[Any] = []
        for el in elements:
            content = el.get("content") if isinstance(el, dict) else None
            if not isinstance(content, str) or not content.strip():
                continue
            if "<table" in content.lower():
                parsed = parse_html_tables_structured(content)
                file_tables.extend(parsed)

        if file_tables:
            result[json_path.stem] = file_tables

    logger.info("EXTRACT_PARSED_TABLES files=%d", len(result))
    return result


async def write_tables_delta_table(
    ws_client: Any,
    table_records: list[Any],
    catalog: str,
    schema: str,
    table_name: str,
) -> str:
    """Write table records to a ``treasury_tables`` Delta table."""
    from dataclasses import asdict

    table_fqn = f"{catalog}.{schema}.{table_name}"
    volume_fqn = f"{catalog}.{schema}.benchmark_staging"

    logger.info("TABLES_DELTA_WRITE table=%s records=%d", table_fqn, len(table_records))

    df = pd.DataFrame([asdict(r) for r in table_records])
    df["row_count"] = df["row_count"].astype("int32")
    df["col_count"] = df["col_count"].astype("int32")
    df["char_count"] = df["char_count"].astype("int32")

    local_path = Path(tempfile.mktemp(suffix=".parquet"))
    df.to_parquet(local_path)
    logger.info(
        "TABLES_PARQUET_WRITTEN path=%s size_mb=%.1f",
        local_path, local_path.stat().st_size / 1e6,
    )

    volume_path = f"/Volumes/{catalog}/{schema}/benchmark_staging/table_records.parquet"
    with open(local_path, "rb") as f:
        ws_client.files.upload(volume_path, f, overwrite=True)
    local_path.unlink(missing_ok=True)

    await _execute_sql(
        ws_client,
        f"""
        CREATE TABLE IF NOT EXISTS {table_fqn} (
            chunk_id STRING,
            file_name STRING,
            bulletin_date STRING,
            page_info STRING,
            table_title STRING,
            annotation STRING,
            content STRING,
            table_json STRING,
            row_count INT,
            col_count INT,
            chunk_type STRING,
            char_count INT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """,
    )
    await _execute_sql(
        ws_client,
        f"ALTER TABLE {table_fqn} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
    )
    await _execute_sql(ws_client, f"TRUNCATE TABLE {table_fqn}")
    await _execute_sql(
        ws_client,
        f"""
        COPY INTO {table_fqn}
        FROM '{volume_path}'
        FILEFORMAT = PARQUET
        COPY_OPTIONS ('force' = 'true')
        """,
    )

    resp = await _execute_sql(ws_client, f"SELECT COUNT(*) FROM {table_fqn}")
    if resp.result and resp.result.data_array:
        count = resp.result.data_array[0][0]
        logger.info("TABLES_DELTA_VERIFY table=%s row_count=%s", table_fqn, count)

    return table_fqn


async def validate_table_records(
    ws_client: Any, catalog: str, schema: str,
    chunks_table: str, tables_table: str,
) -> None:
    """Validate that all treasury_tables chunk_ids exist in treasury_chunks."""
    fqn_chunks = f"{catalog}.{schema}.{chunks_table}"
    fqn_tables = f"{catalog}.{schema}.{tables_table}"
    try:
        resp = await _execute_sql(
            ws_client,
            f"""
            SELECT COUNT(*) FROM {fqn_tables} t
            LEFT JOIN {fqn_chunks} c ON t.chunk_id = c.chunk_id
            WHERE c.chunk_id IS NULL
            """,
        )
        if resp.result and resp.result.data_array:
            orphans = int(resp.result.data_array[0][0])
            if orphans > 0:
                logger.warning(
                    "TABLE_RECORDS_ORPHANS count=%d — chunk_ids in %s not found in %s",
                    orphans, fqn_tables, fqn_chunks,
                )
            else:
                logger.info("TABLE_RECORDS_VALIDATED all_chunk_ids_found=True")
    except Exception as exc:
        logger.warning("TABLE_RECORDS_VALIDATION_FAILED error=%s", exc)


async def run_ingestion(config: dict[str, Any], force_recreate: bool = False) -> str:
    """Full ingestion pipeline: clone → chunk → upload → index.

    Parameters
    ----------
    config:
        Loaded config dict (from config.yaml).
    force_recreate:
        If True, delete existing VS index and recreate from scratch.

    Returns
    -------
    str:
        Fully qualified index name.
    """
    from databricks.sdk import WorkspaceClient

    ws_client = WorkspaceClient()

    # 1. Clone repo
    repo_path = clone_repo(
        url=config["repo"]["url"],
        local_path=Path(config["repo"]["local_path"]),
        branch=config["repo"].get("branch", "main"),
    )

    # 2. Transform JSON → clean .txt  +  3. Chunk all files
    #    Use a temp dir — files are only needed until chunking finishes.
    chunking_cfg = config.get("chunking", {})
    chunk_config = ChunkConfig(
        chunk_max_chars=chunking_cfg.get("chunk_max_chars", 2000),
        chunk_overlap_chars=chunking_cfg.get("chunk_overlap_chars", 200),
        table_max_chars=chunking_cfg.get("table_max_chars", 4000),
        section_max_chars=chunking_cfg.get("section_max_chars", 8000),
    )
    with tempfile.TemporaryDirectory(prefix="officeqa_transformed_") as tmp:
        files = transform_json_files(repo_path, Path(tmp))
        all_chunks = chunk_all_files(files, chunk_config)

    # 4. Write Delta table
    catalog = config["catalog"]
    schema = config["schema"]
    table_fqn = await write_delta_table(
        ws_client, all_chunks, catalog, schema, config["delta_table"]
    )
    logger.info("DELTA_TABLE_READY table=%s", table_fqn)

    # 5. Create VS index
    index_fqn = await create_vs_index(
        ws_client,
        catalog=catalog,
        schema=schema,
        table_name=config["delta_table"],
        index_name=config["vector_index"],
        vs_endpoint_name=config["vs_endpoint_name"],
        embedding_model=config["embedding_model"],
        force_recreate=force_recreate,
    )

    # 6. Validate
    await validate_index(ws_client, index_fqn)

    # 7. Optional: Build treasury_tables for v9+ workflows
    tables_table_name = config.get("tables_delta_table")
    if tables_table_name:
        from benchmarks.officeqa.chunker import build_table_records

        logger.info("TABLES_PIPELINE_START building structured table records")
        parsed_tables_by_file = extract_parsed_tables(repo_path)
        table_records = build_table_records(all_chunks, parsed_tables_by_file)
        if table_records:
            tables_fqn = await write_tables_delta_table(
                ws_client, table_records, catalog, schema, tables_table_name,
            )
            await validate_table_records(
                ws_client, catalog, schema,
                config["delta_table"], tables_table_name,
            )
            logger.info("TABLES_PIPELINE_COMPLETE table=%s records=%d",
                        tables_fqn, len(table_records))
        else:
            logger.warning("TABLES_PIPELINE_EMPTY no table records built")

    logger.info("INGESTION_COMPLETE index=%s", index_fqn)
    return index_fqn
