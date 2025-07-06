"""Data loading module for ccwatch using DuckDB.

This module handles all data loading operations from JSONL files using DuckDB,
providing high-performance query processing with zero-copy reads.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from config import AppConfig
from utils.logging_config import get_logger, log_with_context

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()


def extract_tool_uses_from_assistant_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract tool usage information from assistant messages DataFrame.

    Args:
        df: DataFrame containing assistant messages with 'message' column

    Returns:
        DataFrame with tool usage information
    """
    tool_uses = []

    # Log tool extraction start (DEBUG level for production)
    log_with_context(logger, "DEBUG", "Starting tool extraction", total_rows=len(df))

    for idx, row in df.iterrows():
        # Debug logging for first row only in development
        if idx == 0 and logger.isEnabledFor(logging.DEBUG):
            log_with_context(
                logger,
                "DEBUG",
                "First row message type",
                msg_type=type(row.get("message")).__name__,
                has_message="message" in row,
            )

        if row.get("message") and isinstance(row["message"], dict):
            content = row["message"].get("content", [])

            # If content is a string, parse it as JSON
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    continue

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_uses.append(
                            {
                                "source_file": row["source_file"],
                                "timestamp": row["timestamp"],
                                "session_id": row["session_id"],
                                "uuid": row["uuid"],
                                "cwd": row.get("cwd", ""),
                                "project_path": row["project_path"],
                                "tool_name": item.get("name", "Unknown"),
                                "tool_id": item.get("id", ""),
                            }
                        )

    # Log tool extraction completion
    log_with_context(logger, "DEBUG", "Tool extraction completed", tools_found=len(tool_uses))

    if tool_uses:
        tool_df = pd.DataFrame(tool_uses)
        tool_df["timestamp"] = pd.to_datetime(tool_df["timestamp"])
        return tool_df
    else:
        return pd.DataFrame()


def load_all_logs_with_duckdb(cache_key: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all JSONL files once and extract different data types.

    Args:
        cache_key: Cache control key (update counter)

    Returns:
        Tuple of (assistant_df, system_df, tool_df)
    """
    _ = cache_key  # Used for cache control

    start_time = datetime.now()

    conn = duckdb.connect(":memory:")

    # Load ICU extension for timezone support
    conn.execute("LOAD icu")

    # Use glob pattern to read all JSONL files at once
    glob_pattern = str(config.claude_projects_path / config.jsonl_pattern)

    # Create a temporary table with all logs
    log_with_context(logger, "INFO", "Loading all JSONL files into DuckDB", glob_pattern=glob_pattern)

    try:
        # Load all data into a temporary table
        conn.execute(
            """
            CREATE TABLE all_logs AS 
            SELECT 
                filename,
                timestamp,
                type,
                sessionId,
                uuid,
                parentUuid,
                cwd,
                userType,
                message,
                isApiErrorMessage,
                toolUseResult,
                leafUuid,
                requestId,
                summary,
                isMeta,
                isSidechain,
                isCompactSummary
            FROM read_json_auto(?, format='newline_delimited', filename=true)
        """,
            [glob_pattern],
        )

        # Get total row count
        total_rows = conn.execute("SELECT COUNT(*) FROM all_logs").fetchone()[0]
        log_with_context(logger, "INFO", "Total logs loaded", total_rows=total_rows)

        # Extract assistant messages
        assistant_query = """
        SELECT 
            filename as source_file,
            timezone(current_setting('TimeZone'), timestamp::TIMESTAMP) as timestamp,
            type as log_type,
            TRY_CAST(message.role AS VARCHAR) as role,
            TRY_CAST(message.content AS VARCHAR) as message_content,
            TRY_CAST(json_extract_string(to_json(message), '$.model') AS VARCHAR) as model,
            sessionId as session_id,
            uuid,
            parentUuid as parent_uuid,
            cwd,
            userType as user_type,
            TRY_CAST(message.usage.input_tokens AS BIGINT) as input_tokens,
            TRY_CAST(message.usage.cache_creation_input_tokens AS BIGINT) as cache_creation_input_tokens,
            TRY_CAST(message.usage.cache_read_input_tokens AS BIGINT) as cache_read_input_tokens,
            TRY_CAST(message.usage.output_tokens AS BIGINT) as output_tokens,
            message  -- Include the full message object for tool extraction
        FROM all_logs
        WHERE type = 'assistant'
        """

        assistant_df = conn.execute(assistant_query).df()

        if not assistant_df.empty:
            # Process assistant data
            assistant_df["timestamp"] = pd.to_datetime(assistant_df["timestamp"])
            assistant_df["project_path"] = assistant_df["source_file"].apply(lambda x: Path(x).parent.name)
            assistant_df["session_id"] = assistant_df["session_id"].astype(str)

            # Fill NaN values for token columns with 0
            token_columns = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]
            for col in token_columns:
                if col in assistant_df.columns:
                    assistant_df[col] = assistant_df[col].fillna(0).astype(int)

            # Calculate total effective input tokens
            assistant_df["effective_input_tokens"] = (
                assistant_df["input_tokens"]
                + assistant_df["cache_creation_input_tokens"]
                + (assistant_df["cache_read_input_tokens"] * 0.1)
            )
            assistant_df["total_tokens"] = assistant_df["effective_input_tokens"] + assistant_df["output_tokens"]

        # Extract system/error messages
        system_query = """
        SELECT 
            filename as source_file,
            timezone(current_setting('TimeZone'), timestamp::TIMESTAMP) as timestamp,
            type as log_type,
            CASE 
                WHEN TRY_CAST(isApiErrorMessage AS BOOLEAN) = true THEN 'error'
                WHEN type = 'system' THEN 'system'
                ELSE 'info'
            END as level,
            COALESCE(
                TRY_CAST(message.content AS VARCHAR),
                TRY_CAST(summary AS VARCHAR),
                'System message'
            ) as content,
            TRY_CAST(isApiErrorMessage AS BOOLEAN) as is_api_error,
            sessionId as session_id,
            uuid,
            parentUuid as parent_uuid,
            cwd,
            userType as user_type
        FROM all_logs
        WHERE type = 'system' 
           OR TRY_CAST(isApiErrorMessage AS BOOLEAN) = true
           OR TRY_CAST(isMeta AS BOOLEAN) = true
        """

        system_df = conn.execute(system_query).df()

        if not system_df.empty:
            system_df["timestamp"] = pd.to_datetime(system_df["timestamp"])
            system_df["project_path"] = system_df["source_file"].apply(lambda x: Path(x).parent.name)
            system_df["session_id"] = system_df["session_id"].astype(str)

        # Extract tool usage from assistant messages BEFORE removing the message column
        tool_df = extract_tool_uses_from_assistant_df(assistant_df)

        # Remove the message column from assistant_df to save memory
        if "message" in assistant_df.columns:
            assistant_df = assistant_df.drop(columns=["message"])

        # Log completion
        duration = (datetime.now() - start_time).total_seconds()
        log_with_context(
            logger,
            "INFO",
            "All data loaded successfully",
            duration_seconds=duration,
            assistant_rows=len(assistant_df),
            system_rows=len(system_df),
            tool_rows=len(tool_df),
        )

        return assistant_df, system_df, tool_df

    except Exception as e:
        log_with_context(
            logger,
            "ERROR",
            "Failed to load logs",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()
