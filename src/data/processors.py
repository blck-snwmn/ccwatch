"""Data processing and transformation module for ccwatch.

This module contains functions for processing and transforming data loaded from JSONL files.
"""

import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import AppConfig
from constants import MESSAGE_PREVIEW_LENGTH
from utils.logging_config import get_logger, log_with_context

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()


def get_jsonl_files() -> list[str]:
    """Search for ClaudeCode project log files.

    Returns:
        List of paths to JSONL files, sorted by modification time (newest first)
    """
    if not config.claude_projects_path.exists():
        log_with_context(
            logger, "WARNING", "ClaudeCode projects directory not found", path=str(config.claude_projects_path)
        )
        return []

    pattern = str(config.claude_projects_path / config.jsonl_pattern)
    files = glob.glob(pattern, recursive=True)

    log_with_context(logger, "DEBUG", "Found JSONL files", count=len(files), pattern=pattern)

    return sorted(files, key=os.path.getmtime, reverse=True)


def truncate_message(message: Optional[str], max_length: int = MESSAGE_PREVIEW_LENGTH) -> str:
    """Truncate a message to a maximum length.

    Args:
        message: Message to truncate
        max_length: Maximum length for the message

    Returns:
        Truncated message with ellipsis if necessary
    """
    if message and len(str(message)) > max_length:
        return str(message)[:max_length] + "..."
    return str(message) if message else ""


def process_assistant_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process assistant dataframe with additional calculations.

    Args:
        df: Raw assistant dataframe

    Returns:
        Processed dataframe with additional columns
    """
    if df.empty:
        return df

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract project path from source file
    if "source_file" in df.columns:
        df["project_path"] = df["source_file"].apply(lambda x: Path(x).parent.name)

    # Ensure session_id is string
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str)

    # Fill NaN values for token columns with 0
    token_columns = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]
    for col in token_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Calculate effective input tokens
    if all(col in df.columns for col in ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]):
        df["effective_input_tokens"] = (
            df["input_tokens"] + df["cache_creation_input_tokens"] + (df["cache_read_input_tokens"] * 0.1)
        )
        df["total_tokens"] = df["effective_input_tokens"] + df["output_tokens"]

    return df


def process_system_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process system/error dataframe.

    Args:
        df: Raw system dataframe

    Returns:
        Processed dataframe
    """
    if df.empty:
        return df

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract project path from source file
    if "source_file" in df.columns:
        df["project_path"] = df["source_file"].apply(lambda x: Path(x).parent.name)

    # Ensure session_id is string
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str)

    return df


def process_tool_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process tool usage dataframe.

    Args:
        df: Raw tool dataframe

    Returns:
        Processed dataframe
    """
    if df.empty:
        return df

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure session_id is string
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype(str)

    return df


def filter_by_date_range(
    df: pd.DataFrame, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Filter dataframe by date range.

    Args:
        df: Dataframe with timestamp column
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Filtered dataframe
    """
    if df.empty or "timestamp" not in df.columns:
        return df

    filtered_df = df.copy()

    if start_date:
        filtered_df = filtered_df[filtered_df["timestamp"] >= start_date]

    if end_date:
        filtered_df = filtered_df[filtered_df["timestamp"] <= end_date]

    return filtered_df


def aggregate_by_time(df: pd.DataFrame, freq: str = "1h", column: str = "timestamp") -> pd.DataFrame:
    """Aggregate dataframe by time frequency.

    Args:
        df: Dataframe to aggregate
        freq: Pandas frequency string (e.g., '1h', '1D', '5min')
        column: Column to group by

    Returns:
        Aggregated dataframe with count column
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()

    return df.groupby(pd.Grouper(key=column, freq=freq)).size().reset_index(name="count")
