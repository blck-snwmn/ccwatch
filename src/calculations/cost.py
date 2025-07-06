"""Cost calculation module for ccwatch.

This module handles all cost-related calculations based on model usage and token consumption.
"""

import logging

import pandas as pd

from config import AppConfig
from constants import CACHE_READ_DISCOUNT, TOKENS_PER_MILLION
from utils.logging_config import get_logger, log_with_context

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()


def calculate_cost(row: pd.Series) -> float:
    """Calculate cost for a single row based on model and token usage.

    Args:
        row: Pandas Series containing model and token information

    Returns:
        Calculated cost in dollars
    """
    model = row.get("model", "")
    pricing = config.get_model_pricing(model)

    # Get token counts with defaults
    input_tokens = row.get("input_tokens", 0)
    cache_creation_tokens = row.get("cache_creation_input_tokens", 0)
    cache_read_tokens = row.get("cache_read_input_tokens", 0)
    output_tokens = row.get("output_tokens", 0)

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (input_tokens + cache_creation_tokens) * pricing["input"] / TOKENS_PER_MILLION
    cache_cost = cache_read_tokens * pricing["cache_read"] / TOKENS_PER_MILLION
    output_cost = output_tokens * pricing["output"] / TOKENS_PER_MILLION

    total_cost = input_cost + cache_cost + output_cost

    # Log cost calculation for debugging (only for first few rows in debug mode)
    # Constant for debug row limit
    debug_row_limit = 5
    if logger.isEnabledFor(logging.DEBUG) and hasattr(row, "name") and row.name < debug_row_limit:
        log_with_context(
            logger,
            "DEBUG",
            "Cost calculation",
            model=model,
            input_tokens=input_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
        )

    return total_cost


def calculate_cache_savings(cache_read: int, regular_input: int, cache_creation: int) -> float:
    """Calculate cache savings percentage.

    Args:
        cache_read: Number of cache read tokens
        regular_input: Number of regular input tokens
        cache_creation: Number of cache creation tokens

    Returns:
        Cache savings percentage (0-100)
    """
    total = regular_input + cache_creation + cache_read
    if total > 0:
        # Cache reads cost 10% of regular input, so we save 90%
        savings = (cache_read * (1 - CACHE_READ_DISCOUNT) / total) * 100
        return savings
    return 0.0


def calculate_daily_average_cost(df: pd.DataFrame, total_cost: float) -> float:
    """Calculate daily average cost based on data period.

    Args:
        df: DataFrame with timestamp column
        total_cost: Total cost for the period

    Returns:
        Daily average cost
    """
    if df.empty or "timestamp" not in df.columns:
        return 0.0

    days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)
    return total_cost / days


def aggregate_cost_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate costs and tokens by model.

    Args:
        df: DataFrame with model, cost, and token columns

    Returns:
        DataFrame with aggregated costs by model
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure cost column exists
    if "cost" not in df.columns:
        df["cost"] = df.apply(calculate_cost, axis=1)

    # Aggregate by model
    model_costs = (
        df.groupby("model")
        .agg(
            {
                "cost": "sum",
                "effective_input_tokens": "sum",
                "output_tokens": "sum",
                "input_tokens": "sum",
                "cache_creation_input_tokens": "sum",
                "cache_read_input_tokens": "sum",
            }
        )
        .round(2)
    )

    # Sort by cost descending
    model_costs = model_costs.sort_values("cost", ascending=False)

    # Add percentage columns
    total_cost = model_costs["cost"].sum()
    if total_cost > 0:
        model_costs["cost_percentage"] = (model_costs["cost"] / total_cost * 100).round(1)
    else:
        model_costs["cost_percentage"] = 0.0

    return model_costs


def aggregate_cost_by_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate costs by time period.

    Args:
        df: DataFrame with timestamp and cost columns
        freq: Pandas frequency string (e.g., 'D' for daily, 'W' for weekly)

    Returns:
        DataFrame with aggregated costs by time period
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    # Ensure cost column exists
    if "cost" not in df.columns:
        df["cost"] = df.apply(calculate_cost, axis=1)

    # Aggregate by time period
    daily_costs = df.groupby(pd.Grouper(key="timestamp", freq=freq))["cost"].sum().reset_index()

    return daily_costs


def calculate_cost_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculate various cost metrics.

    Args:
        df: DataFrame with cost and token information

    Returns:
        Dictionary of cost metrics
    """
    if df.empty:
        return {"total_cost": 0.0, "daily_avg_cost": 0.0, "cache_hit_rate": 0.0, "avg_cost_per_response": 0.0}

    # Ensure cost column exists
    if "cost" not in df.columns:
        df["cost"] = df.apply(calculate_cost, axis=1)

    total_cost = df["cost"].sum()
    daily_avg_cost = calculate_daily_average_cost(df, total_cost)

    # Calculate cache hit rate
    cache_rate = 0.0
    if "effective_input_tokens" in df.columns and df["effective_input_tokens"].sum() > 0:
        cache_read_tokens = df.get("cache_read_input_tokens", pd.Series([0])).sum()
        effective_input = df["effective_input_tokens"].sum()
        cache_rate = (cache_read_tokens / effective_input * 100) if effective_input > 0 else 0.0

    # Average cost per response
    avg_cost_per_response = total_cost / len(df) if len(df) > 0 else 0.0

    metrics = {
        "total_cost": total_cost,
        "daily_avg_cost": daily_avg_cost,
        "cache_hit_rate": cache_rate,
        "avg_cost_per_response": avg_cost_per_response,
    }

    log_with_context(logger, "DEBUG", "Cost metrics calculated", **metrics)

    return metrics
