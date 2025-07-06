"""Main Streamlit application for ccwatch.

This module serves as the entry point for the ccwatch application,
orchestrating data loading and visualization components.
"""

import os
from datetime import datetime
from pathlib import Path

import duckdb
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from calculations.cost import aggregate_cost_by_model, aggregate_cost_by_time, calculate_cost, calculate_cost_metrics
from config import AppConfig
from constants import ERROR_CLAUDE_PATH_NOT_FOUND, ERROR_NO_DATA
from data.loader import load_all_logs_with_duckdb
from data.processors import (
    get_jsonl_files,
    process_assistant_dataframe,
    process_system_dataframe,
    process_tool_dataframe,
)
from utils.logging_config import get_logger, log_with_context
from visualizations.analysis import show_project_insights, show_session_analysis, show_tool_usage_analysis
from visualizations.charts import show_cost_charts, show_heatmap, show_model_analysis, show_overall_graphs
from visualizations.metrics import display_cost_metrics, display_date_range, show_cost_calculation_details, show_metrics
from visualizations.monitoring import show_error_warning_monitoring, show_recent_logs, show_token_summary_table

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()

# Page configuration
st.set_page_config(page_title="ccwatch - ClaudeCode Monitor", layout="wide")


def setup_sidebar(jsonl_files: list) -> None:
    """Set up the sidebar with monitoring status and controls.

    Args:
        jsonl_files: List of detected JSONL files
    """
    with st.sidebar:
        st.header("🔍 ccwatch")
        st.caption("ClaudeCode Monitor")

        st.write("📊 Monitoring Status")
        st.write(f"- Auto-refresh: Every {config.check_interval // 60} minutes")
        st.write(f"- Update Count: {st.session_state['update_count']}")

        # Get and display current timezone
        try:
            conn_tz = duckdb.connect(":memory:")
            conn_tz.execute("LOAD icu")
            current_tz = conn_tz.execute("SELECT current_setting('TimeZone') as tz").fetchone()[0]
            conn_tz.close()
            st.write(f"- Timezone: {current_tz}")
        except Exception:
            st.write("- Timezone: System Default")

        st.divider()
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()

        # File information
        if jsonl_files:
            st.divider()
            st.write(f"📁 Files Detected: {len(jsonl_files)}")

            # Latest 5 files
            with st.expander("Latest Files"):
                for f in jsonl_files[:5]:
                    file_name = Path(f).name
                    mtime = datetime.fromtimestamp(os.path.getmtime(f))
                    st.caption(f"- {file_name}")
                    st.caption(f"  Updated: {mtime.strftime('%H:%M:%S')}")


def show_token_and_cost_analysis(df):
    """Display token usage and cost analysis.

    Args:
        df: Assistant messages dataframe
    """
    st.header("💰 Token Usage & Cost Analysis")
    st.caption("Cost calculation based on token usage and model pricing")

    # Calculate costs
    df["cost"] = df.apply(calculate_cost, axis=1)

    # Show data period
    display_date_range(df)

    # Show calculation details
    show_cost_calculation_details(df)

    # Calculate and display cost metrics
    metrics = calculate_cost_metrics(df)
    display_cost_metrics(
        metrics["total_cost"], metrics["daily_avg_cost"], metrics["cache_hit_rate"], metrics["avg_cost_per_response"]
    )

    # Cost breakdown by model and daily trend
    model_costs = aggregate_cost_by_model(df)
    daily_costs = aggregate_cost_by_time(df, "D")
    show_cost_charts(df, model_costs, daily_costs)

    # Token usage summary table
    if not model_costs.empty:
        # Calculate percentage columns
        total_input = df[["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]].sum().sum()
        total_output = df["output_tokens"].sum()
        total_cost = model_costs["cost"].sum()

        model_costs["input_pct"] = (
            model_costs[["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]].sum(axis=1)
            / total_input
            * 100
        ).round(1)
        model_costs["output_pct"] = (model_costs["output_tokens"] / total_output * 100).round(1)
        model_costs["cost_pct"] = (model_costs["cost"] / total_cost * 100).round(1)

        show_token_summary_table(model_costs)


def main():
    """Main application entry point."""
    st.title("🔍 ccwatch - ClaudeCode Monitor")
    st.markdown("Monitor and visualize ClaudeCode logs")

    # Application startup log
    log_with_context(logger, "INFO", "ccwatch application started", claude_path=str(config.claude_projects_path))

    # Auto-refresh setup
    count = st_autorefresh(interval=config.check_interval * 1000, limit=None, key="autorefresh")

    # Initialize session state
    if "update_count" not in st.session_state:
        st.session_state["update_count"] = 0

    if count > 0:
        st.session_state["update_count"] = count

    # Get file list
    jsonl_files = get_jsonl_files()

    # Set up sidebar
    setup_sidebar(jsonl_files)

    # Main content
    if not jsonl_files:
        st.info("Searching for ClaudeCode log files...")
        st.warning(ERROR_CLAUDE_PATH_NOT_FOUND.format(config.claude_projects_path))
        return

    cache_key = st.session_state["update_count"]

    # Data loading start log
    log_with_context(logger, "INFO", "Starting data load", update_count=cache_key, files_found=len(jsonl_files))

    # Load all data at once using the consolidated function
    assistant_df, system_df, tool_df = load_all_logs_with_duckdb(cache_key)

    # Process dataframes
    if assistant_df is not None and not assistant_df.empty:
        assistant_df = process_assistant_dataframe(assistant_df)
    if system_df is not None and not system_df.empty:
        system_df = process_system_dataframe(system_df)
    if tool_df is not None and not tool_df.empty:
        tool_df = process_tool_dataframe(tool_df)

    if assistant_df is not None and not assistant_df.empty:
        # Data load success log
        log_with_context(
            logger,
            "INFO",
            "Data loaded successfully",
            rows=len(assistant_df),
            sessions=assistant_df["session_id"].nunique(),
            projects=assistant_df["project_path"].nunique(),
        )

        # Display all sections
        show_metrics(assistant_df)
        show_overall_graphs(assistant_df)
        show_error_warning_monitoring(system_df)
        show_session_analysis(assistant_df)
        show_model_analysis(assistant_df)
        show_tool_usage_analysis(tool_df)
        show_token_and_cost_analysis(assistant_df)
        show_heatmap(assistant_df)
        show_project_insights(assistant_df)
        show_recent_logs(assistant_df)

        # Footer information
        st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error(ERROR_NO_DATA)
        log_with_context(logger, "ERROR", "Failed to load data - DataFrame is None or empty")


if __name__ == "__main__":
    main()
