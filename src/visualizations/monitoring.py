"""Error and warning monitoring visualization module for ccwatch.

This module handles the display of system errors, warnings, and recent logs.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import AppConfig
from constants import CHART_HEIGHT_SMALL, MESSAGE_PREVIEW_LENGTH
from data.processors import truncate_message
from utils.logging_config import get_logger

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()


def show_error_warning_monitoring(system_df: pd.DataFrame) -> None:
    """Display error and warning monitoring section.

    Args:
        system_df: System/error messages dataframe
    """
    st.header("🚨 Error & Warning Monitoring")
    st.caption("System messages, API errors, and model limit warnings")

    if system_df.empty:
        st.info("No errors or warnings detected in the current data period")
        return

    # Error/Warning metrics
    show_error_warning_metrics(system_df)

    # Timeline of errors and warnings
    col1, col2 = st.columns(2)

    with col1:
        show_error_timeline(system_df)

    with col2:
        show_message_type_distribution(system_df)

    # Recent errors and warnings table
    show_recent_errors_warnings(system_df)


def show_error_warning_metrics(system_df: pd.DataFrame) -> None:
    """Display error and warning metrics.

    Args:
        system_df: System/error messages dataframe
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_errors = len(system_df[system_df["is_api_error"]]) if "is_api_error" in system_df.columns else 0
        st.metric("API Errors", total_errors, help="Total number of API error messages")
    with col2:
        warning_count = len(system_df[system_df["level"] == "warning"]) if "level" in system_df.columns else 0
        st.metric("Warnings", warning_count, help="Total number of warning messages")
    with col3:
        unique_sessions = system_df["session_id"].nunique()
        st.metric("Affected Sessions", unique_sessions, help="Number of sessions with errors or warnings")
    with col4:
        # Check for model limit warnings
        model_limit_warnings = 0
        if "content" in system_df.columns:
            model_limit_warnings = len(
                system_df[system_df["content"].str.contains("limit reached", case=False, na=False)]
            )
        st.metric("Model Limit Warnings", model_limit_warnings, help="Number of model limit reached warnings")


def show_error_timeline(system_df: pd.DataFrame) -> None:
    """Display error timeline chart.

    Args:
        system_df: System/error messages dataframe
    """
    # Error timeline
    if "is_api_error" in system_df.columns:
        error_df = system_df[system_df["is_api_error"]]
    else:
        error_df = system_df[system_df["level"] == "error"] if "level" in system_df.columns else pd.DataFrame()

    if not error_df.empty:
        error_timeline = error_df.groupby(pd.Grouper(key="timestamp", freq="1h")).size().reset_index(name="count")

        if not error_timeline.empty:
            fig_error_timeline = px.line(
                error_timeline,
                x="timestamp",
                y="count",
                title="API Errors Over Time",
                height=CHART_HEIGHT_SMALL,
                markers=True,
            )
            fig_error_timeline.update_traces(hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>Errors: %{y}<extra></extra>")
            st.plotly_chart(fig_error_timeline, use_container_width=True)
    else:
        st.info("No API errors to display")


def show_message_type_distribution(system_df: pd.DataFrame) -> None:
    """Display message type distribution chart.

    Args:
        system_df: System/error messages dataframe
    """
    # Warning types distribution
    if "level" in system_df.columns:
        level_counts = system_df["level"].value_counts()
        if not level_counts.empty:
            fig_levels = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title="Message Types Distribution",
                height=CHART_HEIGHT_SMALL,
            )
            fig_levels.update_traces(
                hovertemplate="Type: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
            )
            st.plotly_chart(fig_levels, use_container_width=True)


def show_recent_errors_warnings(system_df: pd.DataFrame) -> None:
    """Display recent errors and warnings table.

    Args:
        system_df: System/error messages dataframe
    """
    st.subheader("📋 Recent Errors & Warnings")
    recent_system = system_df.nlargest(20, "timestamp")[["timestamp", "level", "content", "project_path", "session_id"]]

    # Truncate long messages
    recent_system["content"] = recent_system["content"].apply(lambda x: truncate_message(x, MESSAGE_PREVIEW_LENGTH))

    # Apply color coding based on level
    def highlight_level(row):
        """Apply color coding based on message level"""
        if row["level"] == "error" or row.get("is_api_error", False):
            return ["background-color: #ffcccc"] * len(row)
        elif row["level"] == "warning":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(recent_system.style.apply(highlight_level, axis=1), use_container_width=True, height=400)


def show_recent_logs(df: pd.DataFrame) -> None:
    """Display recent AI response logs.

    Args:
        df: Assistant messages dataframe
    """
    st.header("📋 Recent AI Responses")
    st.caption("Latest 20 AI assistant responses with message previews")

    if df.empty:
        st.info("No AI responses to display")
        return

    # Select columns to display
    columns_to_show = ["timestamp", "model", "session_id", "project_path", "message_content"]
    available_columns = [col for col in columns_to_show if col in df.columns]

    recent_logs = df.nlargest(20, "timestamp")[available_columns]

    # Truncate long messages
    if "message_content" in recent_logs.columns:
        recent_logs["message_content"] = recent_logs["message_content"].apply(
            lambda x: truncate_message(x, MESSAGE_PREVIEW_LENGTH)
        )

    st.dataframe(recent_logs, use_container_width=True, height=400)


def show_token_summary_table(token_summary: pd.DataFrame) -> None:
    """Display token usage summary table.

    Args:
        token_summary: Token summary dataframe with model-level aggregations
    """
    st.subheader("Token Usage by Model")
    st.caption("Detailed token consumption breakdown by model type")

    # Format for display
    display_cols = {
        "input_tokens": "Input Tokens",
        "cache_creation_input_tokens": "Cache Creation",
        "cache_read_input_tokens": "Cache Read",
        "output_tokens": "Output Tokens",
        "cost": "Cost ($)",
        "input_pct": "Input %",
        "output_pct": "Output %",
        "cost_pct": "Cost %",
    }

    # Rename columns if they exist
    for old_col, new_col in display_cols.items():
        if old_col in token_summary.columns:
            token_summary = token_summary.rename(columns={old_col: new_col})

    # Define formatting for each column
    format_dict = {
        "Input Tokens": "{:,.0f}",
        "Cache Creation": "{:,.0f}",
        "Cache Read": "{:,.0f}",
        "Output Tokens": "{:,.0f}",
        "Cost ($)": "${:.2f}",
        "Input %": "{:.1f}%",
        "Output %": "{:.1f}%",
        "Cost %": "{:.1f}%",
    }

    # Apply formatting only to columns that exist
    existing_format = {col: fmt for col, fmt in format_dict.items() if col in token_summary.columns}

    st.dataframe(
        token_summary.style.format(existing_format),
        use_container_width=True,
    )
