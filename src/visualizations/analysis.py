"""Analysis visualization module for ccwatch.

This module handles the display of session analysis, project insights, and tool usage analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import AppConfig
from constants import CHART_HEIGHT_DEFAULT, CHART_HEIGHT_SMALL, DEFAULT_IDLE_THRESHOLD_MINUTES
from utils.logging_config import get_logger
from utils.session_utils import calculate_session_metrics

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()


def show_session_analysis(df: pd.DataFrame) -> None:
    """Display session analysis with duration and activity patterns.

    Args:
        df: Assistant messages dataframe
    """
    st.header("🎯 Session Analysis")
    st.caption("Analysis of ClaudeCode session duration and activity patterns")

    # Calculate comprehensive session metrics
    session_data = calculate_session_metrics(df, idle_threshold_minutes=DEFAULT_IDLE_THRESHOLD_MINUTES)
    session_data = session_data.sort_values("start_time", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        show_session_duration_distribution(session_data)

    with col2:
        show_session_message_distribution(session_data)

    # Session activity insights
    show_session_activity_insights(session_data)

    # Activity type distribution and scatter plot
    col1, col2 = st.columns(2)
    with col1:
        show_activity_type_distribution(session_data)

    with col2:
        show_session_activity_scatter(session_data)


def show_session_duration_distribution(session_data: pd.DataFrame) -> None:
    """Display session duration distribution chart.

    Args:
        session_data: Session metrics dataframe
    """
    # Show active duration distribution
    duration_bins = [0, 5, 15, 30, 60, float("inf")]
    duration_labels = ["0-5 min", "5-15 min", "15-30 min", "30-60 min", "60+ min"]
    session_data["duration_category"] = pd.cut(
        session_data["active_duration_minutes"], bins=duration_bins, labels=duration_labels
    )

    duration_counts = session_data["duration_category"].value_counts()
    fig_duration = px.bar(
        x=duration_counts.index,
        y=duration_counts.values,
        title="Active Session Duration Distribution",
        labels={"x": "Active Duration", "y": "Number of Sessions"},
        height=CHART_HEIGHT_DEFAULT,
    )
    fig_duration.update_traces(hovertemplate="Active Duration: %{x}<br>Sessions: %{y}<extra></extra>")
    st.plotly_chart(fig_duration, use_container_width=True)


def show_session_message_distribution(session_data: pd.DataFrame) -> None:
    """Display session message count distribution chart.

    Args:
        session_data: Session metrics dataframe
    """
    message_bins = [0, 10, 50, 100, 200, float("inf")]
    message_labels = ["1-10", "11-50", "51-100", "101-200", "200+"]
    session_data["message_category"] = pd.cut(
        session_data["ai_response_count"], bins=message_bins, labels=message_labels
    )

    message_counts = session_data["message_category"].value_counts()
    fig_messages = px.bar(
        x=message_counts.index,
        y=message_counts.values,
        title="Session AI Response Count Distribution",
        labels={"x": "AI Response Count", "y": "Number of Sessions"},
        height=CHART_HEIGHT_DEFAULT,
    )
    fig_messages.update_traces(hovertemplate="AI Responses: %{x}<br>Sessions: %{y}<extra></extra>")
    st.plotly_chart(fig_messages, use_container_width=True)


def show_session_activity_insights(session_data: pd.DataFrame) -> None:
    """Display session activity insights metrics.

    Args:
        session_data: Session metrics dataframe
    """
    st.subheader("📊 Session Activity Insights")
    st.caption("Detailed analysis of session patterns and idle time")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_active = session_data["active_duration_minutes"].mean()
        st.metric(
            "Avg Active Duration",
            f"{avg_active:.1f} min",
            help="Average active session duration excluding idle periods (>30 min gaps).",
        )
    with col2:
        avg_idle_pct = session_data["idle_percentage"].mean()
        st.metric(
            "Avg Idle Time", f"{avg_idle_pct:.1f}%", help="Average percentage of idle time in sessions (gaps >30 min)."
        )
    with col3:
        avg_msg_per_min = session_data["messages_per_minute"].mean()
        st.metric(
            "Avg Messages/Min",
            f"{avg_msg_per_min:.2f}",
            help="Average AI responses per active minute across all sessions.",
        )
    with col4:
        sessions_with_idle = (session_data["idle_periods"] > 0).sum()
        idle_pct = sessions_with_idle / len(session_data) * 100
        st.metric(
            "Sessions with Idle",
            f"{sessions_with_idle} ({idle_pct:.1f}%)",
            help="Number of sessions that had idle periods (>30 min gaps).",
        )


def show_activity_type_distribution(session_data: pd.DataFrame) -> None:
    """Display activity type distribution chart.

    Args:
        session_data: Session metrics dataframe
    """
    activity_counts = session_data["activity_type"].value_counts()
    fig_activity = px.pie(
        values=activity_counts.values,
        names=activity_counts.index,
        title="Session Activity Types",
        height=CHART_HEIGHT_SMALL,
    )
    fig_activity.update_traces(
        hovertemplate="Type: %{label}<br>Sessions: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    st.plotly_chart(fig_activity, use_container_width=True)


def show_session_activity_scatter(session_data: pd.DataFrame) -> None:
    """Display session activity scatter plot.

    Args:
        session_data: Session metrics dataframe
    """
    # Scatter plot of duration vs messages
    fig_scatter = px.scatter(
        session_data.head(100),  # Limit to recent 100 sessions for clarity
        x="active_duration_minutes",
        y="ai_response_count",
        color="activity_type",
        size="total_tokens",
        title="Session Activity Pattern (Recent 100)",
        labels={"active_duration_minutes": "Active Duration (min)", "ai_response_count": "AI Responses"},
        height=CHART_HEIGHT_SMALL,
    )
    fig_scatter.update_traces(
        hovertemplate="Duration: %{x:.1f} min<br>AI Responses: %{y}<br>Activity: %{customdata[0]}<extra></extra>"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def show_tool_usage_analysis(tool_df: pd.DataFrame) -> None:
    """Display tool usage analysis.

    Args:
        tool_df: Tool usage dataframe
    """
    st.header("🛠️ Tool Usage Analysis")
    st.caption("Analysis of tool usage patterns and frequency")

    if tool_df.empty:
        st.info("No tool usage data available in the current data period")
        return

    # Tool usage metrics
    show_tool_usage_metrics(tool_df)

    # Tool usage distribution
    col1, col2 = st.columns(2)

    with col1:
        show_top_tools_chart(tool_df)

    with col2:
        show_tool_usage_timeline(tool_df)

    # Tool usage by project
    show_tool_usage_by_project(tool_df)

    # TODO tool analysis if available
    show_todo_tool_analysis(tool_df)


def show_tool_usage_metrics(tool_df: pd.DataFrame) -> None:
    """Display tool usage metrics.

    Args:
        tool_df: Tool usage dataframe
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_tool_uses = len(tool_df)
        st.metric("Total Tool Uses", total_tool_uses, help="Total number of tool invocations")
    with col2:
        unique_tools = tool_df["tool_name"].nunique()
        st.metric("Unique Tools", unique_tools, help="Number of different tools used")
    with col3:
        sessions_with_tools = tool_df["session_id"].nunique()
        st.metric("Sessions with Tools", sessions_with_tools, help="Number of sessions that used tools")
    with col4:
        # Count WebSearch tool uses
        web_searches = len(tool_df[tool_df["tool_name"] == "WebSearch"]) if "tool_name" in tool_df.columns else 0
        st.metric("Web Searches", web_searches, help="Total number of WebSearch tool uses")


def show_top_tools_chart(tool_df: pd.DataFrame) -> None:
    """Display top tools by usage chart.

    Args:
        tool_df: Tool usage dataframe
    """
    # Top tools by usage
    tool_counts = tool_df["tool_name"].value_counts().head(10)
    if not tool_counts.empty:
        fig_tools = px.bar(
            x=tool_counts.values,
            y=tool_counts.index,
            orientation="h",
            title="Top 10 Most Used Tools",
            labels={"x": "Usage Count", "y": "Tool Name"},
            height=CHART_HEIGHT_DEFAULT,
        )
        fig_tools.update_traces(hovertemplate="Tool: %{y}<br>Uses: %{x}<extra></extra>")
        fig_tools.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_tools, use_container_width=True)


def show_tool_usage_timeline(tool_df: pd.DataFrame) -> None:
    """Display tool usage timeline chart.

    Args:
        tool_df: Tool usage dataframe
    """
    # Tool usage over time
    daily_tools = tool_df.groupby([pd.Grouper(key="timestamp", freq="D"), "tool_name"]).size().reset_index(name="count")
    # Get top 5 tools for cleaner visualization
    top_tools = tool_df["tool_name"].value_counts().head(5).index.tolist()
    daily_tools_filtered = daily_tools[daily_tools["tool_name"].isin(top_tools)]

    if not daily_tools_filtered.empty:
        fig_timeline = px.line(
            daily_tools_filtered,
            x="timestamp",
            y="count",
            color="tool_name",
            title="Daily Tool Usage (Top 5 Tools)",
            height=CHART_HEIGHT_DEFAULT,
        )
        fig_timeline.update_traces(hovertemplate="Date: %{x|%Y-%m-%d}<br>Uses: %{y}<extra></extra>")
        st.plotly_chart(fig_timeline, use_container_width=True)


def show_tool_usage_by_project(tool_df: pd.DataFrame) -> None:
    """Display tool usage by project heatmap.

    Args:
        tool_df: Tool usage dataframe
    """
    st.subheader("Tool Usage by Project")
    project_tool_usage = tool_df.groupby(["project_path", "tool_name"]).size().reset_index(name="count")

    # Get top projects by tool usage
    top_projects_by_tools = tool_df["project_path"].value_counts().head(10).index.tolist()
    project_tool_filtered = project_tool_usage[project_tool_usage["project_path"].isin(top_projects_by_tools)]

    if not project_tool_filtered.empty:
        # Create pivot table for heatmap
        pivot_data = project_tool_filtered.pivot_table(
            index="project_path", columns="tool_name", values="count", fill_value=0
        )

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale="Blues",
                hovertemplate="Project: %{y}<br>Tool: %{x}<br>Uses: %{z}<extra></extra>",
            )
        )

        fig_heatmap.update_layout(
            title="Tool Usage Heatmap (Top 10 Projects)",
            xaxis_title="Tool Name",
            yaxis_title="Project",
            height=CHART_HEIGHT_DEFAULT,
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)


def show_todo_tool_analysis(tool_df: pd.DataFrame) -> None:
    """Display TODO tool usage analysis.

    Args:
        tool_df: Tool usage dataframe
    """
    todo_tools = tool_df[tool_df["tool_name"].str.contains("todo", case=False, na=False)]
    if not todo_tools.empty:
        st.subheader("📝 TODO Tool Usage")
        col1, col2, col3 = st.columns(3)
        with col1:
            todo_count = len(todo_tools)
            st.metric("TODO Tool Uses", todo_count)
        with col2:
            todo_sessions = todo_tools["session_id"].nunique()
            st.metric("Sessions with TODO", todo_sessions)
        with col3:
            todo_projects = todo_tools["project_path"].nunique()
            st.metric("Projects with TODO", todo_projects)


def show_project_insights(df: pd.DataFrame) -> None:
    """Display project insights and statistics.

    Args:
        df: Assistant messages dataframe
    """
    st.header("💼 Project Insights")
    st.caption("Project usage statistics and activity patterns")

    project_stats = df.groupby("project_path").agg(
        {
            "timestamp": ["min", "max", "count"],
            "session_id": "nunique",
            "model": lambda x: x.value_counts().to_dict() if not x.empty else {},
        }
    )

    project_stats.columns = ["first_use", "last_use", "ai_response_count", "session_count", "model_usage"]
    project_stats["days_active"] = (project_stats["last_use"] - project_stats["first_use"]).dt.days + 1
    project_stats = project_stats.sort_values("ai_response_count", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        show_top_projects_chart(project_stats)

    with col2:
        show_recent_projects_chart(project_stats)


def show_top_projects_chart(project_stats: pd.DataFrame) -> None:
    """Display top projects by response count chart.

    Args:
        project_stats: Project statistics dataframe
    """
    top_projects = project_stats.head(config.max_projects_to_show)
    fig_top_projects = px.bar(
        x=top_projects["ai_response_count"],
        y=top_projects.index,
        orientation="h",
        title=f"Top {config.max_projects_to_show} Projects (by AI Response Count)",
        labels={"x": "AI Response Count", "y": "Project"},
        height=CHART_HEIGHT_DEFAULT,
    )
    fig_top_projects.update_traces(hovertemplate="Project: %{y}<br>AI Responses: %{x}<extra></extra>")
    fig_top_projects.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_top_projects, use_container_width=True)


def show_recent_projects_chart(project_stats: pd.DataFrame) -> None:
    """Display recently used projects chart.

    Args:
        project_stats: Project statistics dataframe
    """
    recent_projects = project_stats.sort_values("last_use", ascending=False).head(config.max_projects_to_show)
    # Get timezone from last_use column
    tz = recent_projects["last_use"].dt.tz if not recent_projects.empty else None
    recent_projects["days_since_last_use"] = (pd.Timestamp.now(tz=tz) - recent_projects["last_use"]).dt.days

    fig_recent = px.scatter(
        recent_projects,
        x="days_since_last_use",
        y=recent_projects.index,
        size="ai_response_count",
        title="Recently Used Projects",
        labels={"x": "Days Since Last Use", "y": "Project"},
        height=CHART_HEIGHT_DEFAULT,
    )
    fig_recent.update_traces(
        hovertemplate="Project: %{y}<br>Days since last use: %{x}<br>AI Responses: %{marker.size}<extra></extra>"
    )
    fig_recent.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_recent, use_container_width=True)
