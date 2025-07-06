"""Chart visualization module for ccwatch.

This module handles the display of various charts and graphs in the Streamlit UI.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import AppConfig
from constants import (
    CHART_HEIGHT_DEFAULT,
    CHART_HEIGHT_SMALL,
    DAILY_ACTIVITY_INTERVAL,
    HEATMAP_DAYS,
    HOURLY_ACTIVITY_INTERVAL,
    MINUTE_ACTIVITY_INTERVAL,
)
from data.processors import aggregate_by_time
from utils.logging_config import get_logger

# Initialize logger
logger = get_logger()

# Initialize config
config = AppConfig.from_env()


def show_overall_graphs(df: pd.DataFrame) -> None:
    """Display overall statistics graphs.

    Args:
        df: Assistant messages dataframe
    """
    st.header("📈 Overall Statistics")
    st.caption("Time-based analysis and model distribution of AI assistant responses")

    col1, col2 = st.columns(2)

    with col1:
        timeline_data = aggregate_by_time(df, HOURLY_ACTIVITY_INTERVAL)
        fig_timeline = px.line(
            timeline_data,
            x="timestamp",
            y="count",
            title="AI Responses by Hour",
            height=CHART_HEIGHT_DEFAULT,
        )
        fig_timeline.update_traces(hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>AI Responses: %{y}<extra></extra>")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col2:
        model_counts = df["model"].value_counts()
        fig_model_pie = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title="AI Responses by Model",
            height=CHART_HEIGHT_DEFAULT,
        )
        fig_model_pie.update_traces(
            hovertemplate="Model: %{label}<br>AI Responses: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        st.plotly_chart(fig_model_pie, use_container_width=True)

    # 24-hour activity timeline
    show_24h_activity_timeline(df)


def show_24h_activity_timeline(df: pd.DataFrame) -> None:
    """Display 24-hour activity timeline.

    Args:
        df: Assistant messages dataframe
    """
    st.subheader("📊 24-Hour Activity Timeline")
    st.caption("Recent activity pattern with 5-minute granularity")

    # Get the timezone from the dataframe
    tz = df["timestamp"].dt.tz if not df.empty else None
    last_24h = pd.Timestamp.now(tz=tz) - pd.Timedelta(hours=24)
    recent_df = df[df["timestamp"] > last_24h]

    if len(recent_df) > 0:
        # 5-minute intervals for more granular view
        minute_activity = aggregate_by_time(recent_df, MINUTE_ACTIVITY_INTERVAL)
        fig_24h = px.line(
            minute_activity,
            x="timestamp",
            y="count",
            title="24-Hour Activity (AI Responses per 5 minutes)",
            height=CHART_HEIGHT_SMALL,
            markers=True,
        )
        fig_24h.update_traces(hovertemplate="Time: %{x|%H:%M}<br>AI Responses: %{y}<extra></extra>")
        fig_24h.update_layout(
            xaxis_title="Time",
            yaxis_title="AI Responses",
            showlegend=False,
            xaxis=dict(
                tickformat="%H:%M",
                dtick=3600000,  # Show tick every hour (in milliseconds)
            ),
        )
        st.plotly_chart(fig_24h, use_container_width=True)
    else:
        st.info("No activity in the last 24 hours")


def show_heatmap(df: pd.DataFrame) -> None:
    """Display GitHub-style usage frequency heatmap.

    Args:
        df: Assistant messages dataframe
    """
    st.header("📅 Usage Frequency Heatmap")
    st.caption("GitHub-style visualization of daily AI assistant usage over the past year")

    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size().reset_index(name="count")

    end_date = daily_counts["date"].max()
    start_date = end_date - pd.Timedelta(days=HEATMAP_DAYS)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    date_df = pd.DataFrame({"date": date_range.date})

    heatmap_data = date_df.merge(daily_counts, on="date", how="left").fillna(0)
    heatmap_data["date"] = pd.to_datetime(heatmap_data["date"])

    heatmap_data["weekday"] = heatmap_data["date"].dt.weekday
    heatmap_data["week"] = (heatmap_data["date"] - heatmap_data["date"].min()).dt.days // 7

    matrix = heatmap_data.pivot_table(index="weekday", columns="week", values="count", fill_value=0)

    colorscale = [[0.0, "#e1e4e8"], [0.2, "#9be9a8"], [0.4, "#40c463"], [0.6, "#30a14e"], [1.0, "#216e39"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=colorscale,
            showscale=True,
            hovertemplate="Week %{x}<br>%{y}<br>AI Responses: %{z}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title="Usage Frequency for Past 52 Weeks",
        xaxis_title="Week",
        yaxis_title="Day of Week",
        height=CHART_HEIGHT_SMALL,
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Heatmap metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(daily_counts), help="Total number of days with recorded activity.")
    with col2:
        st.metric(
            "Avg AI Responses/Day",
            f"{daily_counts['count'].mean():.1f}",
            help="Average number of AI responses per day (including days with no activity).",
        )
    with col3:
        st.metric(
            "Max AI Responses/Day",
            daily_counts["count"].max(),
            help="Maximum number of AI responses recorded in a single day.",
        )
    with col4:
        active_days = (daily_counts["count"] > 0).sum()
        st.metric(
            "Active Days",
            f"{active_days} ({active_days / len(daily_counts) * 100:.1f}%)",
            help="Days with at least one AI response (percentage of total days).",
        )


def show_model_analysis(df: pd.DataFrame) -> None:
    """Display model-based analysis charts.

    Args:
        df: Assistant messages dataframe
    """
    st.header("🤖 Model Analysis")
    st.caption("Model usage patterns across projects and time")

    col1, col2 = st.columns(2)

    with col1:
        show_model_usage_by_project(df)

    with col2:
        show_daily_model_usage(df)


def show_model_usage_by_project(df: pd.DataFrame) -> None:
    """Display model usage by project chart.

    Args:
        df: Assistant messages dataframe
    """
    # Get top projects by AI response count
    project_totals = df["project_path"].value_counts().head(config.max_projects_to_show)

    # Get model breakdown for these projects
    model_project_data = df.groupby(["project_path", "model"]).size().reset_index(name="count")
    model_project_filtered = model_project_data[model_project_data["project_path"].isin(project_totals.index)]

    # Convert project_totals to DataFrame for easier manipulation
    project_order_df = project_totals.reset_index()
    project_order_df.columns = ["project_path", "total_count"]
    project_order_df["order"] = range(len(project_order_df))

    # Add order to filtered data
    model_project_filtered = model_project_filtered.merge(
        project_order_df[["project_path", "order"]], on="project_path"
    )

    # Sort by order to ensure correct display
    model_project_filtered = model_project_filtered.sort_values("order")

    # Create the figure with model breakdown
    fig_model_project = px.bar(
        model_project_filtered,
        x="count",
        y="project_path",
        color="model",
        orientation="h",
        title=f"Model Usage by Project (Top {config.max_projects_to_show} Projects)",
        height=CHART_HEIGHT_DEFAULT,
        category_orders={"project_path": project_totals.index.tolist()[::-1]},  # Reverse for correct display
    )
    fig_model_project.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_model_project, use_container_width=True)


def show_daily_model_usage(df: pd.DataFrame) -> None:
    """Display daily model usage chart.

    Args:
        df: Assistant messages dataframe
    """
    model_daily = (
        df.groupby([pd.Grouper(key="timestamp", freq=DAILY_ACTIVITY_INTERVAL), "model"])
        .size()
        .reset_index(name="count")
    )
    fig_model_daily = px.bar(
        model_daily,
        x="timestamp",
        y="count",
        color="model",
        title="Daily Model Usage",
        height=CHART_HEIGHT_DEFAULT,
    )
    st.plotly_chart(fig_model_daily, use_container_width=True)


def show_cost_charts(df: pd.DataFrame, model_costs: pd.DataFrame, daily_costs: pd.DataFrame) -> None:
    """Display cost-related charts.

    Args:
        df: Assistant messages dataframe with cost column
        model_costs: Cost aggregated by model
        daily_costs: Cost aggregated by day
    """
    col1, col2 = st.columns(2)

    with col1:
        if not model_costs.empty:
            fig_cost_by_model = px.bar(
                model_costs,
                y=model_costs.index,
                x="cost",
                orientation="h",
                title="Cost by Model",
                labels={"cost": "Cost ($)", "index": "Model"},
                height=CHART_HEIGHT_DEFAULT,
            )
            fig_cost_by_model.update_traces(hovertemplate="Model: %{y}<br>Cost: $%{x:.2f}<extra></extra>")
            fig_cost_by_model.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_cost_by_model, use_container_width=True)

    with col2:
        if not daily_costs.empty:
            fig_daily_cost = px.line(
                daily_costs,
                x="timestamp",
                y="cost",
                title="Daily Cost Trend",
                labels={"cost": "Cost ($)", "timestamp": "Date"},
                height=CHART_HEIGHT_DEFAULT,
            )
            fig_daily_cost.update_traces(hovertemplate="Date: %{x|%Y-%m-%d}<br>Cost: $%{y:.2f}<extra></extra>")
            st.plotly_chart(fig_daily_cost, use_container_width=True)
