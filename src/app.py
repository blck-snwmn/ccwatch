import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils.logging_config import get_logger, log_with_context

# ロガーの初期化
logger = get_logger()

st.set_page_config(page_title="ccwatch - ClaudeCode Monitor", layout="wide")

# Allow overriding the path via environment variable
DEFAULT_CLAUDE_PATH = Path.home() / ".claude" / "projects"
CLAUDE_PROJECTS_PATH = Path(os.getenv("CLAUDE_PROJECTS_PATH", str(DEFAULT_CLAUDE_PATH)))
JSONL_PATTERN = "**/*.jsonl"
MAX_PROJECTS_TO_SHOW = 10
CHECK_INTERVAL = 5 * 60  # 5 minutes (in seconds)

# Model pricing information (per 1M tokens)
# Source: https://docs.anthropic.com/en/docs/about-claude/models (2025-01)
MODEL_PRICING = {
    # Claude Sonnet 3.5 / 3.7
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    # Claude Opus 3
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    # Claude Sonnet 3
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    # Claude Haiku 3.5
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "cache_read": 0.08},
    # Claude Haiku 3
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cache_read": 0.03},
    # Claude Opus 4
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    # Claude Sonnet 4 (if model ID exists)
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    # Default pricing for unknown models
    "default": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
}


def get_jsonl_files():
    """Search for ClaudeCode project log files"""
    if not CLAUDE_PROJECTS_PATH.exists():
        st.warning(f"ClaudeCode projects directory not found: {CLAUDE_PROJECTS_PATH}")
        return []

    pattern = str(CLAUDE_PROJECTS_PATH / JSONL_PATTERN)
    files = glob.glob(pattern, recursive=True)
    return sorted(files, key=os.path.getmtime, reverse=True)


@st.cache_data(ttl=3600)
def load_logs_with_duckdb(cache_key):
    """Load JSONL files directly using DuckDB

    Args:
        cache_key: Cache control key (update counter)
    """
    _ = cache_key  # Used for cache control

    start_time = datetime.now()

    conn = duckdb.connect(":memory:")

    # Load ICU extension for timezone support
    conn.execute("LOAD icu")

    # Use glob pattern to read all JSONL files at once
    glob_pattern = str(CLAUDE_PROJECTS_PATH / JSONL_PATTERN)

    query = f"""
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
        TRY_CAST(message.usage.output_tokens AS BIGINT) as output_tokens
    FROM read_json_auto('{glob_pattern}', format='newline_delimited', filename=true)
    WHERE type = 'assistant'
    """

    # DEBUG: Executing DuckDB query
    log_with_context(logger, "DEBUG", "Executing DuckDB query", glob_pattern=glob_pattern)

    try:
        df = conn.execute(query).df()
        # Timestamp is already in local timezone from DuckDB
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["project_path"] = df["source_file"].apply(lambda x: Path(x).parent.name)
        df["session_id"] = df["session_id"].astype(str)

        # Fill NaN values for token columns with 0
        token_columns = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]
        for col in token_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Calculate total effective input tokens (cache_read tokens count as 10% of regular tokens)
        df["effective_input_tokens"] = (
            df["input_tokens"] + df["cache_creation_input_tokens"] + (df["cache_read_input_tokens"] * 0.1)
        )
        df["total_tokens"] = df["effective_input_tokens"] + df["output_tokens"]

        # DEBUG: Query completed
        duration = (datetime.now() - start_time).total_seconds()
        log_with_context(
            logger,
            "DEBUG",
            "DuckDB query completed successfully",
            duration_seconds=duration,
            rows_loaded=len(df),
            unique_files=df["source_file"].nunique(),
        )

        return df
    except Exception as e:
        st.error(f"Error loading JSONL files: {e}")
        log_with_context(
            logger,
            "ERROR",
            "Failed to load JSONL files",
            error_type=type(e).__name__,
            error_message=str(e),
            path=str(CLAUDE_PROJECTS_PATH),
            glob_pattern=glob_pattern,
        )
        return None
    finally:
        conn.close()


def show_metrics(df):
    """Display basic metrics"""
    # First row: basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "AI Responses",
            len(df),
            help="Total number of AI assistant responses recorded across all projects and sessions.",
        )
    with col2:
        st.metric(
            "Sessions",
            df["session_id"].nunique(),
            help="Number of unique ClaudeCode sessions. "
            "Each session represents a distinct conversation or work period.",
        )
    with col3:
        st.metric("Projects", df["project_path"].nunique(), help="Number of unique projects where ClaudeCode was used.")
    with col4:
        st.metric(
            "Models", df["model"].nunique(), help="Number of different Claude models used (e.g., Sonnet, Opus, Haiku)."
        )

    # Second row: token metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_input = df["effective_input_tokens"].sum()
        st.metric(
            "Total Input Tokens",
            f"{total_input:,.0f}",
            help="Total effective input tokens consumed. Cache read tokens are counted as 10% of regular tokens.",
        )
    with col2:
        total_output = df["output_tokens"].sum()
        st.metric(
            "Total Output Tokens", f"{total_output:,.0f}", help="Total tokens generated by AI assistant responses."
        )
    with col3:
        avg_tokens = df["total_tokens"].mean()
        st.metric(
            "Avg Tokens/Response",
            f"{avg_tokens:,.0f}",
            help="Average number of tokens (input + output) per AI response.",
        )
    with col4:
        # Calculate 24-hour activity
        # Get the timezone from the first timestamp
        tz = df["timestamp"].dt.tz if not df.empty else None
        last_24h = pd.Timestamp.now(tz=tz) - pd.Timedelta(hours=24)
        recent_count = len(df[df["timestamp"] > last_24h])
        st.metric("24h Activity", recent_count, help="Number of AI responses in the last 24 hours.")

    # Third row: cache metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cache_read = df["cache_read_input_tokens"].sum()
        st.metric(
            "Cache Read Tokens",
            f"{cache_read:,.0f}",
            help="Tokens read from cache. These cost only 10% of regular input tokens.",
        )
    with col2:
        cache_creation = df["cache_creation_input_tokens"].sum()
        st.metric(
            "Cache Creation Tokens",
            f"{cache_creation:,.0f}",
            help="Tokens stored in cache for future reuse. Charged at regular input rates.",
        )
    with col3:
        regular_input = df["input_tokens"].sum()
        st.metric("Regular Input Tokens", f"{regular_input:,.0f}", help="Standard input tokens charged at full price.")
    with col4:
        # Cache savings (cache reads cost 10% of regular input)
        cache_savings = (
            cache_read * 0.9 / (regular_input + cache_creation + cache_read) * 100
            if (regular_input + cache_creation + cache_read) > 0
            else 0
        )
        st.metric(
            "Cache Savings",
            f"{cache_savings:.1f}%",
            help="Percentage saved by using cached tokens (90% discount on cache reads).",
        )


def show_overall_graphs(df):
    """Display overall statistics graphs"""
    st.header("📈 Overall Statistics")
    st.caption("Time-based analysis and model distribution of AI assistant responses")

    col1, col2 = st.columns(2)

    with col1:
        timeline_data = df.groupby(pd.Grouper(key="timestamp", freq="1h")).size().reset_index(name="count")
        fig_timeline = px.line(
            timeline_data,
            x="timestamp",
            y="count",
            title="AI Responses by Hour",
            height=400,
        )
        fig_timeline.update_traces(hovertemplate="Time: %{x|%Y-%m-%d %H:%M}<br>AI Responses: %{y}<extra></extra>")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col2:
        model_counts = df["model"].value_counts()
        fig_model_pie = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title="AI Responses by Model",
            height=400,
        )
        fig_model_pie.update_traces(
            hovertemplate="Model: %{label}<br>AI Responses: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        st.plotly_chart(fig_model_pie, use_container_width=True)

    # 24-hour activity timeline
    st.subheader("📊 24-Hour Activity Timeline")
    st.caption("Recent activity pattern with 5-minute granularity")
    # Get the timezone from the dataframe
    tz = df["timestamp"].dt.tz if not df.empty else None
    last_24h = pd.Timestamp.now(tz=tz) - pd.Timedelta(hours=24)
    recent_df = df[df["timestamp"] > last_24h]

    if len(recent_df) > 0:
        # 5-minute intervals for more granular view
        minute_activity = recent_df.groupby(pd.Grouper(key="timestamp", freq="5min")).size().reset_index(name="count")
        fig_24h = px.line(
            minute_activity,
            x="timestamp",
            y="count",
            title="24-Hour Activity (AI Responses per 5 minutes)",
            height=300,
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


def show_session_analysis(df):
    """Display session analysis"""
    st.header("🎯 Session Analysis")
    st.caption("Analysis of ClaudeCode session duration and activity patterns")

    session_data = df.groupby("session_id").agg(
        {
            "timestamp": ["min", "max", "count"],
            "model": lambda x: x.mode()[0] if not x.empty else None,
            "project_path": "first",
        }
    )

    session_data.columns = ["start_time", "end_time", "ai_response_count", "primary_model", "project"]
    session_data["duration"] = (session_data["end_time"] - session_data["start_time"]).dt.total_seconds() / 60
    session_data = session_data.sort_values("start_time", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        duration_bins = [0, 5, 15, 30, 60, float("inf")]
        duration_labels = ["0-5 min", "5-15 min", "15-30 min", "30-60 min", "60+ min"]
        session_data["duration_category"] = pd.cut(session_data["duration"], bins=duration_bins, labels=duration_labels)

        duration_counts = session_data["duration_category"].value_counts()
        fig_duration = px.bar(
            x=duration_counts.index,
            y=duration_counts.values,
            title="Session Duration Distribution",
            labels={"x": "Session Duration", "y": "Number of Sessions"},
            height=400,
        )
        fig_duration.update_traces(hovertemplate="Duration: %{x}<br>Sessions: %{y}<extra></extra>")
        st.plotly_chart(fig_duration, use_container_width=True)

    with col2:
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
            height=400,
        )
        fig_messages.update_traces(hovertemplate="AI Responses: %{x}<br>Sessions: %{y}<extra></extra>")
        st.plotly_chart(fig_messages, use_container_width=True)


def show_model_analysis(df):
    """Display model-based analysis"""
    st.header("🤖 Model Analysis")
    st.caption("Model usage patterns across projects and time")

    col1, col2 = st.columns(2)

    with col1:
        # Get top 10 projects by AI response count
        project_totals = df["project_path"].value_counts().head(10)

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
            title="Model Usage by Project (Top 10 Projects)",
            height=400,
            category_orders={"project_path": project_totals.index.tolist()[::-1]},  # Reverse for correct display
        )
        fig_model_project.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_model_project, use_container_width=True)

    with col2:
        model_daily = df.groupby([pd.Grouper(key="timestamp", freq="D"), "model"]).size().reset_index(name="count")
        fig_model_daily = px.bar(
            model_daily,
            x="timestamp",
            y="count",
            color="model",
            title="Daily Model Usage",
            height=400,
        )
        st.plotly_chart(fig_model_daily, use_container_width=True)


def calculate_cost(row):
    """Calculate cost for a single row based on model and token usage"""
    model = row["model"]
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (row["input_tokens"] + row["cache_creation_input_tokens"]) * pricing["input"] / 1_000_000
    cache_cost = row["cache_read_input_tokens"] * pricing["cache_read"] / 1_000_000
    output_cost = row["output_tokens"] * pricing["output"] / 1_000_000

    return input_cost + cache_cost + output_cost


def show_cost_calculation_details(df):
    """Show cost calculation details in an expander"""
    with st.expander("Cost Calculation Details"):
        st.caption("Cost formula per response:")
        st.code("""
Input cost = (input_tokens + cache_creation_tokens) * model_input_price / 1,000,000
Cache cost = cache_read_tokens * model_cache_price / 1,000,000  
Output cost = output_tokens * model_output_price / 1,000,000
Total cost = Input cost + Cache cost + Output cost
        """)

        # Show token totals
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Regular Input", f"{df['input_tokens'].sum():,.0f}")
            st.metric("Total Cache Creation", f"{df['cache_creation_input_tokens'].sum():,.0f}")
        with col2:
            st.metric("Total Cache Read", f"{df['cache_read_input_tokens'].sum():,.0f}")
            st.metric("Total Output", f"{df['output_tokens'].sum():,.0f}")
        with col3:
            st.metric("Total Responses", f"{len(df):,}")
            st.metric("Unique Models", df["model"].nunique())


def show_token_and_cost_analysis(df):
    """Display token usage and cost analysis"""
    st.header("💰 Token Usage & Cost Analysis")
    st.caption("Cost calculation based on token usage and model pricing")

    # Calculate costs
    df["cost"] = df.apply(calculate_cost, axis=1)
    total_cost = df["cost"].sum()

    # Show data period
    date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} - {df['timestamp'].max().strftime('%Y-%m-%d')}"
    total_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    st.caption(f"Data Period: {date_range} ({total_days} days)")

    # Show calculation details
    show_cost_calculation_details(df)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Cost", f"${total_cost:.2f}", help="Total cost calculated based on token usage and model pricing."
        )
    with col2:
        daily_avg_cost = total_cost / max((df["timestamp"].max() - df["timestamp"].min()).days, 1)
        st.metric("Daily Avg Cost", f"${daily_avg_cost:.2f}", help="Average daily cost based on the data period.")
    with col3:
        cache_rate = (
            (df["cache_read_input_tokens"].sum() / df["effective_input_tokens"].sum() * 100)
            if df["effective_input_tokens"].sum() > 0
            else 0
        )
        st.metric(
            "Cache Hit Rate",
            f"{cache_rate:.1f}%",
            help="Percentage of input tokens served from cache (90% cheaper than regular input).",
        )
    with col4:
        avg_cost_per_response = total_cost / len(df) if len(df) > 0 else 0
        st.metric("Avg Cost/Response", f"${avg_cost_per_response:.4f}", help="Average cost per AI response.")

    # Cost breakdown by model
    col1, col2 = st.columns(2)

    with col1:
        model_costs = (
            df.groupby("model").agg({"cost": "sum", "effective_input_tokens": "sum", "output_tokens": "sum"}).round(2)
        )
        model_costs = model_costs.sort_values("cost", ascending=False)

        fig_cost_by_model = px.bar(
            model_costs,
            y=model_costs.index,
            x="cost",
            orientation="h",
            title="Cost by Model",
            labels={"cost": "Cost ($)", "index": "Model"},
            height=400,
        )
        fig_cost_by_model.update_traces(hovertemplate="Model: %{y}<br>Cost: $%{x:.2f}<extra></extra>")
        fig_cost_by_model.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cost_by_model, use_container_width=True)

    with col2:
        # Daily cost trend
        daily_costs = df.groupby(pd.Grouper(key="timestamp", freq="D"))["cost"].sum().reset_index()
        fig_daily_cost = px.line(
            daily_costs,
            x="timestamp",
            y="cost",
            title="Daily Cost Trend",
            labels={"cost": "Cost ($)", "timestamp": "Date"},
            height=400,
        )
        fig_daily_cost.update_traces(hovertemplate="Date: %{x|%Y-%m-%d}<br>Cost: $%{y:.2f}<extra></extra>")
        st.plotly_chart(fig_daily_cost, use_container_width=True)

    # Token usage by model
    st.subheader("Token Usage by Model")
    st.caption("Detailed token consumption breakdown by model type")

    token_summary = (
        df.groupby("model")
        .agg(
            {
                "input_tokens": "sum",
                "cache_creation_input_tokens": "sum",
                "cache_read_input_tokens": "sum",
                "output_tokens": "sum",
                "cost": "sum",
            }
        )
        .round(0)
    )

    # Add percentage columns
    total_input = token_summary[["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]].sum().sum()
    total_output = token_summary["output_tokens"].sum()

    token_summary["input_pct"] = (
        token_summary[["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]].sum(axis=1)
        / total_input
        * 100
    ).round(1)
    token_summary["output_pct"] = (token_summary["output_tokens"] / total_output * 100).round(1)
    token_summary["cost_pct"] = (token_summary["cost"] / total_cost * 100).round(1)

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

    token_summary = token_summary.rename(columns=display_cols)
    st.dataframe(
        token_summary.style.format(
            {
                "Input Tokens": "{:,.0f}",
                "Cache Creation": "{:,.0f}",
                "Cache Read": "{:,.0f}",
                "Output Tokens": "{:,.0f}",
                "Cost ($)": "${:.2f}",
                "Input %": "{:.1f}%",
                "Output %": "{:.1f}%",
                "Cost %": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )


def show_heatmap(df):
    """Display GitHub-style heatmap"""
    st.header("📅 Usage Frequency Heatmap")
    st.caption("GitHub-style visualization of daily AI assistant usage over the past year")

    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size().reset_index(name="count")

    end_date = daily_counts["date"].max()
    start_date = end_date - pd.Timedelta(days=364)

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
        height=300,
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


def show_recent_logs(df):
    """Display recent logs"""
    st.header("📋 Recent AI Responses")
    st.caption("Latest 20 AI assistant responses with message previews")

    recent_logs = df.nlargest(20, "timestamp")[["timestamp", "model", "session_id", "project_path", "message_content"]]

    message_preview_length = 100
    recent_logs["message_content"] = recent_logs["message_content"].apply(
        lambda x: str(x)[:message_preview_length] + "..." if x and len(str(x)) > message_preview_length else x
    )

    st.dataframe(recent_logs, use_container_width=True, height=400)


def show_project_insights(df):
    """Display project insights"""
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
        top_projects = project_stats.head(10)
        fig_top_projects = px.bar(
            x=top_projects["ai_response_count"],
            y=top_projects.index,
            orientation="h",
            title="Top 10 Projects (by AI Response Count)",
            labels={"x": "AI Response Count", "y": "Project"},
            height=400,
        )
        fig_top_projects.update_traces(hovertemplate="Project: %{y}<br>AI Responses: %{x}<extra></extra>")
        fig_top_projects.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_top_projects, use_container_width=True)

    with col2:
        recent_projects = project_stats.sort_values("last_use", ascending=False).head(10)
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
            height=400,
        )
        fig_recent.update_traces(
            hovertemplate="Project: %{y}<br>Days since last use: %{x}<br>AI Responses: %{marker.size}<extra></extra>"
        )
        fig_recent.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_recent, use_container_width=True)


def main():
    st.title("🔍 ccwatch - ClaudeCode Monitor")
    st.markdown("Monitor and visualize ClaudeCode logs")

    # アプリケーション起動ログ
    log_with_context(logger, "INFO", "ccwatch application started", claude_path=str(CLAUDE_PROJECTS_PATH))

    # Auto-refresh every 5 minutes (in milliseconds)
    count = st_autorefresh(interval=CHECK_INTERVAL * 1000, limit=None, key="autorefresh")

    # Initialize session state
    if "update_count" not in st.session_state:
        st.session_state["update_count"] = 0

    if count > 0:
        st.session_state["update_count"] = count

    # Get file list
    jsonl_files = get_jsonl_files()

    # Sidebar
    with st.sidebar:
        st.header("🔍 ccwatch")
        st.caption("ClaudeCode Monitor")

        st.write("📊 Monitoring Status")
        st.write("- Auto-refresh: Every 5 minutes")
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

    # Main content
    if not jsonl_files:
        st.info("Searching for ClaudeCode log files...")
        return

    cache_key = st.session_state["update_count"]

    # データ読み込み開始ログ
    log_with_context(logger, "INFO", "Starting data load", update_count=cache_key, files_found=len(jsonl_files))

    df = load_logs_with_duckdb(cache_key)

    if df is not None and not df.empty:
        # データ読み込み成功ログ
        log_with_context(
            logger,
            "INFO",
            "Data loaded successfully",
            rows=len(df),
            sessions=df["session_id"].nunique(),
            projects=df["project_path"].nunique(),
        )
        # Display metrics
        show_metrics(df)

        # Overall statistics
        show_overall_graphs(df)

        # Session analysis
        show_session_analysis(df)

        # Model analysis
        show_model_analysis(df)

        # Token and cost analysis
        show_token_and_cost_analysis(df)

        # Heatmap
        show_heatmap(df)

        # Project insights
        show_project_insights(df)

        # Recent logs
        show_recent_logs(df)

        # Footer information
        st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("Failed to load data")
        log_with_context(logger, "ERROR", "Failed to load data - DataFrame is None or empty")


if __name__ == "__main__":
    main()
