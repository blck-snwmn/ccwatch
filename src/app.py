import glob
import os
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ccwatch - ClaudeCode Monitor", layout="wide")

# Allow overriding the path via environment variable
DEFAULT_CLAUDE_PATH = Path.home() / ".claude" / "projects"
CLAUDE_PROJECTS_PATH = Path(os.getenv("CLAUDE_PROJECTS_PATH", str(DEFAULT_CLAUDE_PATH)))
JSONL_PATTERN = "**/*.jsonl"
ERROR_LOG_FILE = "error.log"
MAX_PROJECTS_TO_SHOW = 10
CHECK_INTERVAL = 5 * 60  # 5 minutes (in seconds)


def get_jsonl_files():
    """Search for ClaudeCode project log files"""
    if not CLAUDE_PROJECTS_PATH.exists():
        st.warning(f"ClaudeCode projects directory not found: {CLAUDE_PROJECTS_PATH}")
        return []

    pattern = str(CLAUDE_PROJECTS_PATH / JSONL_PATTERN)
    files = glob.glob(pattern, recursive=True)
    return sorted(files, key=os.path.getmtime, reverse=True)


@st.cache_data(ttl=3600)
def load_logs_with_duckdb(file_paths, cache_key):
    """Load JSONL files directly using DuckDB

    Args:
        file_paths: List of JSONL file paths to load
        cache_key: Cache control key (update counter)
    """
    _ = cache_key  # Used for cache control
    if not file_paths:
        return None

    conn = duckdb.connect(":memory:")
    queries = []

    for file_path in file_paths:
        try:
            check_query = f"SELECT * FROM read_json_auto('{file_path}', format='newline_delimited') WHERE type = 'assistant' LIMIT 1"
            first_row = conn.execute(check_query).fetchdf()

            if first_row.empty or "timestamp" not in first_row.columns:
                continue

        except Exception:
            continue

        query = f"""
        SELECT 
            '{file_path}' as source_file,
            timestamp,
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
        FROM read_json_auto('{file_path}', format='newline_delimited')
        WHERE type = 'assistant'
        """
        queries.append(query)

    if not queries:
        return None

    combined_query = " UNION ALL ".join(queries)

    try:
        df = conn.execute(combined_query).df()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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

        return df
    except Exception as e:
        st.error(f"Error loading JSONL files: {e}")
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(f"[{datetime.now()}] Error: {e}\n")
        return None
    finally:
        conn.close()


def show_metrics(df):
    """Display basic metrics"""
    # First row: basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AI Responses", len(df))
    with col2:
        st.metric("Sessions", df["session_id"].nunique())
    with col3:
        st.metric("Projects", df["project_path"].nunique())
    with col4:
        st.metric("Models", df["model"].nunique())

    # Second row: token metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_input = df["effective_input_tokens"].sum()
        st.metric("Total Input Tokens", f"{total_input:,.0f}")
    with col2:
        total_output = df["output_tokens"].sum()
        st.metric("Total Output Tokens", f"{total_output:,.0f}")
    with col3:
        avg_tokens = df["total_tokens"].mean()
        st.metric("Avg Tokens/Response", f"{avg_tokens:,.0f}")
    with col4:
        # Calculate 24-hour activity
        last_24h = datetime.now(tz=df["timestamp"].dt.tz) - pd.Timedelta(hours=24)
        recent_count = len(df[df["timestamp"] > last_24h])
        st.metric("24h Activity", recent_count)

    # Third row: cache metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cache_read = df["cache_read_input_tokens"].sum()
        st.metric("Cache Read Tokens", f"{cache_read:,.0f}")
    with col2:
        cache_creation = df["cache_creation_input_tokens"].sum()
        st.metric("Cache Creation Tokens", f"{cache_creation:,.0f}")
    with col3:
        regular_input = df["input_tokens"].sum()
        st.metric("Regular Input Tokens", f"{regular_input:,.0f}")
    with col4:
        # Cache savings (cache reads cost 10% of regular input)
        cache_savings = (
            cache_read * 0.9 / (regular_input + cache_creation + cache_read) * 100
            if (regular_input + cache_creation + cache_read) > 0
            else 0
        )
        st.metric("Cache Savings", f"{cache_savings:.1f}%")


def show_overall_graphs(df):
    """Display overall statistics graphs"""
    st.header("📈 Overall Statistics")

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
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col2:
        model_counts = df["model"].value_counts()
        fig_model_pie = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title="AI Responses by Model",
            height=400,
        )
        st.plotly_chart(fig_model_pie, use_container_width=True)


def show_session_analysis(df):
    """Display session analysis"""
    st.header("🎯 Session Analysis")

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
        st.plotly_chart(fig_messages, use_container_width=True)


def show_model_analysis(df):
    """Display model-based analysis"""
    st.header("🤖 Model Analysis")

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


def show_heatmap(df):
    """Display GitHub-style heatmap"""
    st.header("📅 Usage Frequency Heatmap")

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
        st.metric("Total Days", len(daily_counts))
    with col2:
        st.metric("Avg AI Responses/Day", f"{daily_counts['count'].mean():.1f}")
    with col3:
        st.metric("Max AI Responses/Day", daily_counts["count"].max())
    with col4:
        active_days = (daily_counts["count"] > 0).sum()
        st.metric("Active Days", f"{active_days} ({active_days / len(daily_counts) * 100:.1f}%)")


def show_recent_logs(df):
    """Display recent logs"""
    st.header("📋 Recent AI Responses")

    recent_logs = df.nlargest(20, "timestamp")[["timestamp", "model", "session_id", "project_path", "message_content"]]

    message_preview_length = 100
    recent_logs["message_content"] = recent_logs["message_content"].apply(
        lambda x: str(x)[:message_preview_length] + "..." if x and len(str(x)) > message_preview_length else x
    )

    st.dataframe(recent_logs, use_container_width=True, height=400)


def show_project_insights(df):
    """Display project insights"""
    st.header("💼 Project Insights")

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
        fig_top_projects.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_top_projects, use_container_width=True)

    with col2:
        recent_projects = project_stats.sort_values("last_use", ascending=False).head(10)
        recent_projects["days_since_last_use"] = (
            datetime.now(tz=recent_projects["last_use"].dt.tz) - recent_projects["last_use"]
        ).dt.days

        fig_recent = px.scatter(
            recent_projects,
            x="days_since_last_use",
            y=recent_projects.index,
            size="ai_response_count",
            title="Recently Used Projects",
            labels={"x": "Days Since Last Use", "y": "Project"},
            height=400,
        )
        fig_recent.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_recent, use_container_width=True)


def main():
    st.title("🔍 ccwatch - ClaudeCode Monitor")
    st.markdown("Monitor and visualize ClaudeCode logs")

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
    df = load_logs_with_duckdb(jsonl_files, cache_key)

    if df is not None and not df.empty:
        # Display metrics
        show_metrics(df)

        # Overall statistics
        show_overall_graphs(df)

        # Session analysis
        show_session_analysis(df)

        # Model analysis
        show_model_analysis(df)

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


if __name__ == "__main__":
    main()
