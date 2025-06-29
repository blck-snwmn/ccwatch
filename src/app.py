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

CLAUDE_PROJECTS_PATH = Path.home() / ".claude" / "projects"
JSONL_PATTERN = "**/*.jsonl"
ERROR_LOG_FILE = "error.log"
MAX_PROJECTS_TO_SHOW = 10
CHECK_INTERVAL = 5 * 60  # 5分(秒)


def get_jsonl_files():
    """ClaudeCodeのプロジェクトログファイルを検索"""
    if not CLAUDE_PROJECTS_PATH.exists():
        st.warning(f"ClaudeCode projects directory not found: {CLAUDE_PROJECTS_PATH}")
        return []

    pattern = str(CLAUDE_PROJECTS_PATH / JSONL_PATTERN)
    files = glob.glob(pattern, recursive=True)
    return sorted(files, key=os.path.getmtime, reverse=True)


@st.cache_data(ttl=3600)
def load_logs_with_duckdb(file_paths, cache_key):
    """DuckDBを使用してJSONLファイルを直接読み込む

    Args:
        file_paths: 読み込むJSONLファイルのパスリスト
        cache_key: キャッシュ制御用のキー(更新カウンタ)
    """
    _ = cache_key  # キャッシュ制御のために使用
    if not file_paths:
        return None

    conn = duckdb.connect(":memory:")
    queries = []

    for file_path in file_paths:
        try:
            check_query = f"SELECT * FROM read_json_auto('{file_path}', format='newline_delimited') LIMIT 1"
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
            CASE 
                WHEN type = 'assistant' AND message IS NOT NULL 
                THEN TRY_CAST(json_extract_string(to_json(message), '$.model') AS VARCHAR)
                ELSE NULL
            END as model,
            sessionId as session_id,
            uuid,
            parentUuid as parent_uuid,
            cwd,
            userType as user_type
        FROM read_json_auto('{file_path}', format='newline_delimited')
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
        return df
    except Exception as e:
        st.error(f"Error loading JSONL files: {e}")
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(f"[{datetime.now()}] Error: {e}\n")
        return None
    finally:
        conn.close()


def show_metrics(df):
    """基本的なメトリクスを表示"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総ログ数", len(df))
    with col2:
        st.metric("セッション数", df["session_id"].nunique())
    with col3:
        st.metric("プロジェクト数", df["project_path"].nunique())
    with col4:
        st.metric("モデル数", df["model"].nunique())


def show_overall_graphs(df):
    """全体グラフを表示"""
    st.header("📈 全体統計")

    col1, col2 = st.columns(2)

    with col1:
        timeline_data = df.groupby(pd.Grouper(key="timestamp", freq="1h")).size().reset_index(name="count")
        fig_timeline = px.line(
            timeline_data,
            x="timestamp",
            y="count",
            title="時間別メッセージ数",
            height=400,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col2:
        model_counts = df[df["model"].notna()]["model"].value_counts()
        fig_model_pie = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title="モデル別メッセージ数",
            height=400,
        )
        st.plotly_chart(fig_model_pie, use_container_width=True)


def show_session_analysis(df):
    """セッション分析を表示"""
    st.header("🎯 セッション分析")

    session_data = df.groupby("session_id").agg(
        {
            "timestamp": ["min", "max", "count"],
            "model": lambda x: x.mode()[0] if not x.empty and not x.isna().all() else None,
            "project_path": "first",
        }
    )

    session_data.columns = ["start_time", "end_time", "message_count", "primary_model", "project"]
    session_data["duration"] = (session_data["end_time"] - session_data["start_time"]).dt.total_seconds() / 60
    session_data = session_data.sort_values("start_time", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        duration_bins = [0, 5, 15, 30, 60, float("inf")]
        duration_labels = ["0-5分", "5-15分", "15-30分", "30-60分", "60分以上"]
        session_data["duration_category"] = pd.cut(session_data["duration"], bins=duration_bins, labels=duration_labels)

        duration_counts = session_data["duration_category"].value_counts()
        fig_duration = px.bar(
            x=duration_counts.index,
            y=duration_counts.values,
            title="セッション時間分布",
            labels={"x": "セッション時間", "y": "セッション数"},
            height=400,
        )
        st.plotly_chart(fig_duration, use_container_width=True)

    with col2:
        message_bins = [0, 10, 50, 100, 200, float("inf")]
        message_labels = ["1-10", "11-50", "51-100", "101-200", "200+"]
        session_data["message_category"] = pd.cut(
            session_data["message_count"], bins=message_bins, labels=message_labels
        )

        message_counts = session_data["message_category"].value_counts()
        fig_messages = px.bar(
            x=message_counts.index,
            y=message_counts.values,
            title="セッションメッセージ数分布",
            labels={"x": "メッセージ数", "y": "セッション数"},
            height=400,
        )
        st.plotly_chart(fig_messages, use_container_width=True)


def show_model_analysis(df):
    """モデル別分析を表示"""
    st.header("🤖 モデル別分析")

    col1, col2 = st.columns(2)

    with col1:
        model_project_data = df[df["model"].notna()].groupby(["project_path", "model"]).size().reset_index(name="count")

        top_projects = df["project_path"].value_counts().head(10).index
        model_project_filtered = model_project_data[model_project_data["project_path"].isin(top_projects)]

        fig_model_project = px.bar(
            model_project_filtered,
            x="count",
            y="project_path",
            color="model",
            orientation="h",
            title="モデル別プロジェクト使用状況(上位10プロジェクト)",
            height=400,
        )
        st.plotly_chart(fig_model_project, use_container_width=True)

    with col2:
        model_daily = (
            df[df["model"].notna()]
            .groupby([pd.Grouper(key="timestamp", freq="D"), "model"])
            .size()
            .reset_index(name="count")
        )
        fig_model_daily = px.bar(
            model_daily,
            x="timestamp",
            y="count",
            color="model",
            title="モデル別日次使用数",
            height=400,
        )
        st.plotly_chart(fig_model_daily, use_container_width=True)


def show_heatmap(df):
    """GitHubスタイルのヒートマップを表示"""
    st.header("📅 使用頻度ヒートマップ")

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
            hovertemplate="Week %{x}<br>%{y}<br>Messages: %{z}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title="過去52週間の使用頻度",
        xaxis_title="週",
        yaxis_title="曜日",
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
        st.metric("合計日数", len(daily_counts))
    with col2:
        st.metric("平均メッセージ数/日", f"{daily_counts['count'].mean():.1f}")
    with col3:
        st.metric("最大メッセージ数/日", daily_counts["count"].max())
    with col4:
        active_days = (daily_counts["count"] > 0).sum()
        st.metric("アクティブ日数", f"{active_days} ({active_days / len(daily_counts) * 100:.1f}%)")


def show_recent_logs(df):
    """最新のログを表示"""
    st.header("📋 最新のメッセージ")

    recent_logs = df.nlargest(20, "timestamp")[["timestamp", "model", "session_id", "project_path", "message_content"]]

    message_preview_length = 100
    recent_logs["message_content"] = recent_logs["message_content"].apply(
        lambda x: str(x)[:message_preview_length] + "..." if x and len(str(x)) > message_preview_length else x
    )

    st.dataframe(recent_logs, use_container_width=True, height=400)


def show_project_insights(df):
    """プロジェクト別インサイトを表示"""
    st.header("💼 プロジェクト別インサイト")

    project_stats = df.groupby("project_path").agg(
        {
            "timestamp": ["min", "max", "count"],
            "session_id": "nunique",
            "model": lambda x: x.value_counts().to_dict() if not x.empty else {},
        }
    )

    project_stats.columns = ["first_use", "last_use", "message_count", "session_count", "model_usage"]
    project_stats["days_active"] = (project_stats["last_use"] - project_stats["first_use"]).dt.days + 1
    project_stats = project_stats.sort_values("message_count", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        top_projects = project_stats.head(10)
        fig_top_projects = px.bar(
            x=top_projects["message_count"],
            y=top_projects.index,
            orientation="h",
            title="上位10プロジェクト(メッセージ数)",
            labels={"x": "メッセージ数", "y": "プロジェクト"},
            height=400,
        )
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
            size="message_count",
            title="最近使用したプロジェクト",
            labels={"x": "最終使用からの日数", "y": "プロジェクト"},
            height=400,
        )
        st.plotly_chart(fig_recent, use_container_width=True)


def main():
    st.title("🔍 ccwatch - ClaudeCode Monitor")
    st.markdown("ClaudeCodeのログを監視・可視化")

    # 5分ごとに自動更新(ミリ秒単位)
    count = st_autorefresh(interval=CHECK_INTERVAL * 1000, limit=None, key="autorefresh")

    # セッション状態の初期化
    if "update_count" not in st.session_state:
        st.session_state["update_count"] = 0

    if count > 0:
        st.session_state["update_count"] = count

    # ファイルリストを取得
    jsonl_files = get_jsonl_files()

    # サイドバー
    with st.sidebar:
        st.header("🔍 ccwatch")
        st.caption("ClaudeCode Monitor")

        st.write("📊 監視状態")
        st.write("- 自動更新: 5分ごと")
        st.write(f"- 更新回数: {st.session_state['update_count']}")

        st.divider()
        if st.button("🔄 手動更新", use_container_width=True):
            st.rerun()

        # ファイル情報
        if jsonl_files:
            st.divider()
            st.write(f"📁 検出ファイル数: {len(jsonl_files)}")

            # 最新5ファイル
            with st.expander("最新のファイル"):
                for f in jsonl_files[:5]:
                    file_name = Path(f).name
                    mtime = datetime.fromtimestamp(os.path.getmtime(f))
                    st.caption(f"- {file_name}")
                    st.caption(f"  更新: {mtime.strftime('%H:%M:%S')}")

            st.divider()
            st.subheader("📂 プロジェクトフィルタ")

            # プロジェクトリストを取得
            project_names = list(set(Path(f).parent.name for f in jsonl_files))
            project_names.sort()

            selected_projects = st.multiselect(
                "プロジェクトを選択",
                options=project_names,
                default=None,
                help="特定のプロジェクトのみ表示する場合に選択",
            )

    # メインコンテンツ
    if not jsonl_files:
        st.info("ClaudeCodeのログファイルを検索中...")
        return

    # 選択されたプロジェクトのファイルのみをフィルタ
    if selected_projects:
        filtered_files = [f for f in jsonl_files if Path(f).parent.name in selected_projects]
    else:
        filtered_files = jsonl_files

    cache_key = st.session_state["update_count"]
    df = load_logs_with_duckdb(filtered_files, cache_key)

    if df is not None and not df.empty:
        # メトリクスを表示
        show_metrics(df)

        # 全体統計
        show_overall_graphs(df)

        # セッション分析
        show_session_analysis(df)

        # モデル分析
        show_model_analysis(df)

        # ヒートマップ
        show_heatmap(df)

        # プロジェクト別インサイト
        show_project_insights(df)

        # 最新ログ
        show_recent_logs(df)

        # フッター情報
        st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("データの読み込みに失敗しました")


if __name__ == "__main__":
    main()
