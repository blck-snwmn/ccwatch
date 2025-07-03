"""Session analysis utilities"""

import pandas as pd


def calculate_session_duration(session_df, idle_threshold_minutes=30):
    """Calculate actual session duration excluding idle periods

    Args:
        session_df: DataFrame for a single session with timestamp column
        idle_threshold_minutes: Minutes of inactivity to consider as idle time

    Returns:
        tuple: (total_duration_minutes, active_duration_minutes, idle_periods_count)
    """
    if len(session_df) == 0:
        return 0, 0, 0

    if len(session_df) == 1:
        # Single message session - consider as 1 minute active time
        return 1, 1, 0

    # Sort by timestamp
    session_df = session_df.sort_values("timestamp")
    timestamps = session_df["timestamp"].tolist()

    # Calculate total duration
    total_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60

    # Calculate active duration by excluding idle periods
    active_duration = 0
    idle_periods = 0

    for i in range(1, len(timestamps)):
        time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60

        if time_diff <= idle_threshold_minutes:
            # Active period - add to active duration
            active_duration += time_diff
        else:
            # Idle period detected
            idle_periods += 1
            # Add a minimal active time (1 minute) for the message after idle
            active_duration += 1

    # Add 1 minute for the first message
    active_duration += 1

    return total_duration, active_duration, idle_periods


def categorize_session_activity(ai_response_count, active_duration_minutes):
    """Categorize session based on activity level

    Args:
        ai_response_count: Number of AI responses in the session
        active_duration_minutes: Active duration in minutes

    Returns:
        str: Activity category
    """
    if ai_response_count <= 5:
        return "Quick Query"
    elif ai_response_count <= 20:
        if active_duration_minutes <= 15:
            return "Focused Task"
        else:
            return "Standard Session"
    elif ai_response_count <= 50:
        if active_duration_minutes <= 30:
            return "Intensive Work"
        else:
            return "Extended Session"
    elif active_duration_minutes <= 60:
        return "High-Intensity"
    else:
        return "Marathon Session"


def calculate_session_metrics(df, idle_threshold_minutes=30):
    """Calculate comprehensive session metrics

    Args:
        df: DataFrame with session data
        idle_threshold_minutes: Minutes of inactivity to consider as idle

    Returns:
        DataFrame: Session metrics
    """
    # Handle empty DataFrame
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "session_id",
                "start_time",
                "end_time",
                "ai_response_count",
                "primary_model",
                "project",
                "total_duration_minutes",
                "active_duration_minutes",
                "idle_periods",
                "idle_percentage",
                "total_tokens",
                "total_cost",
                "messages_per_minute",
                "activity_type",
            ]
        )

    sessions = []

    for session_id, session_df in df.groupby("session_id"):
        # Basic stats
        start_time = session_df["timestamp"].min()
        end_time = session_df["timestamp"].max()
        ai_response_count = len(session_df)

        # Get most used model
        primary_model = session_df["model"].mode()[0] if not session_df["model"].empty else None

        # Get project
        project = session_df["project_path"].iloc[0]

        # Calculate durations
        total_duration, active_duration, idle_periods = calculate_session_duration(session_df, idle_threshold_minutes)

        # Calculate tokens and cost
        total_tokens = session_df["total_tokens"].sum()
        total_cost = session_df["cost"].sum() if "cost" in session_df.columns else 0

        # Messages per active minute
        messages_per_minute = ai_response_count / max(active_duration, 1)

        # Activity category
        activity_type = categorize_session_activity(ai_response_count, active_duration)

        sessions.append(
            {
                "session_id": session_id,
                "start_time": start_time,
                "end_time": end_time,
                "ai_response_count": ai_response_count,
                "primary_model": primary_model,
                "project": project,
                "total_duration_minutes": total_duration,
                "active_duration_minutes": active_duration,
                "idle_periods": idle_periods,
                "idle_percentage": (total_duration - active_duration) / max(total_duration, 1) * 100,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "messages_per_minute": messages_per_minute,
                "activity_type": activity_type,
            }
        )

    return pd.DataFrame(sessions)
