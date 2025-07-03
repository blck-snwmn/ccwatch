"""Tests for session analysis utilities"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.session_utils import calculate_session_duration, calculate_session_metrics, categorize_session_activity


class TestCalculateSessionDuration:
    """Test session duration calculation"""

    def test_empty_session(self):
        """Test calculation with empty session"""
        df = pd.DataFrame(columns=["timestamp"])
        total, active, idle_periods = calculate_session_duration(df)

        assert total == 0, "Empty session should have 0 total duration"
        assert active == 0, "Empty session should have 0 active duration"
        assert idle_periods == 0, "Empty session should have 0 idle periods"

    def test_single_message_session(self):
        """Test calculation with single message"""
        df = pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
        total, active, idle_periods = calculate_session_duration(df)

        assert total == 1, "Single message session should have 1 minute total duration"
        assert active == 1, "Single message session should have 1 minute active duration"
        assert idle_periods == 0, "Single message session should have 0 idle periods"

    def test_continuous_session_no_idle(self):
        """Test session with continuous activity (no idle periods)"""
        base_time = datetime.now(timezone.utc)
        timestamps = [base_time + timedelta(minutes=i * 5) for i in range(5)]  # 5-minute intervals
        df = pd.DataFrame({"timestamp": timestamps})

        total, active, idle_periods = calculate_session_duration(df, idle_threshold_minutes=30)

        expected_total = 20  # 0 to 20 minutes
        assert total == pytest.approx(expected_total, rel=0.01), f"Expected {expected_total} min total duration"
        assert active == pytest.approx(expected_total + 1, rel=0.01), (
            f"Expected {expected_total + 1} min active duration"
        )
        assert idle_periods == 0, "Should have no idle periods with 5-minute intervals"

    def test_session_with_idle_periods(self):
        """Test session with idle periods"""
        base_time = datetime.now(timezone.utc)
        # Create timestamps with a 45-minute gap in the middle
        timestamps = [
            base_time,
            base_time + timedelta(minutes=5),
            base_time + timedelta(minutes=10),
            base_time + timedelta(minutes=55),  # 45-minute gap
            base_time + timedelta(minutes=60),
        ]
        df = pd.DataFrame({"timestamp": timestamps})

        total, active, idle_periods = calculate_session_duration(df, idle_threshold_minutes=30)

        assert total == 60, "Total duration should be 60 minutes"
        assert active < total, "Active duration should be less than total when idle periods exist"
        assert idle_periods == 1, "Should detect 1 idle period (45-minute gap)"
        # Active should be: 5 + 5 + 5 + 1 (for message after idle) + 1 (first message) = 17
        assert active == pytest.approx(17, rel=0.01), "Active duration should exclude idle time"

    def test_multiple_idle_periods(self):
        """Test session with multiple idle periods"""
        base_time = datetime.now(timezone.utc)
        timestamps = [
            base_time,
            base_time + timedelta(minutes=5),
            base_time + timedelta(minutes=40),  # 35-minute gap
            base_time + timedelta(minutes=45),
            base_time + timedelta(minutes=90),  # 45-minute gap
        ]
        df = pd.DataFrame({"timestamp": timestamps})

        total, active, idle_periods = calculate_session_duration(df, idle_threshold_minutes=30)

        assert total == 90, "Total duration should be 90 minutes"
        assert idle_periods == 2, "Should detect 2 idle periods"
        # Active: 5 + 1 (after first idle) + 5 + 1 (after second idle) + 1 (first message) = 13
        assert active == pytest.approx(13, rel=0.01), "Active duration should exclude both idle periods"

    def test_custom_idle_threshold(self):
        """Test with custom idle threshold"""
        base_time = datetime.now(timezone.utc)
        timestamps = [
            base_time,
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=30),
        ]
        df = pd.DataFrame({"timestamp": timestamps})

        # With 10-minute threshold, both gaps should be idle
        total, active, idle_periods = calculate_session_duration(df, idle_threshold_minutes=10)
        assert idle_periods == 2, "Should detect 2 idle periods with 10-minute threshold"

        # With 20-minute threshold, no gaps should be idle
        total, active, idle_periods = calculate_session_duration(df, idle_threshold_minutes=20)
        assert idle_periods == 0, "Should detect 0 idle periods with 20-minute threshold"


class TestCategorizeSessionActivity:
    """Test session activity categorization"""

    def test_quick_query(self):
        """Test quick query categorization"""
        assert categorize_session_activity(3, 5) == "Quick Query"
        assert categorize_session_activity(5, 10) == "Quick Query"

    def test_focused_task(self):
        """Test focused task categorization"""
        assert categorize_session_activity(10, 10) == "Focused Task"
        assert categorize_session_activity(20, 15) == "Focused Task"

    def test_standard_session(self):
        """Test standard session categorization"""
        assert categorize_session_activity(15, 25) == "Standard Session"
        assert categorize_session_activity(20, 30) == "Standard Session"

    def test_intensive_work(self):
        """Test intensive work categorization"""
        assert categorize_session_activity(30, 20) == "Intensive Work"
        assert categorize_session_activity(50, 30) == "Intensive Work"

    def test_extended_session(self):
        """Test extended session categorization"""
        assert categorize_session_activity(40, 45) == "Extended Session"
        assert categorize_session_activity(50, 60) == "Extended Session"

    def test_high_intensity(self):
        """Test high intensity categorization"""
        assert categorize_session_activity(100, 45) == "High-Intensity"
        assert categorize_session_activity(75, 60) == "High-Intensity"

    def test_marathon_session(self):
        """Test marathon session categorization"""
        assert categorize_session_activity(100, 120) == "Marathon Session"
        assert categorize_session_activity(200, 180) == "Marathon Session"


class TestCalculateSessionMetrics:
    """Test comprehensive session metrics calculation"""

    def test_basic_session_metrics(self, sample_claude_logs):
        """Test basic session metrics calculation"""
        # Create DataFrame from sample logs
        data = []
        for log in sample_claude_logs:
            if log["type"] == "assistant":
                data.append(
                    {
                        "timestamp": pd.to_datetime(log["timestamp"]),
                        "session_id": log["sessionId"],
                        "model": log["message"]["model"],
                        "project_path": "test_project",
                        "total_tokens": (
                            log["message"]["usage"]["input_tokens"]
                            + log["message"]["usage"]["output_tokens"]
                            + log["message"]["usage"].get("cache_creation_input_tokens", 0)
                            + log["message"]["usage"].get("cache_read_input_tokens", 0)
                        ),
                    }
                )

        df = pd.DataFrame(data)

        # Calculate metrics
        metrics = calculate_session_metrics(df)

        assert len(metrics) == df["session_id"].nunique(), "Should have one row per session"
        assert "active_duration_minutes" in metrics.columns, "Should include active duration"
        assert "idle_periods" in metrics.columns, "Should include idle period count"
        assert "activity_type" in metrics.columns, "Should include activity type"
        assert all(metrics["messages_per_minute"] > 0), "Messages per minute should be positive"

    def test_session_with_cost_column(self):
        """Test session metrics with cost column present"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [base_time, base_time + timedelta(minutes=5)],
                "session_id": ["session-1", "session-1"],
                "model": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"],
                "project_path": ["project1", "project1"],
                "total_tokens": [1000, 1500],
                "cost": [0.01, 0.015],
            }
        )

        metrics = calculate_session_metrics(df)

        assert "total_cost" in metrics.columns, "Should include total cost when cost column present"
        assert metrics.iloc[0]["total_cost"] == 0.025, (
            f"Expected total cost 0.025 but got {metrics.iloc[0]['total_cost']}"
        )

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame(columns=["timestamp", "session_id", "model", "project_path", "total_tokens"])

        metrics = calculate_session_metrics(df)

        assert len(metrics) == 0, "Empty input should produce empty metrics"
        assert list(metrics.columns) == [
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
        ], "Should have all expected columns even when empty"

    def test_single_session_metrics(self):
        """Test metrics for a single session"""
        base_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "timestamp": [
                    base_time,
                    base_time + timedelta(minutes=5),
                    base_time + timedelta(minutes=10),
                    base_time + timedelta(minutes=50),  # 40-minute idle gap
                    base_time + timedelta(minutes=55),
                ],
                "session_id": ["sess-1"] * 5,
                "model": ["claude-3-5-sonnet-20241022"] * 5,
                "project_path": ["proj1"] * 5,
                "total_tokens": [100, 200, 150, 300, 250],
            }
        )

        metrics = calculate_session_metrics(df, idle_threshold_minutes=30)

        assert len(metrics) == 1, "Should have one row for single session"
        row = metrics.iloc[0]

        assert row["ai_response_count"] == 5, f"Expected 5 messages but got {row['ai_response_count']}"
        assert row["total_duration_minutes"] == 55, f"Expected 55 min total but got {row['total_duration_minutes']}"
        assert row["idle_periods"] == 1, f"Expected 1 idle period but got {row['idle_periods']}"
        assert row["total_tokens"] == 1000, f"Expected 1000 total tokens but got {row['total_tokens']}"
        assert row["idle_percentage"] > 50, f"Expected >50% idle but got {row['idle_percentage']:.1f}%"
