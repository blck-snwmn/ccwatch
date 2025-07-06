"""Tests for data processing module"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.processors import (
    aggregate_by_time,
    filter_by_date_range,
    process_assistant_dataframe,
    process_system_dataframe,
    process_tool_dataframe,
    truncate_message,
)


class TestTruncateMessage:
    """Test message truncation functionality"""

    def test_truncate_short_message(self):
        """Test that short messages are not truncated"""
        message = "Short message"
        result = truncate_message(message, max_length=20)
        assert result == "Short message"

    def test_truncate_long_message(self):
        """Test that long messages are truncated with ellipsis"""
        message = "This is a very long message that should be truncated"
        result = truncate_message(message, max_length=20)
        assert result == "This is a very long ..."
        assert len(result) == 23  # 20 chars + "..."

    def test_truncate_exact_length(self):
        """Test message exactly at max length"""
        message = "Exactly twenty chars"
        result = truncate_message(message, max_length=20)
        assert result == "Exactly twenty chars"

    def test_truncate_none_message(self):
        """Test handling of None input"""
        result = truncate_message(None)
        assert result == ""

    def test_truncate_empty_string(self):
        """Test handling of empty string"""
        result = truncate_message("")
        assert result == ""

    def test_truncate_with_default_length(self):
        """Test truncation with default length (200)"""
        message = "a" * 250
        result = truncate_message(message)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (123, "123"),  # Integer
            (12.34, "12.34"),  # Float
            (True, "True"),  # Boolean
            (["a", "b"], "['a', 'b']"),  # List
        ],
    )
    def test_truncate_non_string_types(self, input_val, expected):
        """Test truncation handles non-string types"""
        result = truncate_message(input_val, max_length=50)
        assert result == expected


class TestProcessAssistantDataframe:
    """Test assistant dataframe processing"""

    def test_process_empty_dataframe(self):
        """Test processing empty dataframe"""
        df = pd.DataFrame()
        result = process_assistant_dataframe(df)
        assert result.empty

    def test_process_basic_dataframe(self):
        """Test basic dataframe processing"""
        data = {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "source_file": ["/path/to/project/file.jsonl"],
            "session_id": [123],  # Numeric, should be converted to string
            "input_tokens": [100],
            "cache_creation_input_tokens": [50],
            "cache_read_input_tokens": [200],
            "output_tokens": [300],
        }
        df = pd.DataFrame(data)

        result = process_assistant_dataframe(df)

        # Check timestamp conversion
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

        # Check project path extraction
        assert "project_path" in result.columns
        assert result["project_path"].iloc[0] == "project"

        # Check session_id conversion
        assert result["session_id"].dtype == "object"  # String type
        assert result["session_id"].iloc[0] == "123"

        # Check token calculations
        assert "effective_input_tokens" in result.columns
        assert result["effective_input_tokens"].iloc[0] == 100 + 50 + (200 * 0.1)  # 170
        assert "total_tokens" in result.columns
        assert result["total_tokens"].iloc[0] == 170 + 300  # 470

    def test_process_with_nan_tokens(self):
        """Test processing with NaN token values"""
        data = {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "input_tokens": [100],
            "cache_creation_input_tokens": [None],  # NaN
            "cache_read_input_tokens": [float("nan")],  # NaN
            "output_tokens": [300],
        }
        df = pd.DataFrame(data)

        result = process_assistant_dataframe(df)

        # NaN should be filled with 0
        assert result["cache_creation_input_tokens"].iloc[0] == 0
        assert result["cache_read_input_tokens"].iloc[0] == 0
        assert result["effective_input_tokens"].iloc[0] == 100  # 100 + 0 + 0

    def test_process_without_token_columns(self):
        """Test processing when token columns are missing"""
        data = {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "message": ["Test message"],
        }
        df = pd.DataFrame(data)

        result = process_assistant_dataframe(df)

        # Should not crash, and no token calculations
        assert "effective_input_tokens" not in result.columns
        assert "total_tokens" not in result.columns

    def test_process_partial_token_columns(self):
        """Test when only some token columns exist"""
        data = {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "input_tokens": [100],
            "output_tokens": [200],
            # Missing cache columns
        }
        df = pd.DataFrame(data)

        result = process_assistant_dataframe(df)

        # Should not calculate effective tokens without all required columns
        assert "effective_input_tokens" not in result.columns


class TestProcessSystemDataframe:
    """Test system dataframe processing"""

    def test_process_empty_system_df(self):
        """Test processing empty system dataframe"""
        df = pd.DataFrame()
        result = process_system_dataframe(df)
        assert result.empty

    def test_process_system_df_basic(self):
        """Test basic system dataframe processing"""
        data = {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "source_file": ["/path/to/project/error.jsonl"],
            "session_id": [456],
            "error": ["Test error message"],
        }
        df = pd.DataFrame(data)

        result = process_system_dataframe(df)

        # Check conversions
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        assert result["project_path"].iloc[0] == "project"
        assert result["session_id"].iloc[0] == "456"


class TestProcessToolDataframe:
    """Test tool dataframe processing"""

    def test_process_empty_tool_df(self):
        """Test processing empty tool dataframe"""
        df = pd.DataFrame()
        result = process_tool_dataframe(df)
        assert result.empty

    def test_process_tool_df_basic(self):
        """Test basic tool dataframe processing"""
        data = {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "source_file": ["/path/to/project/tools.jsonl"],
            "session_id": ["session-1"],
            "tool_name": ["Read"],
        }
        df = pd.DataFrame(data)

        result = process_tool_dataframe(df)

        # Check conversions
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        # Note: process_tool_dataframe doesn't add project_path


class TestFilterByDateRange:
    """Test date range filtering"""

    def test_filter_empty_dataframe(self):
        """Test filtering empty dataframe"""
        df = pd.DataFrame()
        result = filter_by_date_range(df, "2024-01-01", "2024-12-31")
        assert result.empty

    def test_filter_with_dates(self):
        """Test filtering with start and end dates"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        }
        df = pd.DataFrame(data)

        result = filter_by_date_range(df, "2024-01-02", "2024-01-04")

        assert len(result) == 3  # Jan 2, 3, 4
        assert result["value"].tolist() == [1, 2, 3]

    def test_filter_no_dates(self):
        """Test when no date filter is applied"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        }
        df = pd.DataFrame(data)

        result = filter_by_date_range(df, None, None)

        assert len(result) == 5  # All records

    def test_filter_start_date_only(self):
        """Test filtering with only start date"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        }
        df = pd.DataFrame(data)

        result = filter_by_date_range(df, "2024-01-03", None)

        assert len(result) == 3  # Jan 3, 4, 5
        assert result["value"].tolist() == [2, 3, 4]

    def test_filter_end_date_only(self):
        """Test filtering with only end date"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        }
        df = pd.DataFrame(data)

        result = filter_by_date_range(df, None, "2024-01-03")

        assert len(result) == 3  # Jan 1, 2, 3
        assert result["value"].tolist() == [0, 1, 2]

    def test_filter_inclusive_boundaries(self):
        """Test that date boundaries are inclusive"""
        data = {
            "timestamp": [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 23:59:59"),
                pd.Timestamp("2024-01-02 00:00:00"),
            ],
            "value": [1, 2, 3],
        }
        df = pd.DataFrame(data)

        # Convert strings to timestamps for proper comparison
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-01")
        result = filter_by_date_range(df, start, end)

        # Since end is "2024-01-01 00:00:00", only the first record matches
        assert len(result) == 1
        assert result["value"].tolist() == [1]


class TestAggregateByTime:
    """Test time-based aggregation"""

    def test_aggregate_empty_dataframe(self):
        """Test aggregating empty dataframe"""
        df = pd.DataFrame()
        result = aggregate_by_time(df, "D")
        assert result.empty

    def test_aggregate_daily(self):
        """Test daily aggregation"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
            "cost": [1.0] * 48,  # 1.0 per hour for 2 days
        }
        df = pd.DataFrame(data)

        result = aggregate_by_time(df, "D")

        assert len(result) == 2  # 2 days
        assert "count" in result.columns
        assert result["count"].tolist() == [24, 24]  # 24 records per day

    def test_aggregate_weekly(self):
        """Test weekly aggregation"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=14, freq="D"),
            "cost": [10.0] * 14,  # 10.0 per day for 2 weeks
        }
        df = pd.DataFrame(data)

        result = aggregate_by_time(df, "W")

        assert len(result) == 2  # 2 weeks
        assert "count" in result.columns
        assert result["count"].sum() == 14  # Total 14 days

    def test_aggregate_with_multiple_columns(self):
        """Test aggregation returns count only"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "cost": [1.0, 2.0, 3.0, 4.0],
            "tokens": [100, 200, 300, 400],
        }
        df = pd.DataFrame(data)

        result = aggregate_by_time(df, "W")

        # aggregate_by_time only returns count, not sum of other columns
        assert "count" in result.columns
        assert "cost" not in result.columns
        assert "tokens" not in result.columns
        assert result["count"].sum() == 4  # 4 records total

    @pytest.mark.parametrize("freq", ["h", "D", "W", "ME"])
    def test_aggregate_different_frequencies(self, freq):
        """Test different aggregation frequencies"""
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "value": range(100),
        }
        df = pd.DataFrame(data)

        result = aggregate_by_time(df, freq)

        # Should not crash and return valid dataframe
        assert not result.empty
        assert "count" in result.columns
