"""Tests for data loading module with DuckDB integration"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import extract_tool_uses_from_assistant_df


class TestExtractToolUsesFromAssistantDf:
    """Test tool usage extraction from assistant messages"""

    def test_extract_tool_uses_basic(self):
        """Test basic tool usage extraction"""
        # Create test data with tool usage
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                "cwd": "/test",
                "project_path": "project1",
                "message": {
                    "content": [
                        {"type": "text", "text": "Let me help you"},
                        {"type": "tool_use", "name": "Read", "id": "tool-1"},
                    ]
                },
            }
        ]
        df = pd.DataFrame(data)

        # Extract tool uses
        result = extract_tool_uses_from_assistant_df(df)

        # Verify extraction
        assert len(result) == 1
        assert result.iloc[0]["tool_name"] == "Read"
        assert result.iloc[0]["tool_id"] == "tool-1"
        assert result.iloc[0]["session_id"] == "session-1"

    def test_extract_tool_uses_multiple_tools(self):
        """Test extraction with multiple tools in one message"""
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                "cwd": "/test",
                "project_path": "project1",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "id": "tool-1"},
                        {"type": "tool_use", "name": "Write", "id": "tool-2"},
                        {"type": "text", "text": "Done"},
                    ]
                },
            }
        ]
        df = pd.DataFrame(data)

        result = extract_tool_uses_from_assistant_df(df)

        assert len(result) == 2
        assert list(result["tool_name"]) == ["Read", "Write"]
        assert list(result["tool_id"]) == ["tool-1", "tool-2"]

    def test_extract_tool_uses_no_tools(self):
        """Test extraction when no tools are used"""
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                "cwd": "/test",
                "project_path": "project1",
                "message": {
                    "content": [
                        {"type": "text", "text": "Just a text response"},
                    ]
                },
            }
        ]
        df = pd.DataFrame(data)

        result = extract_tool_uses_from_assistant_df(df)

        assert result.empty

    def test_extract_tool_uses_json_string_content(self):
        """Test extraction when content is a JSON string"""
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                "cwd": "/test",
                "project_path": "project1",
                "message": {
                    "content": json.dumps(
                        [
                            {"type": "tool_use", "name": "Bash", "id": "tool-1"},
                        ]
                    )
                },
            }
        ]
        df = pd.DataFrame(data)

        result = extract_tool_uses_from_assistant_df(df)

        assert len(result) == 1
        assert result.iloc[0]["tool_name"] == "Bash"

    def test_extract_tool_uses_malformed_json(self):
        """Test extraction with malformed JSON content"""
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                "cwd": "/test",
                "project_path": "project1",
                "message": {"content": "{invalid json}"},
            }
        ]
        df = pd.DataFrame(data)

        result = extract_tool_uses_from_assistant_df(df)

        # Should handle gracefully and return empty
        assert result.empty

    def test_extract_tool_uses_missing_fields(self):
        """Test extraction with missing optional fields"""
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                # cwd is missing
                "project_path": "project1",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Edit"},  # id is missing
                    ]
                },
            }
        ]
        df = pd.DataFrame(data)

        result = extract_tool_uses_from_assistant_df(df)

        assert len(result) == 1
        assert result.iloc[0]["tool_name"] == "Edit"
        assert result.iloc[0]["tool_id"] == ""  # Should default to empty string
        assert result.iloc[0]["cwd"] == ""  # Should default to empty string

    def test_extract_tool_uses_empty_dataframe(self):
        """Test extraction with empty DataFrame"""
        df = pd.DataFrame()

        result = extract_tool_uses_from_assistant_df(df)

        assert result.empty

    def test_extract_tool_uses_none_message(self):
        """Test extraction when message is None"""
        data = [
            {
                "source_file": "test.jsonl",
                "timestamp": "2024-01-01T00:00:00Z",
                "session_id": "session-1",
                "uuid": "uuid-1",
                "cwd": "/test",
                "project_path": "project1",
                "message": None,
            }
        ]
        df = pd.DataFrame(data)

        result = extract_tool_uses_from_assistant_df(df)

        assert result.empty


# DuckDB integration tests are excluded as they require complex setup
# and test DuckDB functionality rather than our code logic.
# The extract_tool_uses_from_assistant_df tests provide meaningful coverage.
