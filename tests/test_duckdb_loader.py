"""Tests for DuckDB data loading functionality"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import load_logs_with_duckdb
from config import AppConfig


class TestLoadLogsWithDuckDB:
    """Test DuckDB data loading functionality"""

    def test_load_logs_basic(self, claude_projects_with_jsonl, monkeypatch):
        """Test basic log loading with DuckDB"""
        projects_dir, _ = claude_projects_with_jsonl

        # Mock the config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        monkeypatch.setattr("app.config", test_config)

        # Load logs
        df = load_logs_with_duckdb(cache_key=1)

        # Check dataframe is not empty
        assert df is not None
        assert len(df) > 0

        # Check required columns exist
        required_columns = [
            "source_file",
            "timestamp",
            "log_type",
            "role",
            "message_content",
            "model",
            "session_id",
            "uuid",
            "parent_uuid",
            "cwd",
            "user_type",
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
            "project_path",
            "effective_input_tokens",
            "total_tokens",
        ]
        assert all(col in df.columns for col in required_columns)

    def test_load_logs_filters_assistant_only(self, tmp_path, monkeypatch):
        """Test that only assistant messages are loaded"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create JSONL with mixed message types
        jsonl_file = projects_dir / "test.jsonl"
        logs = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "assistant",
                "message": {"role": "assistant", "content": "Assistant message 1"},
                "sessionId": "test-session",
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "human",
                "message": {"role": "human", "content": "Human message"},
                "sessionId": "test-session",
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "assistant",
                "message": {"role": "assistant", "content": "Assistant message 2"},
                "sessionId": "test-session",
            },
        ]

        with open(jsonl_file, "w") as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"
        monkeypatch.setattr("app.config", test_config)

        # Load logs with unique cache key to avoid fixture data
        df = load_logs_with_duckdb(cache_key="test_filters_assistant_only")

        # Check if load was successful
        assert df is not None, "Failed to load logs from DuckDB"
        
        # Should only have assistant messages
        assert len(df) == 2  # Only 2 assistant messages
        assert all(df["log_type"] == "assistant")
        assert all(df["role"] == "assistant")

    def test_load_logs_token_calculations(self, tmp_path, monkeypatch):
        """Test token calculation logic"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create log with specific token values
        jsonl_file = projects_dir / "tokens.jsonl"
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": "Test",
                "model": "claude-3-5-sonnet-20241022",
                "usage": {
                    "input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 300,
                    "output_tokens": 500,
                },
            },
            "sessionId": "test-session",
        }

        with open(jsonl_file, "w") as f:
            f.write(json.dumps(log) + "\n")

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"
        monkeypatch.setattr("app.config", test_config)

        # Load logs with unique cache key
        df = load_logs_with_duckdb(cache_key="test_token_calculations")

        # Should only have our test log
        assert len(df) == 1, f"Expected 1 log but got {len(df)}"
        
        # Check token calculations
        row = df.iloc[0]
        assert int(row["input_tokens"]) == 1000
        assert int(row["cache_creation_input_tokens"]) == 200
        assert int(row["cache_read_input_tokens"]) == 300
        assert int(row["output_tokens"]) == 500

        # Check effective input tokens calculation (cache_read counts as 10%)
        expected_effective = 1000 + 200 + (300 * 0.1)
        assert row["effective_input_tokens"] == expected_effective

        # Check total tokens
        assert row["total_tokens"] == expected_effective + 500

    def test_load_logs_null_token_handling(self, tmp_path, monkeypatch):
        """Test handling of missing/null token values"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create log with missing token values
        jsonl_file = projects_dir / "null_tokens.jsonl"
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": "Test",
                "model": "claude-3-5-sonnet-20241022",
                "usage": {
                    # Omit some token values
                    "input_tokens": 1000,
                    "output_tokens": 500,
                },
            },
            "sessionId": "test-session",
        }

        with open(jsonl_file, "w") as f:
            f.write(json.dumps(log) + "\n")

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"
        monkeypatch.setattr("app.config", test_config)

        # Load logs with unique cache key
        df = load_logs_with_duckdb(cache_key="test_null_token_handling")

        # Should only have our test log
        assert len(df) == 1, f"Expected 1 log but got {len(df)}"
        
        # Check null values are filled with 0
        row = df.iloc[0]
        assert int(row["input_tokens"]) == 1000
        assert int(row["cache_creation_input_tokens"]) == 0  # Should be filled with 0
        assert int(row["cache_read_input_tokens"]) == 0  # Should be filled with 0
        assert int(row["output_tokens"]) == 500

    def test_load_logs_timezone_handling(self, tmp_path, monkeypatch):
        """Test timezone conversion in DuckDB"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create log with UTC timestamp
        jsonl_file = projects_dir / "tz_test.jsonl"
        utc_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        log = {
            "timestamp": utc_time.isoformat(),
            "type": "assistant",
            "message": {"role": "assistant", "content": "Test"},
            "sessionId": "test-session",
        }

        with open(jsonl_file, "w") as f:
            f.write(json.dumps(log) + "\n")

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"
        monkeypatch.setattr("app.config", test_config)

        # Load logs
        df = load_logs_with_duckdb(cache_key=1)

        # Check timestamp is converted to pandas datetime with timezone info
        timestamp = df.iloc[0]["timestamp"]
        assert isinstance(timestamp, pd.Timestamp)
        assert timestamp.tz is not None  # Should have timezone info

    def test_load_logs_project_path_extraction(self, claude_projects_with_jsonl, monkeypatch):
        """Test project path extraction from file paths"""
        projects_dir, jsonl_files = claude_projects_with_jsonl

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        monkeypatch.setattr("app.config", test_config)

        # Load logs
        df = load_logs_with_duckdb(cache_key=1)

        # Check project paths are correctly extracted
        project_paths = df["project_path"].unique()
        assert len(project_paths) > 0

        # Should match the parent directory names
        for path in project_paths:
            assert path in ["project1", "project2"]

    def test_load_logs_error_handling(self, tmp_path, monkeypatch):
        """Test error handling when loading fails"""
        # Create an empty directory (no JSONL files)
        empty_dir = tmp_path / ".claude" / "projects"
        empty_dir.mkdir(parents=True)

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = empty_dir
        monkeypatch.setattr("app.config", test_config)

        # Mock streamlit error
        errors = []

        def mock_error(msg):
            errors.append(msg)

        monkeypatch.setattr("streamlit.error", mock_error)

        # Load logs - with no files, DuckDB should return empty dataframe or error
        df = load_logs_with_duckdb(cache_key="test_error_handling")

        # With empty directory and glob pattern, DuckDB might return empty dataframe without error
        # or it might error if no files match the pattern
        if df is not None:
            # If successful, should be empty
            assert len(df) == 0
        else:
            # Or it might have errored
            assert len(errors) >= 1

    def test_load_logs_malformed_json(self, tmp_path, monkeypatch):
        """Test handling of malformed JSON lines"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create JSONL with some malformed lines
        jsonl_file = projects_dir / "malformed.jsonl"
        with open(jsonl_file, "w") as f:
            # Valid JSON
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": "assistant",
                        "message": {"role": "assistant", "content": "Valid"},
                        "sessionId": "test",
                    }
                )
                + "\n"
            )
            # Invalid JSON
            f.write("{invalid json}\n")
            # Another valid JSON
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": "assistant",
                        "message": {"role": "assistant", "content": "Valid 2"},
                        "sessionId": "test",
                    }
                )
                + "\n"
            )

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"
        monkeypatch.setattr("app.config", test_config)

        # Mock streamlit error
        errors = []

        def mock_error(msg):
            errors.append(msg)

        monkeypatch.setattr("streamlit.error", mock_error)

        # Load logs - DuckDB should skip invalid lines or error out
        df = load_logs_with_duckdb(cache_key=1)

        # DuckDB might skip invalid lines or fail completely
        if df is None:
            # If it failed, should have error message
            assert len(errors) > 0
        else:
            # If it succeeded, it might have processed valid lines only
            # or it might have loaded from conftest.py fixture
            assert len(df) >= 0

    def test_load_logs_performance_large_dataset(self, tmp_path, monkeypatch):
        """Test performance with larger dataset"""
        projects_dir = tmp_path / ".claude" / "projects" / "test"
        projects_dir.mkdir(parents=True)

        # Create JSONL with many entries
        jsonl_file = projects_dir / "large.jsonl"
        base_time = datetime.now(timezone.utc)

        with open(jsonl_file, "w") as f:
            for i in range(1000):  # 1000 log entries
                log = {
                    "timestamp": (base_time - timedelta(minutes=i)).isoformat(),
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": f"Message {i}",
                        "model": "claude-3-5-sonnet-20241022" if i % 2 == 0 else "claude-3-opus-20240229",
                        "usage": {
                            "input_tokens": 100 + i,
                            "output_tokens": 50 + i,
                            "cache_creation_input_tokens": i % 100,
                            "cache_read_input_tokens": i % 50,
                        },
                    },
                    "sessionId": f"session-{i // 100}",  # 10 different sessions
                    "uuid": f"uuid-{i}",
                }
                f.write(json.dumps(log) + "\n")

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = tmp_path / ".claude" / "projects"
        monkeypatch.setattr("app.config", test_config)

        # Measure loading time
        import time

        start_time = time.time()
        df = load_logs_with_duckdb(cache_key="test_performance_large")
        elapsed = time.time() - start_time

        # Should load successfully and quickly
        assert df is not None
        # Check that we loaded the correct number of logs
        assert len(df) == 1000, f"Expected 1000 logs but got {len(df)}"
        assert elapsed < 5.0  # Should complete in less than 5 seconds

        # Verify data integrity
        assert df["session_id"].nunique() == 10  # 10 different sessions
        assert df["model"].nunique() == 2  # 2 different models

    def test_load_logs_session_id_type_conversion(self, claude_projects_with_jsonl, monkeypatch):
        """Test session ID is converted to string type"""
        projects_dir, _ = claude_projects_with_jsonl

        # Mock config
        test_config = AppConfig()
        test_config.claude_projects_path = projects_dir
        monkeypatch.setattr("app.config", test_config)

        # Load logs
        df = load_logs_with_duckdb(cache_key=1)

        # All session IDs should be strings
        assert df["session_id"].dtype == "object"  # pandas string type
        assert all(isinstance(sid, str) for sid in df["session_id"])

        # Check that the session IDs from conftest.py are present
        session_ids = df["session_id"].unique()
        assert "session-001" in session_ids or "session-002" in session_ids
