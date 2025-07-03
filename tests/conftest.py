"""Test configuration and fixtures"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_claude_logs():
    """Sample Claude log entries for testing"""
    base_time = datetime.now(timezone.utc)
    logs = []

    # Assistant response with claude-3-5-sonnet model
    logs.append(
        {
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": "I'll help you implement this feature.",
                "model": "claude-3-5-sonnet-20241022",
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 500,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 300,
                },
            },
            "sessionId": "session-001",
            "uuid": "uuid-001",
            "parentUuid": None,
            "cwd": "/home/user/project1",
            "userType": "free",
        }
    )

    # Assistant response with claude-3-opus model
    logs.append(
        {
            "timestamp": (base_time - timedelta(hours=1)).isoformat(),
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": "Here's the analysis you requested.",
                "model": "claude-3-opus-20240229",
                "usage": {
                    "input_tokens": 2000,
                    "output_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 500,
                },
            },
            "sessionId": "session-002",
            "uuid": "uuid-002",
            "parentUuid": None,
            "cwd": "/home/user/project2",
            "userType": "pro",
        }
    )

    # Assistant response with unknown model
    logs.append(
        {
            "timestamp": base_time.isoformat(),
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": "Testing unknown model handling.",
                "model": "claude-unknown-model",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 0,
                },
            },
            "sessionId": "session-001",
            "uuid": "uuid-003",
            "parentUuid": "uuid-001",
            "cwd": "/home/user/project1",
            "userType": "free",
        }
    )

    # Human message (should be filtered out)
    logs.append(
        {
            "timestamp": base_time.isoformat(),
            "type": "human",
            "message": {
                "role": "human",
                "content": "Please help me with this code.",
            },
            "sessionId": "session-001",
            "uuid": "uuid-004",
            "parentUuid": "uuid-003",
            "cwd": "/home/user/project1",
            "userType": "free",
        }
    )

    return logs


@pytest.fixture
def jsonl_file_with_logs(tmp_path: Path, sample_claude_logs: list[dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with sample logs"""
    file_path = tmp_path / "test_logs.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for log in sample_claude_logs:
            f.write(json.dumps(log) + "\n")
    return file_path


@pytest.fixture
def claude_projects_dir(tmp_path: Path) -> Path:
    """Create a temporary Claude projects directory structure"""
    projects_dir = tmp_path / ".claude" / "projects"
    projects_dir.mkdir(parents=True)

    # Create some project directories with JSONL files
    project1 = projects_dir / "project1"
    project1.mkdir()
    project2 = projects_dir / "project2"
    project2.mkdir()

    return projects_dir


@pytest.fixture
def claude_projects_with_jsonl(
    claude_projects_dir: Path, sample_claude_logs: list[dict[str, Any]]
) -> tuple[Path, list[Path]]:
    """Create Claude projects directory with JSONL files"""
    jsonl_files = []

    # Create JSONL files in different projects
    for i, project_name in enumerate(["project1", "project2"]):
        project_dir = claude_projects_dir / project_name
        jsonl_file = project_dir / f"logs_{i}.jsonl"

        # Write some logs to each file
        with open(jsonl_file, "w", encoding="utf-8") as f:
            # Take different logs for each project
            logs_to_write = (
                sample_claude_logs[i : i + 2] if i < len(sample_claude_logs) - 1 else [sample_claude_logs[-1]]
            )
            for log in logs_to_write:
                f.write(json.dumps(log) + "\n")

        jsonl_files.append(jsonl_file)

    return claude_projects_dir, jsonl_files


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture for mocking environment variables"""

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, str(value))

    return _set_env


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for log files"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir
