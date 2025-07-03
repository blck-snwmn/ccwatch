"""Helper functions for generating test log data"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any


def generate_claude_log(
    timestamp: datetime | None = None,
    message_type: str = "assistant",
    model: str = "claude-3-5-sonnet-20241022",
    session_id: str = "test-session",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cache_creation_tokens: int = 100,
    cache_read_tokens: int = 200,
    content: str = "Test response",
    cwd: str = "/home/user/project",
    user_type: str = "free",
    uuid: str | None = None,
    parent_uuid: str | None = None,
) -> dict[str, Any]:
    """Generate a single Claude log entry"""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if uuid is None:
        uuid = f"uuid-{random.randint(1000, 9999)}"

    log = {
        "timestamp": timestamp.isoformat(),
        "type": message_type,
        "sessionId": session_id,
        "uuid": uuid,
        "cwd": cwd,
        "userType": user_type,
    }

    if parent_uuid:
        log["parentUuid"] = parent_uuid

    if message_type == "assistant":
        log["message"] = {
            "role": "assistant",
            "content": content,
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_tokens,
                "cache_read_input_tokens": cache_read_tokens,
            },
        }
    elif message_type == "human":
        log["message"] = {
            "role": "human",
            "content": content,
        }
    else:
        log["message"] = {
            "role": message_type,
            "content": content,
        }

    return log


def generate_session_logs(
    session_id: str,
    num_messages: int = 10,
    start_time: datetime | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    project_path: str = "/home/user/project",
) -> list[dict[str, Any]]:
    """Generate a complete session with multiple messages"""
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)

    logs = []
    current_time = start_time
    parent_uuid = None

    for i in range(num_messages):
        # Alternate between human and assistant messages
        if i % 2 == 0:
            # Human message
            human_log = generate_claude_log(
                timestamp=current_time,
                message_type="human",
                session_id=session_id,
                content=f"User question {i // 2 + 1}",
                cwd=project_path,
                uuid=f"{session_id}-{i}",
                parent_uuid=parent_uuid,
            )
            logs.append(human_log)
            parent_uuid = human_log["uuid"]
        else:
            # Assistant message
            assistant_log = generate_claude_log(
                timestamp=current_time,
                message_type="assistant",
                model=model,
                session_id=session_id,
                input_tokens=random.randint(500, 2000),
                output_tokens=random.randint(200, 1000),
                cache_creation_tokens=random.randint(0, 500),
                cache_read_tokens=random.randint(0, 1000),
                content=f"Assistant response {i // 2 + 1}",
                cwd=project_path,
                uuid=f"{session_id}-{i}",
                parent_uuid=parent_uuid,
            )
            logs.append(assistant_log)
            parent_uuid = assistant_log["uuid"]

        # Add time between messages
        current_time += timedelta(seconds=random.randint(5, 60))

    return logs


def generate_multi_project_logs(
    num_projects: int = 3,
    sessions_per_project: int = 2,
    messages_per_session: int = 5,
    start_time: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate logs for multiple projects"""
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(days=7)

    projects_logs = {}
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
    ]

    current_time = start_time

    for project_idx in range(num_projects):
        project_name = f"project{project_idx + 1}"
        project_path = f"/home/user/{project_name}"
        project_logs = []

        for session_idx in range(sessions_per_project):
            session_id = f"{project_name}-session-{session_idx}"
            # Use different models for variety
            model = models[session_idx % len(models)]

            session_logs = generate_session_logs(
                session_id=session_id,
                num_messages=messages_per_session,
                start_time=current_time,
                model=model,
                project_path=project_path,
            )

            project_logs.extend(session_logs)

            # Add time between sessions
            current_time += timedelta(hours=random.randint(1, 12))

        projects_logs[project_name] = project_logs

    return projects_logs


def generate_cost_test_logs() -> list[dict[str, Any]]:
    """Generate logs specifically for testing cost calculations"""
    logs = []
    base_time = datetime.now(timezone.utc)

    # High token count log (expensive)
    logs.append(
        generate_claude_log(
            timestamp=base_time - timedelta(hours=3),
            model="claude-3-opus-20240229",  # Most expensive model
            input_tokens=100000,
            output_tokens=50000,
            cache_creation_tokens=20000,
            cache_read_tokens=10000,
            content="Large response with many tokens",
        )
    )

    # Low token count log (cheap)
    logs.append(
        generate_claude_log(
            timestamp=base_time - timedelta(hours=2),
            model="claude-3-haiku-20240307",  # Cheapest model
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            content="Small response",
        )
    )

    # Cache-heavy log
    logs.append(
        generate_claude_log(
            timestamp=base_time - timedelta(hours=1),
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=5000,  # High cache creation
            cache_read_tokens=10000,  # High cache read
            content="Cache-optimized response",
        )
    )

    # Zero cost log (all zeros)
    logs.append(
        generate_claude_log(
            timestamp=base_time,
            model="claude-3-5-sonnet-20241022",
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            content="Empty token usage",
        )
    )

    return logs


def generate_malformed_logs() -> list[str]:
    """Generate malformed log entries for error testing"""
    malformed = []

    # Valid JSON but missing required fields
    malformed.append('{"timestamp": "2024-01-01T00:00:00Z"}')

    # Invalid JSON
    malformed.append("{invalid json content}")

    # Valid JSON but wrong structure
    malformed.append('{"data": [1, 2, 3]}')

    # Empty object
    malformed.append("{}")

    # Null values
    malformed.append('{"timestamp": null, "type": null}')

    return malformed
